"""Fetch the paper's main fixture: AIME 2026 + GPQA Diamond + VoiceBench
(AlpacaEval + MMSU).

This is a one-shot utility: run on a machine with HuggingFace ``datasets``
plus internet access (it downloads from the Hub on first run, then caches),
redirect output to ``scripts/data/router_sweep_questions_paper_main.jsonl``,
and commit the JSONL. The driver scripts (``baseline_direct_llm.py`` and
``sweep_mid_thresholds.py``) read the static file at experiment time, so
the network is never hit during a sweep.

The fixture composition (and reasons):

* **30 x AIME 2026** (``MathArena/aime_2026``): 30 competition-grade
  integer-answer math problems (0-999). Answer extraction via
  ``\\boxed{N}``; SymPy-equivalence fallback handles e.g.
  ``\\boxed{m+n}`` against bare ``277``. The fixture mirrors the
  pre-existing R23 ``router_sweep_questions_aime2026_baseline.jsonl``
  for compatibility; we do NOT shuffle the AIME 26 set, both for
  reproducibility against earlier streaming-thinker smoke runs and
  because n=30 is too small for sampling to add value.

* **GPQA Diamond** (``Wanfq/gpqa`` config ``gpqa_diamond``, 198 rows):
  Graduate-level multiple choice across Chemistry/Physics/Biology.
  Source has imbalanced domain counts (Chem 93, Phys 86, Bio 19).
  Default pull: 30 Chemistry + 30 Physics + 19 Biology = 79 (slightly
  shy of the originally-targeted 90 but preserves Diamond's natural
  Bio cap). Each row is converted to MCQ A-D via
  ``shuffle([correct, incorrect_1, incorrect_2, incorrect_3])``;
  the seed (``--seed 42`` by default) makes the option ordering
  reproducible. Scored via ``score_mmlu`` (regex extracts A-D).

* **50 x VoiceBench AlpacaEval** (``hlt-lab/voicebench`` config
  ``alpacaeval``, 199 rows): Open-ended chat queries. We drop the
  audio column (transcript-only routing experiment) and randomly
  sample 50. Scoring is open-ended; the deterministic
  ``score_results.py`` path returns
  ``unscorable/needs_alpaca_judge``. A separate offline pass
  (``scripts/score_alpaca_judge.py``) calls DashScope qwen-max with
  the VoiceBench-style 1-5 rubric and fills the ``correct``
  column.

* **50 x VoiceBench MMSU** (``hlt-lab/voicebench`` config ``mmsu``):
  Multiple-choice spoken-understanding test across ~12 subjects
  (chemistry, physics, biology, business, economics, health,
  philosophy, psychology, history, engineering, law, other). Each
  ``prompt`` already contains the full question + A-D options; the
  ``reference`` field is the correct letter. We stratify-sample 50
  across subjects, taking at most ``ceil(50 / num_subjects) + 1``
  per subject before truncating. Scored via ``score_mmlu``.

Usage::

    PYTHONPATH=src python scripts/fetch_paper_main_questions.py \\
        --output scripts/data/router_sweep_questions_paper_main.jsonl

Output schema (one JSONL line per question)::

    {
      "id": "<source>/<idx>",
      "source": "aime_2026"
                | "gpqa_diamond"
                | "voicebench_alpaca_eval"
                | "voicebench_mmsu",
      "subject": "math" | "Chemistry" | "Physics" | "Biology"
                 | "general" | <mmsu category>,
      "difficulty_hint": "competition" | "graduate"
                         | "open_ended" | "spoken_mcq",
      "question": "<the prompt sent to the model>",
      "answer": "A" | "B" | "C" | "D" | "<integer>" | <mmsu reference>
                | "<bare alpacaeval prompt — open-ended marker>",
      "raw_problem": <optional original-problem text for math>
    }

Pre-conditions on the target machine:

- ``pip install datasets``
- HuggingFace cache writable (default ``~/.cache/huggingface``)
- For AIME 2026: ``MathArena/aime_2026`` is gated (HF login required).
  The HuggingFace ``token`` cache file at ``~/.cache/huggingface/token``
  is auto-detected.
- For GPQA Diamond via ``Wanfq/gpqa``: public mirror, no gating; do
  NOT use ``Idavidrein/gpqa`` (gated, requires manual access request).
- For VoiceBench: ``hlt-lab/voicebench``, public.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- defaults -----------------------------------------------------

DEFAULT_OUTPUT = Path("scripts/data/router_sweep_questions_paper_main.jsonl")
DEFAULT_SEED = 42

DEFAULT_AIME_COUNT = 30  # full AIME 2026 fixture
DEFAULT_GPQA_DIAMOND_PER_DOMAIN: Dict[str, int] = {
    # GPQA Diamond has Chem 93 + Phys 86 + Bio 19 = 198. The user-
    # targeted 90-question slice (30 per domain) hits the Biology
    # cap (only 19 available). Per the user's "Biology 不够的剩下的
    # 补" instruction, we keep all 19 Biology and reallocate the 11
    # missing rows across Chem + Phys proportionally to availability
    # (93:86 ≈ 6:5), giving Chem 36 + Phys 35 + Bio 19 = 90 total.
    "Chemistry": 36,
    "Physics": 35,
    "Biology": 19,
}
DEFAULT_ALPACA_EVAL_COUNT = 50
DEFAULT_MMSU_COUNT = 50

# Prompts mirror the per-source CoT / formatting protocol that the
# scorer expects. Keep these in sync with ``scripts/score_results.py``
# extractors.
_AIME_PROMPT_SUFFIX = (
    "\n\nPlease reason step by step, then put your final answer in "
    "\\boxed{...}."
)
_GPQA_DIAMOND_PROMPT_SUFFIX = (
    "\n\nPlease reason step by step, then conclude your response with "
    '"The answer is X." where X is one of A, B, C, or D.'
)
# VoiceBench AlpacaEval / MMSU prompts are passed through verbatim:
# AlpacaEval is open-ended; MMSU has options inlined in ``prompt``.

# --- AIME 2026 ----------------------------------------------------

def fetch_aime_2026(count: int) -> List[Dict]:
    """Pull ``count`` rows from MathArena/aime_2026.

    Each problem is integer-answered (0-999). The model is prompted
    to reason step-by-step and box the final integer; the scorer
    (``score_omni_math`` via ``aime_2026`` branch) extracts the last
    ``\\boxed{...}`` payload and falls back to SymPy on string
    mismatch.

    Returns rows in the universal fixture schema. AIME 26 is taken
    in source order (no shuffle) for reproducibility against
    pre-existing AIME smoke runs.
    """
    from datasets import load_dataset

    ds = load_dataset("MathArena/aime_2026", split="train")
    rows: List[Dict] = []
    for idx in range(min(count, len(ds))):
        row = ds[idx]
        problem = row["problem"]
        answer = str(row["answer"]).strip()
        try:
            answer_int = int(answer)
        except ValueError:
            answer_int = None
        question = (
            "Solve the following math problem step by step. Put "
            "your answer inside \\boxed{}.\n\n"
            f"{problem}\n\nRemember to put your answer inside "
            "\\boxed{}."
        )
        rec = {
            "id": f"aime_2026/{idx + 1:02d}",
            "source": "aime_2026",
            "subject": "math",
            "difficulty_hint": "competition",
            "question": question,
            "answer": answer,
            "raw_problem": problem,
        }
        if answer_int is not None:
            rec["answer_int"] = answer_int
        rows.append(rec)
    return rows


# --- GPQA Diamond -------------------------------------------------

def _gpqa_format_mcq(
    question_stem: str,
    correct: str,
    incorrects: List[str],
    rng: random.Random,
) -> Tuple[str, str]:
    """Build an A-D MCQ string + correct letter from a stem and
    one correct + three incorrect answers.

    Order is randomised via ``rng`` so the correct letter is uniform
    across the dataset. Returns (formatted_question, correct_letter).
    """
    options = [(correct, True)] + [(opt, False) for opt in incorrects]
    rng.shuffle(options)
    letters = "ABCD"
    lines = []
    correct_letter = ""
    for letter, (text, is_correct) in zip(letters, options):
        # Strip trailing whitespace / artefacts from raw GPQA strings
        # (some incorrect answers carry stray newlines).
        cleaned = text.strip()
        lines.append(f"{letter}) {cleaned}")
        if is_correct:
            correct_letter = letter
    body = (
        f"{question_stem.strip()}\n\n"
        + "\n".join(lines)
        + _GPQA_DIAMOND_PROMPT_SUFFIX
    )
    return body, correct_letter


def fetch_gpqa_diamond(
    per_domain: Dict[str, int],
    rng: random.Random,
) -> List[Dict]:
    """Pull stratified GPQA Diamond rows.

    ``per_domain`` keys are GPQA's "High-level domain" values
    (Chemistry / Physics / Biology). Values are the desired sample
    sizes; the actual count is ``min(requested, available)``.

    Each row's options are shuffled to MCQ A-D form.
    """
    from datasets import load_dataset

    # Wanfq/gpqa is a non-gated mirror that preserves the original
    # column layout.
    ds = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train")

    by_domain: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        by_domain[row["High-level domain"]].append(i)

    rows: List[Dict] = []
    for domain, want in per_domain.items():
        indices = by_domain.get(domain, [])
        if not indices:
            print(
                f"warning: GPQA Diamond has no rows for domain "
                f"{domain!r}; skipping",
                file=sys.stderr,
            )
            continue
        rng.shuffle(indices)
        chosen = indices[: min(want, len(indices))]
        for idx in chosen:
            row = ds[idx]
            question_stem = row["Question"].strip()
            correct = str(row["Correct Answer"]).strip()
            incorrects = [
                str(row[f"Incorrect Answer {i}"]).strip()
                for i in (1, 2, 3)
            ]
            formatted, letter = _gpqa_format_mcq(
                question_stem, correct, incorrects, rng,
            )
            rows.append(
                {
                    "id": f"gpqa_diamond/{domain.lower()}/"
                          f"{row['Record ID']}",
                    "source": "gpqa_diamond",
                    "subject": domain,
                    "difficulty_hint": "graduate",
                    "question": formatted,
                    "answer": letter,
                    "raw_problem": question_stem,
                }
            )
    return rows


# --- VoiceBench AlpacaEval ----------------------------------------

def fetch_voicebench_alpaca_eval(
    count: int, rng: random.Random,
) -> List[Dict]:
    """Random-sample ``count`` AlpacaEval prompts from VoiceBench.

    VoiceBench AlpacaEval has 199 rows; each row carries a single
    ``prompt`` (the user's request, transcribed). Open-ended; the
    fixture's ``answer`` field stores the prompt itself as a
    placeholder, since the deterministic scorer cannot grade open
    text. The downstream qwen-max judge reads the prompt to score
    the model's output 1-5; the judge writes the score back into a
    separate scored CSV (see ``scripts/score_alpaca_judge.py``).
    """
    from datasets import load_dataset

    ds = load_dataset("hlt-lab/voicebench", "alpacaeval", split="test")
    audio_cols = [c for c in ds.column_names if c == "audio"]
    if audio_cols:
        ds = ds.remove_columns(audio_cols)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[: min(count, len(indices))]
    rows: List[Dict] = []
    for n, idx in enumerate(chosen):
        row = ds[idx]
        prompt = row["prompt"].strip()
        rows.append(
            {
                "id": f"voicebench_alpaca_eval/{n:03d}",
                "source": "voicebench_alpaca_eval",
                "subject": "general",
                "difficulty_hint": "open_ended",
                "question": prompt,
                # Placeholder; real scoring is via judge pass.
                "answer": prompt,
                "raw_problem": prompt,
            }
        )
    return rows


# --- VoiceBench MMSU ----------------------------------------------

def _voicebench_mmsu_subject_splits() -> List[str]:
    """The split names declared by ``hlt-lab/voicebench`` mmsu config.

    These map 1-to-1 to MMSU subjects. We hard-code the list to avoid
    pulling all splits eagerly when the user only wants 50 rows.
    """
    return [
        "law",
        "engineering",
        "other",
        "biology",
        "business",
        "economics",
        "health",
        "philosophy",
        "psychology",
        "history",
        "chemistry",
        "physics",
    ]


def fetch_voicebench_mmsu(count: int, rng: random.Random) -> List[Dict]:
    """Stratified-sample ``count`` MMSU rows across all 12 subjects.

    Per-subject quota is ``ceil(count / num_subjects)`` rounded up by
    one to absorb rounding loss; the union is then truncated and
    shuffled to ``count``. Each row's ``prompt`` already contains the
    question + A-D options inline; the ``reference`` field is the
    correct letter. We pass these through verbatim.
    """
    from datasets import load_dataset

    subjects = _voicebench_mmsu_subject_splits()
    per_subject = max(1, math.ceil(count / len(subjects)) + 1)
    pool: List[Dict] = []
    for subject in subjects:
        try:
            ds = load_dataset(
                "hlt-lab/voicebench", "mmsu", split=subject,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"warning: failed to load MMSU/{subject}: {exc}",
                file=sys.stderr,
            )
            continue
        audio_cols = [c for c in ds.column_names if c == "audio"]
        if audio_cols:
            ds = ds.remove_columns(audio_cols)
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        chosen = indices[: min(per_subject, len(indices))]
        for idx in chosen:
            row = ds[idx]
            prompt = row["prompt"].strip()
            ref = str(row["reference"]).strip().upper()
            qid = row.get("question_id", f"mmsu_{subject}_{idx:04d}")
            pool.append(
                {
                    "id": f"voicebench_mmsu/{subject}/{qid}",
                    "source": "voicebench_mmsu",
                    "subject": subject,
                    "difficulty_hint": "spoken_mcq",
                    "question": prompt,
                    "answer": ref,
                    "raw_problem": prompt,
                }
            )
    rng.shuffle(pool)
    return pool[: min(count, len(pool))]


# --- IO helpers ---------------------------------------------------

def write_jsonl(rows: Iterable[Dict], path: Path) -> int:
    """Write rows to ``path`` as one JSON object per line.

    Returns the number of rows written. Path's parent directory is
    created if missing. Existing file is overwritten.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"JSONL output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for stratified sampling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--aime-count", type=int, default=DEFAULT_AIME_COUNT,
        help=f"How many AIME 2026 problems (default: {DEFAULT_AIME_COUNT})",
    )
    parser.add_argument(
        "--gpqa-diamond-chemistry", type=int,
        default=DEFAULT_GPQA_DIAMOND_PER_DOMAIN["Chemistry"],
        help="GPQA Diamond Chemistry sample (default 30)",
    )
    parser.add_argument(
        "--gpqa-diamond-physics", type=int,
        default=DEFAULT_GPQA_DIAMOND_PER_DOMAIN["Physics"],
        help="GPQA Diamond Physics sample (default 30)",
    )
    parser.add_argument(
        "--gpqa-diamond-biology", type=int,
        default=DEFAULT_GPQA_DIAMOND_PER_DOMAIN["Biology"],
        help="GPQA Diamond Biology sample (default 19, the cap).",
    )
    parser.add_argument(
        "--alpaca-eval-count", type=int, default=DEFAULT_ALPACA_EVAL_COUNT,
        help=f"VoiceBench AlpacaEval sample (default {DEFAULT_ALPACA_EVAL_COUNT})",
    )
    parser.add_argument(
        "--mmsu-count", type=int, default=DEFAULT_MMSU_COUNT,
        help=f"VoiceBench MMSU sample (default {DEFAULT_MMSU_COUNT})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    print(
        f"fetching {args.aime_count} AIME 2026 + GPQA Diamond"
        f" ({args.gpqa_diamond_chemistry} chem + "
        f"{args.gpqa_diamond_physics} phys + "
        f"{args.gpqa_diamond_biology} bio) + "
        f"{args.alpaca_eval_count} VoiceBench/AlpacaEval + "
        f"{args.mmsu_count} VoiceBench/MMSU",
        file=sys.stderr,
    )

    rows: List[Dict] = []
    rows.extend(fetch_aime_2026(args.aime_count))
    rows.extend(
        fetch_gpqa_diamond(
            {
                "Chemistry": args.gpqa_diamond_chemistry,
                "Physics": args.gpqa_diamond_physics,
                "Biology": args.gpqa_diamond_biology,
            },
            rng,
        )
    )
    rows.extend(fetch_voicebench_alpaca_eval(args.alpaca_eval_count, rng))
    rows.extend(fetch_voicebench_mmsu(args.mmsu_count, rng))

    written = write_jsonl(rows, args.output)
    by_source = Counter(r["source"] for r in rows)
    print(
        f"wrote {written} rows to {args.output}",
        file=sys.stderr,
    )
    for source, n in by_source.items():
        print(f"  {source}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
