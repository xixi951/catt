"""Fetch the training-side fixture for the SW Ranking router.

The SW Ranking router (``router_sw_qwen3.py``) needs ~600-1000 labelled
"battles" to learn cluster-level preferences for thinking-on vs
thinking-off. Each battle = (prompt, model_a_won_or_b_won_or_tie).

This fetcher pulls the *training-side* prompt set (no labels yet); a
separate driver (``scripts/run_battle_labels_vllm.sh``) then runs both
thinking modes through vLLM Qwen3-32B and scores the outputs to derive
the winner column.

Holdout-safety: every source here is **disjoint from the paper-main
fixture** (``router_sweep_questions_paper_main.jsonl``):

* AIME 2022, 2023, 2024, 2025 (the paper-main holdout uses 2026)
* GPQA main + extended **minus** the 198 Diamond rows
* MMLU-Pro (stratified across all 14 subjects; the paper holdout uses
  VoiceBench MMSU which is a different dataset)
* AlpacaEval-original (805 prompts; paper-main's VoiceBench
  AlpacaEval mirrors a 199-row subset, the IDs are not preserved
  upstream so we just shrink to a train-only sample of 200 here and
  rely on the natural disjointness)
* GSM8K train split (the paper holdout doesn't use GSM8K)

Default total: 30 + 30 + 30 + 30 (AIME 22/23/24/25) + 200 (GPQA non-Diamond)
+ 200 (MMLU-Pro) + 200 (AlpacaEval) + 100 (GSM8K) = 820 prompts.

Output schema is the same JSONL row shape the paper-main fixture uses
(id / source / subject / difficulty_hint / question / answer /
raw_problem). After this fetcher finishes, run the battle-label driver:

    bash scripts/run_battle_labels_vllm.sh \\
        --train-fixture scripts/data/router_train_questions.jsonl \\
        --output-csv data/router/battles.csv

Usage::

    PYTHONPATH=src python scripts/fetch_router_train_questions.py \\
        --output scripts/data/router_train_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_OUTPUT = Path("scripts/data/router_train_questions.jsonl")
DEFAULT_SEED = 17  # different seed from the paper-main fixture (42)


# AIME splits per year. MathArena publishes per-paper datasets:
# ``aime_<year>_I`` and ``aime_<year>_II`` (15 problems each); plus
# a combined ``aime_<year>`` for some years. We use the paper-level
# datasets for 2023-2025 because they are reliably present.
# (No aime_2022 on MathArena; aime_2026 is reserved for the test
# fixture.)
AIME_YEAR_DATASETS: List[str] = [
    "aime_2023_I", "aime_2023_II",
    "aime_2024_I", "aime_2024_II",
    "aime_2025_I", "aime_2025_II",
]
DEFAULT_AIME_PER_YEAR = 15  # one paper per dataset is 15 problems
DEFAULT_GPQA_NON_DIAMOND_COUNT = 200
DEFAULT_MMLU_PRO_COUNT = 200
DEFAULT_ALPACA_EVAL_COUNT = 200
DEFAULT_GSM8K_COUNT = 100

# Math: same prompt suffix as the paper-main AIME fetcher so the model
# emits ``\\boxed{}`` consistently across train + test.
_MATH_PROMPT_SUFFIX = (
    "\n\nPlease reason step by step, then put your final answer in "
    "\\boxed{...}."
)
# MCQ: same prompt suffix as the paper-main GPQA fetcher.
_MCQ_PROMPT_SUFFIX_4 = (
    "\n\nPlease reason step by step, then conclude your response with "
    '"The answer is X." where X is one of A, B, C, or D.'
)


# ---- AIME 22/23/24/25 -------------------------------------------------

def fetch_aime_dataset(dataset_name: str, count: int) -> List[Dict]:
    """Pull up to ``count`` rows from MathArena/<dataset_name>.

    ``dataset_name`` is e.g. ``aime_2024_I`` or ``aime_2025_II``. The
    schema mirrors aime_2026: ``problem`` (str), ``answer`` (str,
    integer 0-999). We keep the same id / format conventions so
    train/test rows share a schema.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(f"MathArena/{dataset_name}", split="train")
    except Exception as exc:  # noqa: BLE001
        print(
            f"warning: failed to load MathArena/{dataset_name}: {exc!r}",
            file=sys.stderr,
        )
        return []
    rows: List[Dict] = []
    for idx in range(min(count, len(ds))):
        row = ds[idx]
        problem = row["problem"]
        answer = str(row["answer"]).strip()
        question = (
            "Solve the following math problem step by step. Put "
            "your answer inside \\boxed{}.\n\n"
            f"{problem}\n\nRemember to put your answer inside "
            "\\boxed{}."
        )
        rec: Dict = {
            "id": f"{dataset_name}/{idx + 1:02d}",
            "source": dataset_name,
            "subject": "math",
            "difficulty_hint": "competition",
            "question": question,
            "answer": answer,
            "raw_problem": problem,
        }
        try:
            rec["answer_int"] = int(answer)
        except ValueError:
            pass
        rows.append(rec)
    return rows


# ---- GPQA non-Diamond (main + extended minus diamond) -----------------

def fetch_gpqa_non_diamond(count: int, rng: random.Random) -> List[Dict]:
    """Pull GPQA main + extended rows that are NOT in Diamond.

    Diamond is a 198-question subset of main + extended. We load all
    three configs and exclude any row whose ``Record ID`` appears in
    Diamond. The remainder is shuffled and we take ``count`` rows.

    Format conversion: same MCQ A-D shuffle as the paper-main fetcher
    (random option order via ``rng``).
    """
    from datasets import load_dataset

    diamond = load_dataset(
        "Wanfq/gpqa", "gpqa_diamond", split="train",
    )
    diamond_ids = {row["Record ID"] for row in diamond}

    pool: List[Dict] = []
    for cfg in ("gpqa_main", "gpqa_extended"):
        ds = load_dataset("Wanfq/gpqa", cfg, split="train")
        for row in ds:
            if row["Record ID"] in diamond_ids:
                continue
            pool.append(row)

    rng.shuffle(pool)
    chosen = pool[: min(count, len(pool))]
    rows: List[Dict] = []
    for row in chosen:
        question_stem = (row.get("Question") or "").strip()
        if not question_stem:
            continue
        correct = str(row["Correct Answer"]).strip()
        incorrects = [
            str(row[f"Incorrect Answer {i}"]).strip()
            for i in (1, 2, 3)
        ]
        # MCQ A-D shuffle via the same helper logic as paper-main
        # (replicated inline to avoid cross-file imports)
        options = [(correct, True)] + [(opt, False) for opt in incorrects]
        rng.shuffle(options)
        letters = "ABCD"
        lines = []
        correct_letter = ""
        for letter, (text, is_correct) in zip(letters, options):
            lines.append(f"{letter}) {text.strip()}")
            if is_correct:
                correct_letter = letter
        formatted = (
            f"{question_stem}\n\n"
            + "\n".join(lines)
            + _MCQ_PROMPT_SUFFIX_4
        )
        domain = row.get("High-level domain") or "unknown"
        rows.append(
            {
                "id": f"gpqa_non_diamond/{domain.lower()}/"
                      f"{row['Record ID']}",
                "source": "gpqa_non_diamond",
                "subject": domain,
                "difficulty_hint": "graduate",
                "question": formatted,
                "answer": correct_letter,
                "raw_problem": question_stem,
            }
        )
    return rows


# ---- MMLU-Pro (full subjects, stratified) -----------------------------

def fetch_mmlu_pro(count: int, rng: random.Random) -> List[Dict]:
    """Stratified-sample ``count`` rows from TIGER-Lab/MMLU-Pro.

    14 subjects present in MMLU-Pro; we draw approximately ``count /
    14`` per subject (rounded up), then truncate to ``count``.
    """
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    by_subject: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(ds):
        by_subject[row["category"]].append(i)
    subjects = list(by_subject.keys())
    per_subject = max(1, (count // len(subjects)) + 1)

    pool: List[Dict] = []
    for subject, indices in by_subject.items():
        rng.shuffle(indices)
        chosen = indices[: min(per_subject, len(indices))]
        for idx in chosen:
            row = ds[idx]
            options = list(row["options"])
            letters = "ABCDEFGHIJ"[: len(options)]
            choices = ", ".join(letters[:-1]) + ", or " + letters[-1] \
                if len(letters) > 2 else " or ".join(letters)
            option_lines = "\n".join(
                f"{letter}) {opt}"
                for letter, opt in zip(letters, options)
            )
            suffix = (
                "\n\nPlease reason step by step, then conclude your "
                f'response with "The answer is X." where X is one of '
                f"{choices}."
            )
            question = f"{row['question']}\n{option_lines}{suffix}"
            pool.append(
                {
                    "id": f"mmlu_pro/{row['category']}/"
                          f"{row['question_id']:06d}",
                    "source": "mmlu_pro",
                    "subject": row["category"],
                    "difficulty_hint": "medium",
                    "question": question,
                    "answer": row["answer"],
                    "raw_problem": row["question"],
                }
            )
    rng.shuffle(pool)
    return pool[: min(count, len(pool))]


# ---- AlpacaEval (original 805 prompts) --------------------------------

def fetch_alpaca_eval(count: int, rng: random.Random) -> List[Dict]:
    """Random-sample ``count`` open-ended instructions from
    yahma/alpaca-cleaned (a 52K instruction-tuning corpus that mirrors
    the AlpacaEval domain — daily-life / how-to / open-ended chat).

    Note: ``tatsu-lab/alpaca_eval`` is the canonical 805-prompt eval
    set but it ships as a Python script-loader which the current HF
    ``datasets`` library no longer supports (script datasets were
    deprecated in 2025). yahma/alpaca-cleaned is the parquet fork
    of the underlying 52K Stanford Alpaca instruction set; for our
    purpose (router-training open-ended exemplars) the broader
    52K pool is in fact better than the 805-prompt eval slice
    because it covers more topical diversity.

    Filter: skip rows with non-empty ``input`` (those are templated
    inputs that are off-distribution from voice-assistant chat).
    """
    from datasets import load_dataset

    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    eligible = [
        i for i, row in enumerate(ds)
        if not (row.get("input") or "").strip()
    ]
    rng.shuffle(eligible)
    chosen = eligible[: min(count, len(eligible))]
    rows: List[Dict] = []
    for n, idx in enumerate(chosen):
        row = ds[idx]
        prompt = row["instruction"].strip()
        rows.append(
            {
                "id": f"alpaca_eval_train/{n:03d}",
                "source": "alpaca_eval_train",
                "subject": "general",
                "difficulty_hint": "open_ended",
                "question": prompt,
                "answer": prompt,  # placeholder; judge-scored
                "raw_problem": prompt,
            }
        )
    return rows


# ---- GSM8K --------------------------------------------------------

def fetch_gsm8k(count: int, rng: random.Random) -> List[Dict]:
    """Random-sample ``count`` GSM8K train rows.

    GSM8K uses ``####`` as the answer trigger; we strip it at fetch
    time so the fixture's ``answer`` column carries a bare numeric
    string.
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[: min(count, len(indices))]
    rows: List[Dict] = []
    for n, idx in enumerate(chosen):
        row = ds[idx]
        question_stem = row["question"].strip()
        question = (
            f"{question_stem}\n\nPlease reason step by step. Then "
            "give the final numeric answer after #### , for example "
            "#### 42."
        )
        raw_answer = row["answer"]
        if "####" in raw_answer:
            numeric = raw_answer.split("####", 1)[1].strip()
        else:
            numeric = raw_answer.strip()
        rows.append(
            {
                "id": f"gsm8k_train/{n:04d}",
                "source": "gsm8k_train",
                "subject": "math",
                "difficulty_hint": "elementary",
                "question": question,
                "answer": numeric.replace(",", ""),
                "raw_problem": question_stem,
            }
        )
    return rows


# ---- IO -----------------------------------------------------------

def write_jsonl(rows: Iterable[Dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--aime-per-year", type=int, default=DEFAULT_AIME_PER_YEAR,
        help=(
            "How many problems per AIME year (22/23/24/25). "
            f"Default: {DEFAULT_AIME_PER_YEAR}."
        ),
    )
    p.add_argument(
        "--gpqa-non-diamond-count", type=int,
        default=DEFAULT_GPQA_NON_DIAMOND_COUNT,
    )
    p.add_argument(
        "--mmlu-pro-count", type=int, default=DEFAULT_MMLU_PRO_COUNT,
    )
    p.add_argument(
        "--alpaca-eval-count", type=int, default=DEFAULT_ALPACA_EVAL_COUNT,
    )
    p.add_argument(
        "--gsm8k-count", type=int, default=DEFAULT_GSM8K_COUNT,
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    print(
        f"fetching {args.aime_per_year} per AIME paper "
        f"(2023/2024/2025 I+II = up to {6 * args.aime_per_year}) + "
        f"{args.gpqa_non_diamond_count} GPQA non-Diamond + "
        f"{args.mmlu_pro_count} MMLU-Pro + "
        f"{args.alpaca_eval_count} AlpacaEval + "
        f"{args.gsm8k_count} GSM8K",
        file=sys.stderr,
    )

    rows: List[Dict] = []
    for ds_name in AIME_YEAR_DATASETS:
        rows.extend(fetch_aime_dataset(ds_name, args.aime_per_year))
    rows.extend(fetch_gpqa_non_diamond(args.gpqa_non_diamond_count, rng))
    rows.extend(fetch_mmlu_pro(args.mmlu_pro_count, rng))
    rows.extend(fetch_alpaca_eval(args.alpaca_eval_count, rng))
    rows.extend(fetch_gsm8k(args.gsm8k_count, rng))

    written = write_jsonl(rows, args.output)
    by_source = Counter(r["source"] for r in rows)
    print(f"wrote {written} rows to {args.output}", file=sys.stderr)
    for source, n in by_source.most_common():
        print(f"  {source}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
