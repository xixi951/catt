"""Build the 'train' fixture for DuetTalk threshold validation.

Mirrors the schema of router_sweep_questions_paper_main.jsonl (the 220Q
test set), but uses DIFFERENT source data so the train and test sets do
not overlap:

  - AIME 2024 (30) + AIME 2025 (30) = 60 math problems    [different years from test's AIME 2026]
  - GPQA Diamond, with paper_main IDs excluded (~100)     [different sample of same dataset]
  - VoiceBench AlpacaEval, with paper_main IDs excluded (~100)
  - VoiceBench MMSU, with paper_main IDs excluded (~100)

Total: ~290 questions. Used by EXP-A to defensibly select the router
hard-threshold without contaminating the test set.

Run:
    PYTHONPATH=src python3 scripts/fetch_paper_train_questions.py \\
        --output scripts/data/router_sweep_questions_paper_train.jsonl \\
        --exclude scripts/data/router_sweep_questions_paper_main.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

DEFAULT_OUTPUT = Path("scripts/data/router_sweep_questions_paper_train.jsonl")
DEFAULT_EXCLUDE = Path("scripts/data/router_sweep_questions_paper_main.jsonl")
DEFAULT_SEED = 137  # different from paper_main's seed 42

# Counts; tweakable via CLI
DEFAULT_AIME_2024 = 30
DEFAULT_AIME_2025 = 30
DEFAULT_GPQA = 100
DEFAULT_ALPEVAL = 100
DEFAULT_MMSU = 100


def fetch_aime_year(year: int, count: int) -> List[Dict]:
    """Fetch AIME problems for a given year (2024 or 2025). Returns
    rows in the universal fixture schema, with source = aime_<year>."""
    from datasets import load_dataset

    if year == 2024:
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        # columns: ID, Problem, Solution, Answer
        problem_key = "Problem"
        answer_key = "Answer"
        idx_offset = 1
    elif year == 2025:
        ds = load_dataset("MathArena/aime_2025", split="train")
        # columns: problem_idx, problem, answer, problem_type
        problem_key = "problem"
        answer_key = "answer"
        idx_offset = 1
    else:
        raise ValueError(f"unsupported AIME year: {year}")

    rows: List[Dict] = []
    for idx in range(min(count, len(ds))):
        row = ds[idx]
        problem = row[problem_key]
        answer = str(row[answer_key]).strip()
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
            "id": f"aime_{year}/{idx + idx_offset:02d}",
            "source": f"aime_{year}",
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


def load_excluded_ids(path: Path) -> Set[str]:
    """Read an existing fixture jsonl and return the set of {source}/{raw_problem_prefix}
    keys so we can exclude them from sampling. Uses source+question-prefix as
    a stable fingerprint since some sources do not have stable upstream IDs."""
    excluded: Set[str] = set()
    if not path.exists():
        return excluded
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Use the raw question text (first 80 chars) as fingerprint
            # because upstream IDs may differ between fetches.
            q = (row.get("raw_problem") or row.get("question") or "")[:160]
            excluded.add(f"{row['source']}/{q}")
    return excluded


def _question_fp(source: str, raw_text: str) -> str:
    return f"{source}/{raw_text[:160]}"


def fetch_gpqa_diamond_remaining(
    target_count: int,
    excluded_fps: Set[str],
    rng: random.Random,
) -> List[Dict]:
    """Pull GPQA Diamond rows that are NOT in excluded_fps. Stratify
    by subject (Chemistry / Physics / Biology) as in paper_main."""
    from datasets import load_dataset

    ds = load_dataset("Wanfq/gpqa", "gpqa_diamond", split="train")
    # subject column is 'High-level domain'
    by_subj: Dict[str, List[int]] = {}
    for i, row in enumerate(ds):
        domain = (
            row.get("High-level domain")
            or row.get("high_level_domain")
            or "Unknown"
        )
        raw = row.get("Question") or row.get("question") or ""
        # paper_main source is 'gpqa_diamond', subject is normalized
        fp = _question_fp("gpqa_diamond", raw)
        if fp in excluded_fps:
            continue
        by_subj.setdefault(domain, []).append(i)

    # take roughly even per subject up to remaining
    selected_indices: List[int] = []
    domains = sorted(by_subj)
    per_domain = max(1, target_count // max(1, len(domains)))
    for d in domains:
        cand = by_subj[d][:]
        rng.shuffle(cand)
        selected_indices.extend(cand[: per_domain])
    rng.shuffle(selected_indices)
    selected_indices = selected_indices[: target_count]

    rows: List[Dict] = []
    for idx_in_target, ds_idx in enumerate(selected_indices, start=1):
        row = ds[ds_idx]
        correct = row["Correct Answer"]
        incorrect = [
            row.get("Incorrect Answer 1"),
            row.get("Incorrect Answer 2"),
            row.get("Incorrect Answer 3"),
        ]
        options = [correct] + [a for a in incorrect if a is not None]
        rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct)]
        question_text = row["Question"]
        mcq = question_text + "\n\n"
        for letter, opt in zip("ABCD", options):
            mcq += f"{letter}. {opt}\n"
        mcq += "\nAnswer with just the letter (A, B, C, or D)."
        rec = {
            "id": f"gpqa_diamond/{idx_in_target:03d}",
            "source": "gpqa_diamond",
            "subject": (
                row.get("High-level domain") or "unknown"
            ).lower(),
            "difficulty_hint": "graduate",
            "question": mcq,
            "answer": correct_letter,
            "raw_problem": question_text,
        }
        rows.append(rec)
    return rows


def fetch_voicebench_alpaca_remaining(
    target_count: int,
    excluded_fps: Set[str],
    rng: random.Random,
) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("hlt-lab/voicebench", "alpacaeval", split="test")
    # drop the 'audio' column to avoid torchcodec dependency
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")
    # columns include 'prompt' for the open-ended question
    candidates: List[int] = []
    for i, row in enumerate(ds):
        raw = row.get("prompt") or row.get("instruction") or ""
        fp = _question_fp("voicebench_alpaca_eval", raw)
        if fp in excluded_fps:
            continue
        candidates.append(i)
    rng.shuffle(candidates)
    candidates = candidates[: target_count]
    rows: List[Dict] = []
    for n, ds_idx in enumerate(candidates, start=1):
        row = ds[ds_idx]
        prompt = row.get("prompt") or row.get("instruction") or ""
        rec = {
            "id": f"voicebench_alpaca_eval/{n:03d}",
            "source": "voicebench_alpaca_eval",
            "subject": "open_ended",
            "difficulty_hint": "open",
            "question": prompt,
            "answer": "",  # open-ended; judged separately
            "raw_problem": prompt,
        }
        rows.append(rec)
    return rows


def fetch_voicebench_mmsu_remaining(
    target_count: int,
    excluded_fps: Set[str],
    rng: random.Random,
) -> List[Dict]:
    from datasets import load_dataset, get_dataset_split_names

    # In current voicebench packaging, MMSU is a single config 'mmsu'
    # whose SPLITS are the subjects (chemistry, physics, biology, ...).
    splits = get_dataset_split_names("hlt-lab/voicebench", "mmsu")
    by_subj: Dict[str, List[int]] = {}
    full_ds: Dict[str, "Dataset"] = {}
    for split in splits:
        ds = load_dataset("hlt-lab/voicebench", "mmsu", split=split)
        if "audio" in ds.column_names:
            ds = ds.remove_columns("audio")
        full_ds[split] = ds
        for i, row in enumerate(ds):
            raw = row.get("prompt") or ""
            fp = _question_fp("voicebench_mmsu", raw)
            if fp in excluded_fps:
                continue
            by_subj.setdefault(split, []).append(i)
    subjects = sorted(by_subj)
    per_subj = max(1, target_count // max(1, len(subjects)))
    selected: List[tuple] = []  # (split_name, idx_in_ds)
    for split in subjects:
        cand = by_subj[split][:]
        rng.shuffle(cand)
        for ds_idx in cand[: per_subj]:
            selected.append((split, ds_idx))
    rng.shuffle(selected)
    selected = selected[: target_count]
    rows: List[Dict] = []
    for n, (split, ds_idx) in enumerate(selected, start=1):
        row = full_ds[split][ds_idx]
        prompt = row.get("prompt") or ""
        reference = row.get("reference") or ""
        rec = {
            "id": f"voicebench_mmsu/{n:03d}",
            "source": "voicebench_mmsu",
            "subject": split,
            "difficulty_hint": "undergrad",
            "question": prompt,
            "answer": reference,
            "raw_problem": prompt,
        }
        rows.append(rec)
    return rows


def write_jsonl(rows: Iterable[Dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--aime-2024", type=int, default=DEFAULT_AIME_2024)
    p.add_argument("--aime-2025", type=int, default=DEFAULT_AIME_2025)
    p.add_argument("--gpqa", type=int, default=DEFAULT_GPQA)
    p.add_argument("--alpeval", type=int, default=DEFAULT_ALPEVAL)
    p.add_argument("--mmsu", type=int, default=DEFAULT_MMSU)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    excluded = load_excluded_ids(args.exclude)
    print(f"loaded {len(excluded)} excluded fingerprints from {args.exclude}", file=sys.stderr)

    rows: List[Dict] = []
    if args.aime_2024 > 0:
        r = fetch_aime_year(2024, args.aime_2024)
        print(f"  AIME 2024: {len(r)}", file=sys.stderr)
        rows.extend(r)
    if args.aime_2025 > 0:
        r = fetch_aime_year(2025, args.aime_2025)
        print(f"  AIME 2025: {len(r)}", file=sys.stderr)
        rows.extend(r)
    if args.gpqa > 0:
        r = fetch_gpqa_diamond_remaining(args.gpqa, excluded, rng)
        print(f"  GPQA Diamond (excl. test): {len(r)}", file=sys.stderr)
        rows.extend(r)
    if args.alpeval > 0:
        r = fetch_voicebench_alpaca_remaining(args.alpeval, excluded, rng)
        print(f"  AlpacaEval (excl. test): {len(r)}", file=sys.stderr)
        rows.extend(r)
    if args.mmsu > 0:
        r = fetch_voicebench_mmsu_remaining(args.mmsu, excluded, rng)
        print(f"  MMSU (excl. test): {len(r)}", file=sys.stderr)
        rows.extend(r)

    n = write_jsonl(rows, args.output)
    print(f"wrote {n} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
