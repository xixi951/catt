"""Offline pipeline: battles.csv + train fixture -> router artifacts.

Inputs:

* ``--battles-csv``: per-prompt battle table from
  ``scripts/run_battle_labels.py`` with columns:

    prompt_id (matches the train fixture's ``id`` column)
    winner    (one of "strong" / "weak" / "tie")
    weight    (optional, defaults to 1.0)

* ``--train-fixture``: the JSONL produced by
  ``scripts/fetch_router_train_questions.py`` carrying the prompt
  text per ``id``.

Outputs (in ``--output-dir``, default ``data/router/``):

* ``battles.csv`` -- canonical CSV with columns
  ``prompt_id, winner, weight`` plus the prompt text aligned to row
  order; embeddings.npy row ``i`` corresponds to ``df.iloc[i]``.
* ``embeddings.npy`` -- ``(N, dim)`` float32 L2-normalised.
* ``manifest.json`` -- metadata: embedding model, instruction,
  dim, normalize, count, source SHA256s.

Online (``SWRankingQwen3Router``) loads these three files and runs
weighted full-similarity Bradley-Terry to compute strong-win rate
per request.

Usage::

    PYTHONPATH=src python scripts/build_router_artifacts.py \\
        --battles-csv data/router/battles_raw.csv \\
        --train-fixture scripts/data/router_train_questions.jsonl \\
        --output-dir data/router

This script intentionally re-implements the offline embedding pass
in ~150 lines so it has zero overlap with online inference; the
SAME instruction text and model name must be used on both sides
(see ``ROUTER_QUERY_INSTRUCTION`` and ``ROUTER_QUERY_INSTRUCTION_VERSION``
in ``src/metahuman/adapters_impl/sw_ranking_qwen3_router.py``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Must match SWRankingQwen3Router constants exactly.
ROUTER_QUERY_INSTRUCTION = (
    "Represent the user request for LLM routing. Similar requests "
    "should require similar levels of reasoning, domain expertise, "
    "coding ability, mathematical ability, and need for a stronger "
    "reasoning model."
)
ROUTER_QUERY_INSTRUCTION_VERSION = "route_query_v1"

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 32


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_train_fixture(path: Path) -> Dict[str, str]:
    """Read the JSONL training fixture and return ``id -> question``."""
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["id"]] = row["question"]
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--battles-csv", type=Path, required=True,
        help="Per-prompt battle CSV with columns prompt_id,winner[,weight].",
    )
    p.add_argument(
        "--train-fixture", type=Path, required=True,
        help="Training fixture JSONL (carries the prompt text per id).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/router"),
        help="Where to write battles.csv, embeddings.npy, manifest.json.",
    )
    p.add_argument(
        "--embedding-model", type=str, default=DEFAULT_EMBED_MODEL,
        help=f"HF model id (default: {DEFAULT_EMBED_MODEL}).",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Embed batch size; raise on free-GPU machines for speed.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help='"cuda" / "cpu" / None (autodetect).',
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. read battles + align prompt text from the training fixture.
    import pandas as pd

    battles = pd.read_csv(args.battles_csv)
    required = {"prompt_id", "winner"}
    missing = required - set(battles.columns)
    if missing:
        print(
            f"ERROR: battles CSV missing columns {missing}",
            file=sys.stderr,
        )
        return 2
    if "weight" not in battles.columns:
        battles["weight"] = 1.0

    fixture = load_train_fixture(args.train_fixture)
    missing_ids = [pid for pid in battles["prompt_id"] if pid not in fixture]
    if missing_ids:
        print(
            f"ERROR: {len(missing_ids)} battle rows have prompt_id not in "
            f"training fixture; sample: {missing_ids[:5]}",
            file=sys.stderr,
        )
        return 2

    battles["prompt"] = battles["prompt_id"].map(fixture)

    # 2. encode with sentence-transformers.
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        args.embedding_model,
        device=args.device,
        trust_remote_code=True,
    )
    print(
        f"loaded {args.embedding_model} on "
        f"{encoder.device} (dim={encoder.get_sentence_embedding_dimension()})",
        file=sys.stderr,
    )

    wrapped = [
        f"Instruct: {ROUTER_QUERY_INSTRUCTION}\nQuery: {prompt}"
        for prompt in battles["prompt"].tolist()
    ]

    embeddings = encoder.encode(
        wrapped,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # 3. write artifacts.
    embeddings_path = args.output_dir / "embeddings.npy"
    battles_path = args.output_dir / "battles.csv"
    manifest_path = args.output_dir / "manifest.json"

    np.save(embeddings_path, embeddings)
    battles[["prompt_id", "winner", "weight"]].to_csv(
        battles_path, index=False,
    )

    manifest = {
        "embedding_model": args.embedding_model,
        "embedding_dim": int(embeddings.shape[1]),
        "row_count": int(len(battles)),
        "instruction_version": ROUTER_QUERY_INSTRUCTION_VERSION,
        "instruction": ROUTER_QUERY_INSTRUCTION,
        "normalize": True,
        "battles_csv_sha256": _sha256(battles_path),
        "embeddings_npy_sha256": _sha256(embeddings_path),
        "train_fixture_sha256": _sha256(args.train_fixture),
        "source_battles_csv_sha256": _sha256(args.battles_csv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"wrote {len(battles)} rows of battles + embeddings "
        f"({embeddings.shape[1]} dim) to {args.output_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
