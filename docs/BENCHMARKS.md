# Benchmark items: how to obtain them

The benchmark items are not redistributed. Each keeps its own terms of use; obtain them from their sources and cite
them. `data/items.csv` lists every evaluated item by its source identifier together with the sha256 of the exact
question string given to the model, so that a rebuilt fixture can be checked item by item.

| Dataset | Items evaluated | Source used by the fixture scripts | Notes |
|---|---|---|---|
| AIME 2026 | 30 (all) | `MathArena/aime_2026` on the Hugging Face Hub | Gated; problem text, model responses and reasoning on these items are withheld (identifiers, metrics and hashes only). |
| GPQA Diamond | 134 of 198 | `Wanfq/gpqa`, config `gpqa_diamond` (a copy of the GPQA release) | The GPQA authors ask that the questions not be published; this release gives identifiers, metrics and hashes only, also for model responses and reasoning. 64 further Diamond items were used to set the routing threshold and are therefore excluded from evaluation. |
| VoiceBench MMSU | 50 | `hlt-lab/voicebench`, config `mmsu` | Text transcripts with lettered options. |
| VoiceBench AlpacaEval | 50 | `hlt-lab/voicebench`, config `alpacaeval` | Text transcripts only. |
| VoiceBench AlpacaEval (full) | 474 | `hlt-lab/voicebench`, config `alpacaeval_full` | Prompts checked against all development and calibration sources for overlap. |

## Rebuilding the fixture

The study's fixture was built in three steps; the scripts are in `orchestration/`:

1. `v3/v2/exp003/snapshot/scripts/fetch_paper_main_questions.py` (seed 42): AIME 2026 (30, source order), a
   stratified 90-item GPQA Diamond subset (Chemistry 36, Physics 35, Biology 19; options shuffled with the seeded
   generator), 50 AlpacaEval and 50 MMSU items (seeded samples). Prompt suffixes are fixed in the script.
2. `v3/v2/code/build_fixtures.py` (seed 20261006): the remaining 108 GPQA Diamond items in Record-ID order with a
   fresh seeded option shuffle and the same formatting.
3. `v3/code/build_main_fixture.py`: concatenates the two and adds the clean `alpacaeval_full` prompts after an
   overlap audit against every development and calibration source.

The resulting file (`orchestration/v3/fixtures/main.jsonl`) has sha256
`b095728d5980f913e35dfe22b651cf4bd567ec7c9095532e6244fea9ac934c5d` (recorded in `orchestration/config_v5.json`) and
contains 802 items; the 738 evaluated items are those in `data/items.csv`. Step 3 also reads development and
calibration files that are not released, so it cannot be rerun as is; the per-item hashes in `data/items.csv`
(`question_sha256` for the exact string, `question_norm_sha256` after normalisation) let you check a fixture
rebuilt with steps 1 and 2 and the `alpacaeval_full` identifiers. Source datasets can change after our download, in
which case some hashes will not match.

## Citations

- AIME 2026 problems: Mathematical Association of America; as distributed by MathArena (dataset `MathArena/aime_2026`):
  Balunović et al., "MathArena: Evaluating LLMs on Uncontaminated Math Competitions", Advances in Neural Information
  Processing Systems 38 (NeurIPS 2025, Datasets and Benchmarks Track).
- GPQA: Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", Conference on Language Modeling (COLM),
  2024.
- VoiceBench: Chen et al., "VoiceBench: Benchmarking LLM-Based Voice Assistants", Transactions of the Association for
  Computational Linguistics 14:378–398, 2026, doi:10.1162/tacl.a.628 (MMSU and AlpacaEval subsets).
- AlpacaEval: Dubois et al., "Length-Controlled AlpacaEval: A Simple Debiasing of Automatic Evaluators", Conference on
  Language Modeling (COLM), 2024.
