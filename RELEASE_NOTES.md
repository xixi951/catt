# Release notes

## 1.0.0 (2026-10-16)

First public release, accompanying the revised manuscript.

### Included

- `orchestration/`: 76 files of the main experiment's frozen package (run driver, CATT layers, router, controller,
  baselines, scorer, launch scripts, configuration, prompts, fixture-building scripts, model-free unit tests), each
  byte-identical to the frozen file and checked against the freeze record (`FROZEN_HASHES.tsv`).
- `router_bank/`: router reference vectors without the GPQA rows, the scripts that built the bank, and an assembly
  helper.
- `orchestration_post_study_fix/`: the speech-normalizer fix made after the main runs (not used in the main runs; the
  transfer runs and a separate digital-human case study used it).
- `evaluation/judging/`: judge prompts (substantive content, intermediate-speech errors, listening rating), unit
  builders and judge scripts. On the study's records, the unit builders rebuild the recorded judge inputs byte for
  byte and the judge scripts re-render all 15,936 recorded judge prompts with identical sha256.
- `evaluation/analysis/`: `build_turns.py` (per-turn table from run records; rebuilds `data/v4/turns.csv` byte for
  byte) and `reproduce_paper.py` (recomputes 5,968 reported values from `data/` with 0 differences: 1,553 of the
  main contrasts of the three CATT configurations against both baselines on the 738 items, 6 of the test-retest
  agreement of the gpt-5.6-luna labels, 523 of the listening rating, 3,240 of the ReVerT comparison and 646 of the
  transfer study; see the next section for reported results it does not recompute).
- `data/`: item table (738 items, question hashes, routes), per-turn metrics (3,690 turns), gpt-5.6-luna labels
  and their re-judged sample (test-retest), run records of every turn, listening-rating scores of both judges
  (including validity probes and retests), the ReVerT per-turn metrics (139 items, three arms), and the transfer
  study's per-turn metrics, gpt-5.6-luna labels and run records for DeepSeek V4.1 Flash, GLM-5.2 and Qwen3.5-27B on
  Bailian (139 items, 3 conditions each).

### Reported results not recomputed by `reproduce_paper.py`

The script does not recompute several reported results, including:

- From released per-turn data, but not computed by the script: the per-dataset contrasts, the McNemar tests, the
  distribution appendix, the failure-case table, the unit-level intermediate-error counts, the between-configuration
  listening contrasts, the listening-rating ICCs, its retest mean change and largest pooled-contrast change, the
  between-judge Spearman correlation, the association of the rating with length, the output-length table and the
  routing-threshold curve of the main study.
- Not recomputable from this release: the ReVerT semantic-correctness accuracy (labels and judge prompt not
  released), the ReVerT fidelity check F4 (a separate control run), the perturbation replay (needs the token-level
  timing logs), the routing-policy comparison, the progress-update content audit, the backlog-threshold check
  (development items) and the case study.

### Not included, and why

| Not included | Reason |
|---|---|
| Benchmark question, option and answer text (all datasets) | Benchmarks keep their own terms and are not redistributed; `docs/BENCHMARKS.md` explains how to obtain them. |
| GPQA Diamond and AIME texts: model responses, reasoning, spoken text and judge inputs | These texts reproduce or closely paraphrase the questions (the GPQA authors ask that questions not be published; AIME problems are not redistributed); GPQA and AIME turns carry identifiers, metrics and hashes only. |
| Raw reasoning of the hosted models in the transfer study | Kept separately by the run harness; token and character counts are released. |
| Transfer-study harness (the orchestration packaged for hosted APIs and its API runner) | Not part of this release; the per-turn data of the transfer runs and the values computed from them are released (`docs/DATA.md`). |
| ReVerT adapter code (runs ReVerT's released verbalizer on the recorded reasoning streams) | Not part of this release; the per-turn data and the values computed from them are released (`docs/DATA.md`). |
| ReVerT semantic-correctness labels and their judge prompt | Not part of this release; the semantic accuracy of the ReVerT comparison is therefore not recomputed (the strict scorer's correctness is in `data/revert/turns.jsonl`). |
| ReVerT spoken text | Not part of this release; only per-turn metrics of the ReVerT arms are released (`data/revert/turns.jsonl`). |
| ReVerT fidelity check F4 (control run on the ReVerT authors' GSM8k training data) | A separate control run outside the evaluated items; not part of this release. |
| The 64 GPQA Diamond items used to set the routing threshold | Used to set the routing threshold and therefore excluded from evaluation. |
| Judge free-text notes and reasons, reference answers shown to judges | They can quote benchmark content. |
| Audio (WAV) and token-level streaming logs | Size; all timings derived from them are in the run records and the per-turn table. |
| Router reference vectors of the 171 GPQA rows | They encode GPQA questions; `router_bank/` releases the other 590 vectors and the steps to rebuild the GPQA rows. |
| Item fixtures, development and calibration item files | Benchmark text. |
| Internal planning documents, full freeze records, development and audit scripts, and the frozen analysis scripts | Not needed to run the released configuration or to recompute the values that `reproduce_paper.py` checks. |

### Known issues of the main experiment

- The speech normalizer read markdown emphasis (`*title*`) as "times" in a small share of the turns in every
  condition and a spaced dash between words as "minus" in up to 8 turns per condition, equally in all conditions.
  Fixed after the main runs (`orchestration_post_study_fix/`); the main runs were not repeated, and the transfer runs
  used the fix.
- Three AIME turns per thinking condition reached the 16,000-token reasoning budget (`aime_2026/14`, `/26`, `/27`;
  `forced` and `budget_forced_calls` in `data/v4/turns.csv`).
