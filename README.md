# CATT: code and data release

Code and per-item data for the paper *CATT: Complexity-Aware Thinker-Talker Orchestration for Low-Latency Spoken
Responses in Digital Humans* (Xi Cheng, Zezhen Ding, Fengyu Xu, Kaidi Xu and Tianlong Chen; manuscript under review at
*Electronics*).

CATT is a deployment-only orchestration layer for a spoken agent. A router sends each turn either to a direct
answer or to a reasoning path. On the reasoning path a talker speaks a short answer-free acknowledgment (called the
bridge in the code and data) at once, then turns the thinker's streamed reasoning into spoken progress updates, and
finally speaks the answer. This repository holds the exact orchestration code of the paper's main experiment, the
evaluation code, the per-item records of the 738 evaluated items, and the per-item records of the transfer study on
three hosted models (139 of those items).

## What is here

| Path | Contents | Licence |
|---|---|---|
| `orchestration/` | The CATT orchestration as it ran in the main experiment: runner, CATT layers, router, controller, speech normalizer, baselines, launch scripts, configuration and prompts. The 76 code, configuration and prompt files are byte-identical to the frozen package; `orchestration/FROZEN_HASHES.tsv` lists each sha256 next to the value in the run's freeze record. | Apache-2.0 |
| `router_bank/` | Router reference vectors without the GPQA rows, with the scripts and steps to rebuild them. | Apache-2.0 (scripts), CC BY 4.0 (vectors, rows.csv) |
| `orchestration_post_study_fix/` | A later fix of the speech normalizer (markdown emphasis and spaced dashes): not used in the main runs; the transfer runs used it. | Apache-2.0 |
| `evaluation/judging/` | Judge prompts and scripts: substantive-content and intermediate-speech labels (gpt-5.6-luna) and the listening rating with the full timeline (gpt-5.6-luna and gemini-3.1-flash-lite). | Apache-2.0 |
| `evaluation/analysis/` | `reproduce_paper.py` recomputes the main reported values from `data/` (see the quick check below); `expected/` holds the values reported by the study for comparison. | Apache-2.0 (code), CC BY 4.0 (`expected/`) |
| `data/` | Per-item data of the 738 evaluated items: item table with question hashes, per-turn metrics, judge labels, listening-rating scores, run records (timelines, spoken text, reasoning), the ReVerT comparison, and the transfer study (DeepSeek V4.1 Flash, GLM-5.2, Qwen3.5-27B on Bailian). See `docs/DATA.md`. | CC BY 4.0 (see `LICENSE-DATA`) |
| `docs/` | `DATA.md` (data dictionary), `REPRODUCE.md` (three levels of reproduction), `BENCHMARKS.md` (how to obtain the benchmark items). | CC BY 4.0 |

Benchmark items (AIME 2026, GPQA Diamond, VoiceBench MMSU and AlpacaEval) are **not** redistributed. They keep
their own terms; `docs/BENCHMARKS.md` explains how to obtain them and how to check that you have the same items
(sha256 of each question as used). GPQA Diamond and AIME texts are withheld: their questions, and every model
response, reasoning text, spoken sentence and judge input on those items, are released as identifiers, metrics and
hashes only.

## Quick check: recompute the main reported values

Python 3.9 or later with numpy:

```bash
cd evaluation/analysis
python reproduce_paper.py --data ../../data --out results --check expected/paper_numbers_expected.json
```

This recomputes 6,003 reported values from the released per-item data with the study's statistics and compares them
with `expected/paper_numbers_expected.json`: 1,585 values of the main contrasts (the three CATT configurations
against both baselines on the 738 items: timing, accuracy, calls, tokens and energy, with silence on turns routed to
reasoning, intermediate-speech errors, routing and reasoning identity), 9 of the test-retest agreement of the
gpt-5.6-luna labels, 523 of the listening rating, 3,240 of the ReVerT comparison and 646 of the transfer study. It
takes about 20 seconds and needs no GPU, model or network access. Intermediate-speech errors are recomputed for the
turns that count under the main rule (closed-form turns only with a strictly correct final answer) and, for the
router configuration, over all 410 turns routed to reasoning: 62 of the 410 turns, 58 of the 168 closed-form turns,
32 of the 65 closed-form turns with a wrong final answer, and the split of the 168 closed-form turns into correct
final answer without and with an intermediate error (77 and 26 turns) and wrong final answer with and without one
(32 and 33), each with its Wilson interval.

The script does not recompute several reported results, including the per-dataset contrasts, the McNemar tests, the
distribution appendix, the failure-case table, the unit-level intermediate-error counts, the between-configuration
listening contrasts, the listening-rating ICCs, its retest mean change and largest pooled-contrast change, the
between-judge Spearman correlation, the association of the rating with length, the output-length table, the
routing-threshold curve, the ReVerT semantic-correctness accuracy and fidelity check F4, the perturbation replay, the
routing-policy comparison, the progress-update content audit, the backlog-threshold check and the case study. For the
main-study analyses among them (per-dataset contrasts, McNemar tests, distributions, failure cases, intermediate-error
counts, the listening-rating analyses, output length, routing-threshold curve), the per-turn data are released in
`data/`. The other results rest on inputs that are not part of this release (token-level timing logs for the replay,
additional labels, development items, separate runs).

## Main experiment in brief

- Model: Qwen3-32B (revision 9216db5781bf21249d130ec9da846c4624c16137), bf16, served by vLLM 0.10.2 as two engines
  (thinker and talker, tensor parallel 4 each) sharing four A100 40 GB GPUs; speech synthesis with Piper
  (`en_US-amy-low`, through sherpa-onnx) on CPU; virtual playback timing; one user; one generation per item and
  condition, generation seed 20261009.
- Conditions: CATT with the router at threshold 0.5444878691642917 (`catt_v4_route`), CATT with the NVIDIA prompt
  task and complexity classifier at 0.1191 (`catt_v4_nvroute`), CATT with every turn on the reasoning path
  (`catt_v4_allhard`), and two prompt-matched single-call baselines (`thinking`, `non_thinking`).
- Items: 738 evaluated items: AIME 2026 (30), GPQA Diamond (134), VoiceBench MMSU (50), VoiceBench AlpacaEval (50)
  and 474 further VoiceBench AlpacaEval prompts. 64 further GPQA Diamond items were used to set the routing
  threshold and are therefore excluded from evaluation; they appear nowhere in this release.
- Measures: first audio; first substantive content (gpt-5.6-luna sentence labels); first candidate and final answer
  (closed-form items); silence on turns routed to reasoning; strict accuracy; intermediate-speech errors
  (gpt-5.6-luna); listening rating with the full timeline (two judges); calls, tokens and GPU energy.

## Layout of the orchestration code

The frozen package keeps its historical layered layout because each layer imports the one below through relative
paths: `orchestration/code/` (main-experiment runner and CATT layer), `orchestration/v3/code/`,
`orchestration/v3/v2/code/` and `orchestration/v3/v2/exp003/` (earlier layers, the original controller, talker,
thinker, router and baselines under `snapshot/src/metahuman/`). `orchestration/README.md` maps the files.

## Citation

See `CITATION.cff`. Please also cite the benchmarks you use (`docs/BENCHMARKS.md`).

## Licence

Code: Apache License 2.0 (`LICENSE`). Data generated in this study (everything under `data/`, the expected values in
`evaluation/analysis/expected/`, the reference vectors and row map in `router_bank/`) and the documentation under
`docs/`: Creative Commons Attribution 4.0 International (`LICENSE-DATA`). Third-party components and benchmark terms:
`NOTICE`.
