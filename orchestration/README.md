# Orchestration code of the main experiment

Every file in this directory, except this README, `FROZEN_HASHES.tsv` and `freeze_record_excerpt.json`, is a
byte-identical copy of the frozen package that ran the paper's main experiment. `FROZEN_HASHES.tsv` lists each
file's sha256 next to the value recorded in the experiment's freeze record before the first formal attempt
(76 of 76 match). `freeze_record_excerpt.json` reproduces the record's fields that describe this configuration.

Because the files are kept byte-identical, they still contain the paths, host aliases and internal record
references of the original deployment (for example `/data/userhome/...` paths in `config_v5.json` and the launch
scripts, one local development path (`/Users/...`), comments that cite internal experiment identifiers, plans or
review rounds). These refer to the lab's servers and project records; they are not needed to read or run the code.
Adapt the paths in `config_v5.json` and the launch scripts to your machine. The engine launcher
(`code/launch_engines_v5.sh`) records the `HF_*` environment variables (with the `CUDA_*`, `VLLM_*` and `NCCL_*`
ones) in an `.env` log next to its engine log, so unset `HF_TOKEN` before launching or delete that log before
sharing logs.

## Layers

The package grew in layers; each layer imports the one below through paths relative to its own file, so the
directory structure must stay as it is.

| Path | Role |
|---|---|
| `config_v5.json` | Frozen run configuration: model and weights revision, the two engine endpoints, routing threshold τ* = 0.5444878691642917, reasoning budget 16,000 tokens, seeds (schedule and generation 20261009), talker and controller parameters, switches of the main-experiment version, NVIDIA classifier threshold 0.1191, talker GPU share. |
| `code/run_v5.py` | Driver: runs one set of items under the five conditions (three CATT configurations, two prompt-matched baselines), with virtual playback and Piper TTS; writes one directory per attempt (`summary.json`, `events.jsonl`, `deltas.jsonl`, workspace log, WAV files). Refuses to run if a frozen hash differs. |
| `code/catt_v4.py` | CATT version used in the paper: parallel routing with a held bridge, prompts by question kind, sentence splitter and bridge-guard fixes, spoken-form normalizer (`code/speech_norm.py`). |
| `code/v5_common.py` | Shared helpers: played chunks, sentence units, silence components (used by the evaluation code). |
| `code/nv_model.py`, `code/nv_prepass.py` | NVIDIA prompt task and complexity classifier (`nvidia/prompt-task-and-complexity-classifier`) used as the alternative router. |
| `code/launch_engines_v5.sh`, `code/start_mps_v5.sh`, `code/launch_router_v5.sh`, `code/bringup_v5.py` | Start the two vLLM engines under CUDA MPS (talker at 50 % of the SMs), the router process, and the engine health checks. |
| `prompts/` | Open-ended final-answer prompt (O1) and open-ended easy-path prompt (E1open). |
| `v3/code/catt_v3.py`, `v3/prompts/` | Previous layer: two-engine thinker and talker, bridge sizing, progress dispatch, final cut-in, forced-final prompt. |
| `v3/code/scorer_v4.py` (+ `v3/v2/code/scorer_v3.py`, `v3/v2/exp003/snapshot/scripts/score_results.py`) | Strict answer extraction and scoring for AIME, GPQA and MMSU, and answer-time detection. |
| `v3/v2/exp003/snapshot/scripts/fetch_paper_main_questions.py`, `.../fetch_paper_train_questions.py`, `v3/v2/code/build_fixtures.py`, `v3/code/build_main_fixture.py` | How the item fixtures were built from the public sources (see `docs/BENCHMARKS.md`). The fixtures themselves are not released. |
| `v3/code/make_hard_lists.py`, `v3/v2/exp003/code/v2/router_prepass.py` | Router pre-pass over the fixture and the lists of items routed to reasoning. |
| `v3/judges/` | The two gpt-5.6-luna judge prompts (also in `evaluation/judging/prompts/`). |
| `v3/v2/code/catt_v2.py`, `v3/v2/prompts_v3/` | Earlier layer: progress guard, dialogue workspace integration. |
| `v3/v2/exp003/code/v2/run_v2.py`, `v3/v2/exp003/prompts_v2/baseline_matched.md`, `v3/v2/exp003/config_v2.json` | Base driver classes and the prompt-matched baseline prompt. |
| `v3/v2/exp003/snapshot/src/metahuman/` | Original system: controller, thinker, talker, workspace, LLM client, router RPC server and client, similarity-ranking router (`adapters_impl/sw_ranking_qwen3_router.py`, Qwen3-Embedding-0.6B), sherpa-onnx Piper TTS adapter. |
| `v3/v2/exp003/snapshot/scripts/baseline_direct_llm.py` | Single-call baseline helpers. |
| `v3/v2/exp003/snapshot/data/router/battles.csv`, `manifest.json` | Router reference bank labels (item ids, which model won, weight) and its manifest. |
| `*/test_*.py` | Model-free unit tests (speech normalizer, scorer, bridge guard, loop guard, progress guard). |

## Not included from the frozen package, and why

- Item fixtures and question files (`v3/fixtures/`, `v3/v2/fixtures/`, `v3/inputs/*.jsonl`,
  `v3/v2/exp003/snapshot/scripts/data/`): benchmark text; rebuild them as in `docs/BENCHMARKS.md`.
- Router reference embeddings (`v3/v2/exp003/snapshot/data/router/embeddings.npy`): released in `router_bank/`
  without the 171 GPQA rows, with the scripts and steps to rebuild them.
- Development, selection, mock and audit scripts, internal planning documents and the full freeze records: not
  needed to run or evaluate the released configuration.
- The frozen analysis and unit-building scripts are omitted. The released `evaluation/analysis/reproduce_paper.py`
  recomputes the main reported values (see the top-level README), not every analysis of the paper. Judge units are
  built by `evaluation/judging/build_luna_units.py` and `build_listening_units.py`; run on the recorded runs, both
  rebuild the recorded judge inputs byte for byte.

The run covered 802 items (4,010 attempts; `freeze_record_excerpt.json` counts them). 64 GPQA Diamond items were
used to set the routing threshold and are therefore excluded from evaluation; their records are not released.

## Running it

A full rerun needs four GPUs with about 40 GB each, Qwen3-32B at the revision above, vLLM 0.10.2, the Piper voice
`vits-piper-en_US-amy-low` (sherpa-onnx release), Qwen3-Embedding-0.6B for the router, the NVIDIA classifier for
the alternative router, and the rebuilt fixture. See `docs/REPRODUCE.md`.
