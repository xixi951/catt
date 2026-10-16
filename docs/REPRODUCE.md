# Reproducing the results

There are three levels. Level 1 needs only this repository; levels 2 and 3 need the benchmark items, and level 3
needs GPUs.

## Level 1: recompute the main reported values from the released data (minutes, CPU)

```bash
cd evaluation/analysis
python reproduce_paper.py --data ../../data --out results --check expected/paper_numbers_expected.json
```

Requires Python 3.9 or later and numpy. The script recomputes 5,968 reported values with the study's statistics and
checks them against `expected/paper_numbers_expected.json`: 1,553 values of the main contrasts (the three CATT
configurations against both baselines on the 738 evaluated items: timing, accuracy, calls, tokens and energy, with
silence on turns routed to reasoning, intermediate-speech errors, routing and reasoning identity), 6 of the
test-retest agreement of the gpt-5.6-luna labels (same first substantive sentence: 359 of 373 re-judged turns;
same intermediate-error flag: 151 of 152), 523 of the listening rating (including the validity probes, and
exact-agreement summaries of its test-retest and inter-judge agreement; these summaries are not the ICCs reported in
the paper), 3,240 of the ReVerT comparison and 646 of the transfer study (with that study's own bootstrap seed,
20261011). Expected result: `0 differences`.

The script does not recompute several reported results, including:

- computed from released per-turn data, but not by this script: the per-dataset contrasts, the McNemar tests, the
  distribution appendix, the failure-case table, the unit-level intermediate-error counts, the between-configuration
  listening contrasts, the listening-rating ICCs, its retest mean change and largest pooled-contrast change, the
  between-judge Spearman correlation, the association of the rating with length, the output-length table and the
  routing-threshold curve of the main study;
- not recomputable from this release: the ReVerT semantic-correctness accuracy (labels and judge prompt not
  released), the ReVerT fidelity check F4 (a separate control run), the perturbation replay (needs the token-level
  timing logs), the routing-policy comparison, the progress-update content audit, the backlog-threshold check
  (development items) and the case study.

You can also inspect any single turn: `data/v4/transcripts/<condition>.jsonl.gz` shows what was played and when,
the talker's outputs, the reasoning and every model call, and `data/v4/luna_labels.jsonl.gz` shows the judge
labels for each sentence (GPQA and AIME turns as hashes and lengths, see `docs/DATA.md`); `data/transfer/` holds the
same for the transfer study.

## Level 2: re-judge (hours, an API account for the judge models)

1. Rebuild the item fixture (`docs/BENCHMARKS.md`).
2. Build the judge units from run records with `evaluation/judging/build_luna_units.py` and
   `build_listening_units.py`. They need the run records of a rerun (level 3); for the released turns the judge
   inputs can be assembled from `data/v4/transcripts/` and the rebuilt fixture (the unit of judgment is the played
   sentence in its spoken form; see the builders). This works for MMSU and AlpacaEval turns only; GPQA Diamond and
   AIME turns carry hashes of their spoken text, so re-judging them needs a rerun (level 3).
3. Call the judges with `judge_luna.py` and `judge_listen.py` (environment variables `CATT_JUDGE_API_KEY`,
   `CATT_JUDGE_BASE_URL`).

The judge prompts are exactly those of the study: on the study's records, `--check-prompts` re-renders all 15,936
recorded judge prompts with identical sha256. Hosted judge models may have changed since October 2026, so new
labels can differ from the released ones.

## Level 3: rerun the experiment (about one day on four GPUs)

Requirements, as in the study:

- 4 GPUs with about 40 GB each (the study used A100 40 GB), CUDA MPS;
- Qwen3-32B, revision `9216db5781bf21249d130ec9da846c4624c16137`, bf16; vLLM 0.10.2;
- Qwen3-Embedding-0.6B and the router reference bank (`orchestration/README.md`, router section);
- the NVIDIA prompt task and complexity classifier (`nvidia/prompt-task-and-complexity-classifier`, revision
  `fea11215`) for the alternative router;
- the Piper voice `vits-piper-en_US-amy-low` for sherpa-onnx (CPU);
- the rebuilt fixture at `orchestration/v3/fixtures/main.jsonl`.

Steps (paths in `orchestration/config_v5.json` and the launch scripts must be adapted first):

```bash
cd orchestration
bash code/start_mps_v5.sh              # CUDA MPS daemon for the four GPUs
bash code/launch_engines_v5.sh         # thinker and talker vLLM engines (TP4 each; talker at 50 % of the SMs)
bash code/launch_router_v5.sh          # CATT router process (Unix socket)
python code/bringup_v5.py ...          # engine health checks
python code/run_v5.py --config config_v5.json --set MAIN --out runs/<run>/MAIN
```

`run_v5.py` runs the five conditions item by item in a seeded order (seed 20261009) and writes one directory per
attempt. In its formal mode it also checks every file of the complete frozen package and the full freeze record,
which are not all part of this release; a reader's rerun therefore runs the same code without that check (for
example by registering the fixture as a development set in `SETS` at the top of `run_v5.py`). Generation uses
seed 20261009, but vLLM outputs are not guaranteed to be identical across hardware and driver versions, so a rerun
reproduces the procedure, not the exact text.

After judging (level 2), `evaluation/analysis/build_turns.py` turns the run records, the fixture, the luna labels
and the GPU energy samples into the per-turn table; on the study's records it rebuilds `data/v4/turns.csv` byte for
byte. `reproduce_paper.py` then computes the main reported values from it: point `--data` at a copy of `data/` in which
`v4/turns.csv` is replaced by your table (the script also reads the other files under `data/`).
