# Judging

LLM judges used in the paper, with the exact prompts. Judges are called through an OpenAI-compatible endpoint; set
`CATT_JUDGE_API_KEY` and `CATT_JUDGE_BASE_URL` before calling a model. Judging needs the benchmark items (the
judge sees the question), so rebuild the fixture first (`docs/BENCHMARKS.md`).

| File | Role |
|---|---|
| `prompts/judge_substantive.md` | Sentence labels Y/N: does the sentence give substantive content for this question? First substantive content = start of the first sentence labelled Y. |
| `prompts/judge_g2.md` | Labels of the bridge and progress updates of turns routed to reasoning: OK, WRONG, REVERSED, TENTATIVE_CHANGED. A turn has an intermediate-speech error if any update is WRONG or REVERSED. |
| `prompts/judge_listen.md` | Listening rating (1 to 10) of the whole reply given its full timeline: every sentence with its start and end time and every silence. |
| `build_luna_units.py` | Builds the units for the two luna label tasks from run records. |
| `judge_luna.py` | Calls gpt-5.6-luna (reasoning effort "medium", max_tokens 16,000, up to 3 attempts with a format reminder). |
| `build_listening_units.py` | Builds the listening-rating units (main set, the two validity probes, a seeded 10 % retest sample) and the key map. |
| `judge_listen.py` | Calls gpt-5.6-luna and gemini-3.1-flash-lite with the listening prompt; the paper uses each judge and their per-turn mean. |
| `api_client.py` | Minimal chat client (standard library only). |

Both judge scripts have `--check-prompts`: given units and existing label files, they re-render every prompt and
compare its sha256 with the `prompt_sha256` stored in the label record, without calling a model. On the study's
records this check passes for all 6,210 luna label calls and all 9,726 listening-rating calls, and the two unit
builders rebuild the recorded units byte for byte.

```bash
python build_luna_units.py --runs RUNS/MAIN:formal_v4 --fixtures main.jsonl --out units/units_main.jsonl
python judge_luna.py units/units_main.jsonl labels/
python judge_luna.py units/units_main.jsonl labels/ --retest 0.10 --retest-seed 20261009
python build_listening_units.py --run RUNS/MAIN --pkg ../../orchestration --exclude excluded_ids.txt --out listen_units/
python judge_listen.py --units listen_units/units_probeA.jsonl listen_units/units_probeB.jsonl \
    listen_units/units_main.jsonl listen_units/units_retest.jsonl --out listen_labels/
```

`build_listening_units.py --pkg` expects `code/v5_common.py` and `v3/fixtures/main.jsonl` under the given directory:
pass `../../orchestration` after placing the rebuilt fixture at `orchestration/v3/fixtures/main.jsonl`.
`--exclude` takes a file with the ids of items to leave out, one per line.

Hosted judge models change over time; the same prompts may not return the same labels today. The released labels
and scores (`data/`) are the study's original ones. The ReVerT semantic-correctness labels and the labels of the
progress-update content audit are not released (see RELEASE_NOTES.md).
