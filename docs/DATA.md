# Data dictionary

All data in `data/` were generated in this study and are released under CC BY 4.0. They cover the 738 evaluated
items and the transfer study on 139 of them. No benchmark question or option text is included. On GPQA Diamond and
AIME items every text field (spoken sentences, talker outputs, reasoning, call outputs, judge units) is replaced by
`{"sha256": ..., "chars": n}`; for texts shorter than 64 characters only the length is given (`"sha256": null`),
because a short sentence could be recovered from its hash by guessing. The same rule holds for every `sha256` field in
the `.jsonl.gz` files, on all items: a hash is given only for texts of 64 characters or more. Times are seconds after
the end of the user's turn; `1200` means the event did not occur within the turn watchdog.

Item identifiers are those of the sources: `aime_2026/NN`, `gpqa_diamond/<domain>/<record id>`,
`voicebench_mmsu/<subject>/<id>`, `voicebench_alpaca_eval/NNN` (the 50-item sample), and
`voicebench_alpacaeval_full/NNN` (474 further prompts). Conditions: `catt_v4_route` (CATT, router at
τ* = 0.5444878691642917), `catt_v4_nvroute` (CATT, NVIDIA classifier router at 0.1191), `catt_v4_allhard` (CATT,
every turn on the reasoning path), `thinking` and `non_thinking` (prompt-matched single-call baselines). On a turn
routed to reasoning, CATT first speaks a short answer-free acknowledgment (called the bridge in the code and data).

## `items.csv` (738 rows)

| Column | Meaning |
|---|---|
| `question_id` | Item identifier. |
| `dataset`, `subject` | Source dataset and subject or domain. |
| `question_type` | `closed` (AIME, GPQA, MMSU; scored for accuracy) or `open` (AlpacaEval). |
| `question_sha256` | sha256 of the UTF-8 question string exactly as given to the model (for GPQA and MMSU including the lettered options in our order). |
| `question_norm_sha256` | sha256 of the same string after NFKC, lower-casing, replacing non-alphanumerics by spaces and collapsing spaces. |
| `source_problem_sha256` | sha256 of the source problem text before formatting. |
| `router_score`, `route_router` | Score of the CATT router and its route at τ* (`hard` = reasoning path). |
| `nvidia_score`, `route_nvidia` | Score of the NVIDIA classifier and its route at 0.1191. |
| `in_transfer_and_revert_subset` | 1 for the 139 items also used in the ReVerT comparison and the transfer study. |

## `v4/turns.csv` (3,690 rows: 738 items x 5 conditions)

Per-turn metrics computed by the study's frozen per-attempt code from the run records.

| Column | Meaning |
|---|---|
| `key`, `attempt` | Turn key (`formal_v4/MAIN/NNNN`, links all files) and attempt number (run order). |
| `question_id`, `source`, `condition`, `status`, `kind`, `closed`, `open` | Item, condition, completion status, item kind. |
| `route` | `hard` or `easy` as decided at run time for routed CATT turns; `hard` for all-thinking; `single_call` for baselines. |
| `route_confidence`, `route_decided_s` | Router score and time of the route decision. |
| `hard_item`, `hard_item|route`, `hard_item|nvroute` | Whether the item is on the reasoning path in this configuration, in the router configuration, in the NVIDIA configuration (pre-pass lists). |
| `t_first_audio` | Start of the first played sentence. |
| `t_first_substantive`, `first_substantive_unit` | Start and index of the first sentence unit labelled substantive by gpt-5.6-luna. |
| `t_first_progress` | Start of the first progress update (CATT turns routed to reasoning). |
| `t_first_candidate`, `t_final_answer` | Closed-form items: first time any candidate answer is heard, and the time the final answer is heard. |
| `playback_completion` | End of the last played sentence. |
| `correct_strict`, `extract_method`, `extracted` | Strict correctness of the final answer (closed-form items), how the answer was extracted, and the extracted answer (empty for GPQA and AIME). |
| `gap_total`, `gap_*` | Total silence after first audio and its components (bridge to progress, between progress updates, before the final answer, within a dispatch, other, longest single gap). |
| `n_played`, `n_units`, `n_dropped` | Played chunks, sentence units, queued chunks dropped at the final cut-in. |
| `interleaved`, `preempted`, `cut_mid_sentence` | Final/progress interleaving, final cut-in, cut-in in the middle of a sentence. |
| `progress_first`, `progress_regular`, `progress_low_queue` | Progress dispatches by trigger. |
| `final_prompt`, `easy_prompt` | Prompt variant used for the final answer or the easy-path answer. |
| `bridge_audio_s`, `bridge_played_before_easy`, `bridge_guard_v4` | Bridge audio length; bridge audio before an easy answer; bridge answer-guard firings. |
| `norm_changed_chunks`, `norm_long_formula` | Speech-normalizer changes and long formulas replaced. |
| `g2_included`, `intermediate_error` | Whether the turn counts for intermediate-speech errors (CATT turns routed to reasoning; closed-form only if strictly correct) and whether gpt-5.6-luna labelled any update WRONG or REVERSED. The 65 router-configuration turns that this rule leaves out (closed-form, final answer not strictly correct) are labelled in `v4/luna_labels_wrong_final.jsonl.gz`. |
| `llm_calls`, `calls_thinker`, `calls_talker`, `peak_concurrent_llm_calls` | Model calls. |
| `budget_forced_calls`, `forced` | Calls and turns that reached the 16,000-token reasoning budget. |
| `tokens_*` | Prompt, completion and reasoning tokens by engine. |
| `reasoning_sha256` | sha256 of the reasoning text (equal to the thinking baseline's on every CATT turn routed to reasoning). |
| `wall_seconds`, `gpu_energy_j` | Turn wall time and gross energy of the four GPUs over the turn (NVML samples). |

## `v4/luna_labels.jsonl.gz` (3,690 records)

`{"key", "question_id", "condition", "judge": "gpt-5.6-luna", "substantive": ["Y"|"N", ...], "intermediate":
["OK"|"WRONG"|"REVERSED"|"TENTATIVE_CHANGED", ...] or null, "sentences": [...], "updates": [...] or null,
"update_seqs": [...] or null}`. `sentences` are the spoken sentence units the judge saw (first 25 per turn);
`updates` are the bridge and progress updates judged for intermediate errors (CATT turns routed to reasoning that
count, see `g2_included`). Judge free-text notes are not included.

## `v4/luna_labels_retest.jsonl.gz` (511 records)

The gpt-5.6-luna labels of the re-judged sample: a seeded 10 % of the units of each label task, drawn over the whole
run and judged a second time with the same prompt (`judge_luna.py --retest 0.10 --retest-seed 20261009`). Only the
records of the 738 evaluated items are given: 373 turns with re-judged sentence labels and 152 turns with re-judged
update labels. The format and text rules are those of `v4/luna_labels.jsonl.gz`, and the sentences and updates
equal those of the same turn there. `substantive` and `sentences` are null when only the turn's updates were
re-judged; `intermediate`, `updates` and `update_seqs` are null when only its sentences were. `reproduce_paper.py`
compares the two files: the same first substantive sentence in 359 of 373 turns (96.2 %) and the same
intermediate-error flag in 151 of 152 turns (99.3 %).

## `v4/luna_labels_wrong_final.jsonl.gz` (65 records) and `v4/luna_labels_wrong_final_retest.jsonl.gz` (6 records)

Added in release 1.1.0. The gpt-5.6-luna intermediate-error labels of the turns that `g2_included` leaves out in the
router configuration: the 65 closed-form `catt_v4_route` turns routed to reasoning whose final answer is not
strictly correct (`correct_strict` 0; 50 GPQA Diamond, 13 AIME 2026 and 2 MMSU turns). They come from a separate
judging run with the same judge (served snapshot gpt-5.6-luna-2026-07-09), prompt (`judge_g2.md`), settings,
parsing and counting rule as the labels in `v4/luna_labels.jsonl.gz`. The units are built as by
`evaluation/judging/build_luna_units.py`, with its strict-correctness filter inverted to select these turns;
`judge_luna.py --check-prompts` re-renders all 71 recorded prompts of this run (65 turns and 6 re-judged turns) with
identical sha256.

The format and text rules are those of `v4/luna_labels.jsonl.gz` (one record per turn, 549 updates in all; GPQA
and AIME updates as hashes and lengths). `substantive` and `sentences` are null: the sentence labels of the same turns
are in `v4/luna_labels.jsonl.gz`, where `intermediate` and `updates` of these turns stay null. `v4/turns.csv` is
unchanged: for these turns `g2_included` is 0 and `intermediate_error` is empty. Together with `intermediate_error`
of the 345 turns that `g2_included` counts, these labels cover all 410 turns that the router configuration sent to
reasoning. The NVIDIA-router and all-thinking configurations have no such labels. From both sources,
`reproduce_paper.py` computes the intermediate-error rate over all 410 turns routed to reasoning (62 turns, 15.1 %),
over their 168 closed-form turns (58, 34.5 %) and over the 65 turns with a wrong final answer (32, 49.2 %), and it
splits the 168 closed-form turns into correct final answer without (77) and with (26) an intermediate error and
wrong final answer with (32) and without (33) one.

The retest file holds the re-judged sample of these units, drawn with the same step (`--retest 0.10 --retest-seed
20261009` over the 65 units: 6 turns), in the format of `v4/luna_labels_retest.jsonl.gz`. The intermediate-error
flag is the same in 6 of the 6 turns.

## `v4/transcripts/<condition>.jsonl.gz` (738 records each)

Run record of each turn, without the request prompts (which contain the benchmark question):

- `played`: every played chunk with `seq` (dispatch), `mode` (`hard_first` bridge, `hard_mid` progress,
  `hard_final` final answer, `easy` direct answer), `text` (as written by the talker), `spoken` (after the speech
  normalizer, what was synthesized), `synth_ready_at`, `playback_start`, `playback_end`;
- `dropped`: queued chunks that were never played, with the reason;
- `segments`: talker outputs per dispatch;
- `reasoning`: the thinker's reasoning text with `sha256` (null when the text is shorter than 64 characters) and
  `chars`; on CATT turns whose reasoning is byte-identical to the thinking baseline's, the text is given once, in
  the thinking turn (`"identical_to": "thinking turn of the same item"`);
- `calls`: every model call with role, engine, thinking flag, token limit, status, start, first-token and end
  times, prompt and completion tokens, and `output_text` (the visible output of the call);
- `speculative`, `preemption`, `trigger_counts`, `bridge_plan`, `prompt_variant`, `norm_counts`, `reasoning_meta`:
  routing, cut-in, progress-trigger, bridge-sizing, prompt and normalizer details.

Audio files and token-level streaming logs are not released; every timing above is taken from them.

## `listening_rating/exp20261015001_scores.csv`, `exp20261015002_scores.csv`

One row per rated timeline and set, scores 1 to 10 from both judges (`score_luna`, `score_flash_lite`; the paper
also uses their mean). Sets: `main` (CATT router and both baselines, 738 items each), `cfg` (NVIDIA-router and
all-thinking configurations), `revert` (ReVerT, 139 items), `main001` (the `main` scores repeated for the second
file's analyses), `probeA` (validity probe: the thinking baseline's reply with the first sentence moved to 0.5, 3,
10, 30 or 60 s; `level`), `probeB` (CATT reply with 0, 10 or 30 s of silence inserted after the acknowledgment;
`level`), `retest` and `retest2` (the same timelines rated again). Timeline summaries: `first_audio`, `reply_end`,
`n_sentences`, `silence_total`, `first_mode`, `words`, `speech_s`. The judge prompt is
`evaluation/judging/prompts/judge_listen.md`; judge reasons and the reference answers are not included.

## `revert/turns.jsonl` (417 records: 139 items x 3 ReVerT arms)

ReVerT external comparison (`revert_3b`: the authors' released 3B verbalizer, the primary arm; `revert_05b`: their
0.5B verbalizer; `revert_3b_nl1`: the 3B verbalizer with the newline collapsing of its training preprocessing applied
to the reasoning stream), run on the main experiment's recorded thinking-baseline reasoning
stream replayed in real time, with the same speech synthesis and timing. Fields: `t_first_audio`, `t_first_sub`
(gpt-5.6-luna), `t_final_answer`, `correct`, `gap_total`, `playback_completion`, `llm_calls`, `hard` (item on the
reasoning path of the CATT router configuration), `r_end` (end of reasoning), `speech_cover` (share of the reasoning time covered by speech),
segment and verbalizer counts, `reasoning_identical`, `first_sub_mode`, `g2_luna`. The CATT and baseline values
for the same items are in `v4/turns.csv`. The per-turn data and the values computed from them are released; the
adapter code that ran ReVerT's released verbalizer on these reasoning streams is not part of this release.

## `transfer/` (transfer study: CATT on three hosted models, 139 items)

CATT in the router configuration, unchanged and with the router not recalibrated, and the two prompt-matched
baselines, run through hosted APIs on the 139 items flagged in `items.csv`: `deepseek-v4.1-flash` and `glm-5.2`
through a third-party OpenAI-compatible API service, and `qwen3.5-27b-bailian` (Qwen3.5-27B directly through
Alibaba Cloud Bailian, DashScope). Hosted models do not reproduce
reasoning from a seed, so the CATT thinker and the thinking baseline reason separately. These runs used the main
experiment's CATT orchestration as packaged for hosted APIs (it includes the later speech-normalizer fix in
`orchestration_post_study_fix/`) with the main runs' playback and speech path; that harness is not part of this
release.

`transfer/turns.csv` (1,251 rows: 3 runs x 139 items x 3 conditions):

| Column | Meaning |
|---|---|
| `run`, `analysis_order`, `key`, `attempt` | Run, the item order used by the study's loader (its continuity bootstrap depends on it), turn key, attempt number. |
| `question_id`, `source`, `condition`, `closed`, `open`, `complete`, `route`, `route_ts` | Item, condition (`catt_v4_route`, `thinking`, `non_thinking`), completion, CATT route and its decision time. |
| `t_first_audio`, `t_first_substantive`, `first_substantive_unit`, `first_sub_mode`, `labels_present` | As in `v4/turns.csv` (gpt-5.6-luna labels); `first_sub_mode` is the mode of the first substantive sentence. |
| `correct`, `t_final_answer` | Closed-form items: strict correctness and final-answer time. |
| `t_first_progress`, `bridge_audio_s`, `n_progress_played`, `gap_*` | Progress and silence measures as in `v4/turns.csv`. |
| `g2_included`, `intermediate_error` | Intermediate-speech errors (gpt-5.6-luna), counting rules as in `v4/turns.csv`. |
| `t_reasoning_end`, `reasoning_tokens`, `reasoning_sha256` | End of the model's reasoning stream (CATT thinker or thinking baseline), its token count; the hash is not recorded for these runs. |
| `silence_*`, `silence_attributed_total`, `n_progress_calls`, `n_progress_skip`, `n_progress_cancelled`, `thinker_decode_tok_s` | CATT turns routed to reasoning: silence split by the thinker and talker state (after the reasoning end; a progress call in flight that returns text or SKIP; after a SKIP; no progress dispatched), progress calls and how many returned SKIP or were cancelled, and the thinker's decode rate. |

`transfer/<run>/luna_labels.jsonl.gz` and `transfer/<run>/transcripts.jsonl.gz` have the format of their `v4`
counterparts. The transcripts hold played and dropped chunks, talker segments with start and finish times, and the
model calls (role, thinking flag, status, start, first-token, first-content and elapsed times, finish reason, token
counts). The hosted models' raw reasoning text is not released; `reasoning` gives its end time, characters and tokens.

