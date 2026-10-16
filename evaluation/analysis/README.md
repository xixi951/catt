# Analysis

`build_turns.py` builds the per-turn table `data/v4/turns.csv` from run records, the fixture, the gpt-5.6-luna labels
and the GPU energy samples; on the study's records its output equals the released file byte for byte.

`reproduce_paper.py` recomputes the main reported values (see the top-level README) from the released per-item data
(`data/`): the main experiment on 738 items for all three CATT configurations against both baselines (first audio,
first substantive content, first candidate and final answer, playback completion, silence, accuracy, calls, tokens,
energy, intermediate-speech errors, routing, reasoning identity), the test-retest agreement of the gpt-5.6-luna
labels, the listening rating (contrasts with the baselines and of ReVerT with CATT, validity probes, and
exact-agreement summaries of test-retest and inter-judge agreement, which are not the ICCs reported in the paper),
the ReVerT comparison, and the transfer study on three hosted models (contrasts, continuity, accuracy, intermediate
errors, and the final-answer time measured from each condition's own reasoning end). The top-level README lists
reported results that it does not recompute.

Statistics follow the study's analysis code: paired differences per item, items sorted by question id (silence on
turns routed to reasoning: run order), median for times and mean for accuracy and ratings, 95 % percentile
bootstrap with 10,000 draws from `numpy.random.default_rng(20261009)` (transfer study: seed 20261011, as in that
study), Wilson intervals for rates.

```bash
python reproduce_paper.py --data ../../data --out results --check expected/paper_numbers_expected.json
```

`expected/paper_numbers_expected.json` holds 5,968 values as reported by the study's own analyses (the test-retest
agreement of the gpt-5.6-luna labels as reported in the paper: computed from the study's label files on the turns of
the 738 evaluated items), keyed like the script's output. The check passes with 0 differences (relative tolerance
1e-9). Keys follow the pattern `metric|condition-baseline|subset`; subsets are `all`, `closed`, `open`, `hard`
(turns routed to reasoning), `easy`, and their intersections.
