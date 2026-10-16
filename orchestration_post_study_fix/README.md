# Speech-normalizer fix made after the main runs

`speech_norm.py` is the spoken-form normalizer of the main experiment (`orchestration/code/speech_norm.py`) with
two later additions; `speech_norm.diff` shows the change. **It was not used in the main runs.** The main experiment
ran the frozen normalizer, and the paper reports its side effects. The transfer runs (`data/transfer/`) and a separate
digital-human case study used this fixed normalizer.

- Markdown emphasis (`*title*`) was read as "times" in a small share of the turns in every condition. The fix
  removes emphasis asterisks; star notation (`a*`) and products (`2*3`, `k*b`) read as before.
- A spaced dash between words (`Radio - Study`) was read as "minus" in up to 8 turns per condition. The fix keeps
  it as a dash; math dashes (`n - R`, `5 - 3`) still read "minus".

On every chunk of the main experiment's run (26,253 distinct texts) and on held-out runs, every change made by the
fix is one of these two kinds, and no math reading changed. Two misreadings remain: "24/7" is read "24 over 7" and
"C++" "C plus plus". To use the fix, replace `orchestration/code/speech_norm.py` with this file (the run driver
will then refuse to start, because the file no longer matches the frozen hash).
