# Matched baseline prompt (EXP-20260930-003 v2)

`baseline_matched.md` is used verbatim (Python `str.format(user_text=...)`, the same mechanism as the CATT talker prompts) by both single-call baselines. The only difference between the two baselines is the thinking flag.

Provenance of each part, relative to the CATT prompts in `snapshot/prompts/` (unchanged system under test):

- System Role: `talker_hard_final.md` System Role first sentence pair ("Your output goes directly into TTS for speech, so it must be conversational, accurate, pedagogical, and easy to follow by ear"), with the formatting ban and language rule of `talker_easy.md` (adds "tables" and "long parenthetical asides").
- Goal: "State the conclusion first, then give only the necessary explanation that helps the user trust and learn from the result" (`talker_hard_final.md`); "answer directly in 1-3 sentences" (`talker_easy.md`); "usually last about 10-18 seconds when read aloud: normally 3-5 spoken sentences. Include the final answer, then 1-2 decisive reasoning steps or checks" (`talker_hard_final.md`).
- Caveat, mathematical-answer and no-internal-details sentences: `talker_hard_final.md`. Edits: "reasoning result" became "result"; "just state the integer" became "just state the answer" (the fixture contains multiple-choice and open-ended items).
- Output constraints: `talker_hard_final.md`, minus the clause about avoiding repetition of earlier speech (a single call has no earlier speech).
- Removed because they do not exist in a single call: thinker history, talker history, cross-turn dialogue (always empty in this isolated run), and the hard_first/hard_mid modes.

The question text is passed unchanged, exactly as CATT's talker receives it, including the per-benchmark answer instructions contained in the fixture questions.
