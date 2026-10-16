"""Unit checks for scorer_v4 (constructed strings, no outcome data)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scorer_v4 as S

Q = "Which is it?\nA. alpha ester\nB. ethyl ester\nC. methyl ketone\nD. none of these\n\nWhat is the answer to the above multiple choice question? Select one of the following: A, B, C, or D."
cases = [
    ('gpqa_diamond', 'The least accurate description of legal positivism is B. It holds that law is a social fact.', 'B'),
    ('gpqa_diamond', 'The correct choice here would be (C), because the ring opens.', 'C'),
    ('gpqa_diamond', 'The answer is option D, none of these.', 'D'),
    ('gpqa_diamond', 'The product is B, the ethyl ester, since A is ruled out.', 'B'),
    ('gpqa_diamond', 'It is not B. The answer is C.', 'C'),
    ('voicebench_mmsu', 'Vitamin A is the one that matters here, so the answer is D.', 'D'),
]
ok = 0
for src, text, want in cases:
    chunks = [{'seq': 1, 'mode': 'hard_final', 'text': text, 'playback_start': 1.0}]
    x, _ = S.score_chunks(src, want, chunks, Q)
    ok += x.answer == want
    print('OK ' if x.answer == want else 'BAD', want, x.answer, x.method, '|', text[:60])
# candidate timing must skip the bridge and the article
chunks = [{'seq': 0, 'mode': 'hard_first', 'text': 'A question about ester hydrolysis, and which product forms.', 'playback_start': 0.8},
          {'seq': 1, 'mode': 'hard_mid', 'text': 'Now checking the carbonyl carbon after the base attacks.', 'playback_start': 4.5},
          {'seq': 2, 'mode': 'hard_mid', 'text': 'It might be B, the ethyl ester, but this is being checked.', 'playback_start': 9.0},
          {'seq': 3, 'mode': 'hard_final', 'text': 'The answer is A, the alpha ester.', 'playback_start': 15.0},
          {'seq': 3, 'mode': 'hard_final', 'text': 'A check of the second step confirms it.', 'playback_start': 17.0}]
t = S.answer_times('gpqa_diamond', 'A', Q, chunks)
print(t)
ok += t['first_candidate'] == 9.0 and t['final_answer'] == 15.0 and t['final'] == 'A'
chunks2 = [{'seq': 0, 'mode': 'easy', 'text': 'A catalyst lowers the barrier, so the answer is C.', 'playback_start': 0.5}]
t2 = S.answer_times('gpqa_diamond', 'C', Q, chunks2)
print(t2)
ok += t2['final'] == 'C' and t2['final_answer'] == 0.5
aime = [{'seq': 0, 'mode': 'hard_first', 'text': 'A counting question about 2024 and its divisors.', 'playback_start': 0.6},
        {'seq': 4, 'mode': 'hard_final', 'text': 'The answer is 30.', 'playback_start': 20.0}]
t3 = S.answer_times('aime_2025', '30', '', aime)
print(t3)
ok += t3['final'] == '30' and t3['final_answer'] == 20.0 and t3['first_candidate'] == 20.0
print(f'{ok}/{len(cases) + 3} checks passed')
sys.exit(0 if ok == len(cases) + 3 else 1)
