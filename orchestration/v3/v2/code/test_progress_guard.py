"""Unit test of the progress-sentence guard (change 3): unhedged answer claims are suppressed,
hedged candidates and ordinary content are kept. Run on newsrv: python test_progress_guard.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catt_v2 as V  # noqa: E402

cases = [
    ('The answer is D.', True),
    ('So the correct answer is option B, the ethyl ester.', True),
    ('Option C is correct.', True),
    ('B is the correct answer.', True),
    ('Tentatively, the answer is D.', False),
    ('My current guess is that the answer is C.', False),
    ('The answer may be B, but the second case is still being checked.', False),
    ('The most deshielded protons are on the CH2 next to the carbonyl.', False),
    ('Now working out how many CH2 groups the ring contributes.', False),
    ('This matches option D.', False),
    ('The ratio of the two eigenvalues sets the correlation length.', False),
]
bad = [(s, want, V.unhedged_answer_claim(s)) for s, want in cases if V.unhedged_answer_claim(s) != want]
for s, want, got in bad:
    print('FAIL', repr(s), 'want', want, 'got', got)
print(f'{len(cases) - len(bad)}/{len(cases)} guard cases passed')
sys.exit(1 if bad else 0)
