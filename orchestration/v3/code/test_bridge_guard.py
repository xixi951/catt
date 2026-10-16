"""Checks of the bridge answer guard (constructed sentences)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catt_v3 import bridge_states_answer as f
cases = [('The answer is D.', True), ('The answer is likely B.', True), ('The correct option here is C.', True),
         ('Answer: 42.', True), ('Option B seems right.', True), ('So the final answer would be 17.', True),
         ('A question about whether this quantum state is entangled and why.', False),
         ('Okay, a question about the half-life of this decay chain.', False),
         ('A question asking which answer choice describes legal positivism least accurately.', False),
         ('You are asking how long it takes to become a CPA.', False)]
ok = sum(f(s) == w for s, w in cases)
for s, w in cases:
    print('OK ' if f(s) == w else 'BAD', w, s)
print(f'{ok}/{len(cases)}')
sys.exit(0 if ok == len(cases) else 1)
