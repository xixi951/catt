"""Synthetic checks of the loop-guard predicate (constructed strings, no model output)."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catt_v3 import repeated_tail, LOOP_PROBE_CHARS as P, LOOP_WINDOW_CHARS as W, LOOP_MIN_COUNT as K, \
    CONT_PROBE_CHARS as CP, CONT_WINDOW_CHARS as CW, CONT_MIN_COUNT as CK

rng = random.Random(1)
words = 'the energy level spin orbital ratio factor compute check value term sum product hence wait so'.split()
natural = ' '.join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(3000))
para = 'Wait, let me recompute the ratio of the two terms again carefully, because the sign may be wrong here. ' * 2
loop = natural[:5000] + para * 30
ok = [not repeated_tail(natural, P, W, K), repeated_tail(loop, P, W, K),
      not repeated_tail(natural[:1000], P, W, K), repeated_tail('The answer is B. ' * 60, CP, CW, CK),
      not repeated_tail('The answer is B, because the ring opens and the ester forms. ' * 2, CP, CW, CK)]
print(ok, f'{sum(ok)}/{len(ok)}')
sys.exit(0 if all(ok) else 1)
