"""EXP-20261009-001 bring-up check for one talker GPU share (change 2; development items only).

With the talker engine running as an MPS client limited to SHARE percent of the SMs (thinker unlimited), for
each bring-up item (pkg/fixtures/bringup_ids.txt, development pool, disjoint from Dev-D and the main set):
  (a) the thinking request alone on the thinker engine (max MAX_TOKENS tokens, fixed seed),
  (b) the same request while the talker engine serves the CATT pattern (a bridge-sized call at t=0, then a
      progress-sized call with a ~2,000-token prompt every PERIOD s),
prefix caches of both engines reset before each. Recorded: streams identical (a vs b), thinker decode rate in
(a) and (b), every talker call's time to first token and total time. Uses EXP-20261007-001's bring-up client
(pkg/v3/code/bringup_engines.py, read-only).
Usage: python bringup_v5.py OUT.json --share 100 [--max-tokens 1500] [--period 6]
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'v3' / 'code'))
import bringup_engines as B  # noqa: E402
import httpx  # noqa: E402

BRIDGE = ('Say one short spoken sentence of about twelve words that names the topic of this question: '
          'a question about reaction rates and the effect of temperature on a catalysed reaction.')


async def catt_pattern(http, period, stop, seed0):
    filler = ' '.join(f'step {i} works out the next relation and checks it against the earlier one;' for i in range(160))
    lat = []
    o = await B.stream(http, B.K, B.talker_body(BRIDGE, seed0, 40))
    lat.append(('bridge', o['first_t'], o['elapsed']))
    n = 0
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
            break
        except asyncio.TimeoutError:
            pass
        o = await B.stream(http, B.K, B.talker_body('Turn the new reasoning into one short spoken sentence about the '
                                                    'work being done: ' + filler, seed0 + 1 + n, 60))
        lat.append(('progress', o['first_t'], o['elapsed']))
        n += 1
    return lat


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--share', type=int, required=True)
    ap.add_argument('--max-tokens', type=int, default=1500)
    ap.add_argument('--period', type=float, default=6.0)
    args = ap.parse_args()
    ids = [l.strip() for l in open(PKG / 'fixtures' / 'bringup_ids.txt') if l.strip()]
    pool = {json.loads(l)['id']: json.loads(l) for l in open(PKG / 'v3' / 'v2' / 'fixtures' / 'dev_pool.jsonl')}
    items = [pool[i] for i in ids]
    rep = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'share': args.share, 'max_tokens': args.max_tokens,
           'period_s': args.period, 'rows': []}
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=64)) as http:
        for name, root in (('thinker', B.T), ('talker', B.K)):
            rep[f'{name}_version'] = (await http.get(root + '/version')).json()
            rep[f'{name}_models'] = (await http.get(root + '/v1/models')).json()
        for k, it in enumerate(items):
            seed = B.seed_of(20261009, 'bringup', it['id'])
            body = B.thinking_body(it['question'], seed, args.max_tokens)
            await B.reset(http, B.T)
            await B.reset(http, B.K)
            a = await B.stream(http, B.T, body)
            await B.reset(http, B.T)
            await B.reset(http, B.K)
            stop = asyncio.Event()
            pt = asyncio.create_task(catt_pattern(http, args.period, stop, 1000 + 10 * k))
            b = await B.stream(http, B.T, body)
            stop.set()
            lat = await pt
            ta, tb = a['reasoning'] + '\x00' + a['content'], b['reasoning'] + '\x00' + b['content']
            row = {'id': it['id'], 'tokens': [a['usage']['completion_tokens'], b['usage']['completion_tokens']],
                   'identical': ta == tb, 'sha_alone': B.hashlib.sha256(ta.encode()).hexdigest(),
                   'decode_alone': B.rate(a), 'decode_with_talker': B.rate(b), 'talker_calls': lat}
            rep['rows'].append(row)
            print(json.dumps({k2: v for k2, v in row.items() if k2 != 'talker_calls'}), flush=True)
    rows = rep['rows']
    prog = [c for r in rows for c in r['talker_calls'] if c[0] == 'progress']
    rep['summary'] = {'share': args.share, 'identical': sum(r['identical'] for r in rows), 'total': len(rows),
                      'decode_alone_median': statistics.median([r['decode_alone'] for r in rows if r['decode_alone']]),
                      'decode_with_talker_median': statistics.median([r['decode_with_talker'] for r in rows if r['decode_with_talker']]),
                      'progress_ttft_median': statistics.median([c[1] for c in prog]) if prog else None,
                      'progress_total_median': statistics.median([c[2] for c in prog]) if prog else None,
                      'n_progress_calls': len(prog), 'sha_alone': {r['id']: r['sha_alone'] for r in rows}}
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    Path(args.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep['summary']))


asyncio.run(main())
