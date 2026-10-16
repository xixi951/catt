"""EXP-20261007-001 bring-up of the two-engine deployment (engineering evidence only; development items).

Checks
 1. identity of both engines (/version, /v1/models root = weights revision)
 2. speed: thinker single-stream decode alone; talker TTFT/decode alone; thinker decode while the talker
    engine serves a continuous progress-sized load (the CATT situation)
 3. reproducibility of solo thinker streams: N development items (not in Dev-A/Dev-B), each run
    (a) alone, (b) alone again, (c) alone on the thinker engine while the talker engine is under load;
    prefix cache of the thinker engine reset before each; streams compared character by character
 4. worst case (--long): one 17,000-token thinker generation with talker load; preemption counters of
    both engines must not change
Usage: python bringup_engines.py OUT.json [--n 20] [--max-tokens 3000] [--long]
"""
import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path

import httpx

T = 'http://127.0.0.1:18011'
K = 'http://127.0.0.1:18012'
PKG = Path(__file__).resolve().parents[1]
TEMPLATE = (PKG / 'v2/exp003/prompts_v2/baseline_matched.md').read_text(encoding='utf-8')
REV = '9216db5781bf21249d130ec9da846c4624c16137'


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2 ** 31)


def thinking_body(question, seed, max_tokens):
    return {'model': 'Qwen3-32B', 'messages': [{'role': 'user', 'content': TEMPLATE.format(user_text=question)}],
            'stream': True, 'max_tokens': max_tokens, 'seed': seed, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
            'chat_template_kwargs': {'enable_thinking': True},
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}


def talker_body(prompt, seed, max_tokens):
    return {'model': 'Qwen3-32B', 'messages': [{'role': 'user', 'content': prompt + ' /no_think'}], 'stream': True,
            'max_tokens': max_tokens, 'seed': seed, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
            'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}


async def stream(http, root, body):
    t0 = time.monotonic()
    out = {'reasoning': '', 'content': '', 'first_t': None, 'usage': None, 'finish': None, 'tok_times': []}
    async with http.stream('POST', root + '/v1/chat/completions', json=body, timeout=3600) as resp:
        assert resp.status_code == 200, (await resp.aread())[:300]
        async for line in resp.aiter_lines():
            if not line.startswith('data:') or line.strip() == 'data: [DONE]':
                continue
            ch = json.loads(line[5:])
            if ch.get('usage'):
                out['usage'] = ch['usage']
                out['tok_times'].append((time.monotonic() - t0, ch['usage'].get('completion_tokens')))
            for c in ch.get('choices', []):
                d = c.get('delta') or {}
                r = d.get('reasoning_content') or d.get('reasoning') or ''
                if (r or d.get('content')) and out['first_t'] is None:
                    out['first_t'] = time.monotonic() - t0
                out['reasoning'] += r
                out['content'] += d.get('content') or ''
                out['finish'] = c.get('finish_reason') or out['finish']
    out['elapsed'] = time.monotonic() - t0
    return out


def rate(o):
    tt = o['tok_times']
    if len(tt) < 20:
        return None
    a, b = tt[5], tt[-1]
    return (b[1] - a[1]) / (b[0] - a[0]) if b[0] > a[0] else None


async def metrics(http, root):
    txt = (await http.get(root + '/metrics', timeout=10)).text
    out = {}
    for line in txt.splitlines():
        for name in ('vllm:num_preemptions_total', 'vllm:num_requests_running', 'vllm:num_requests_waiting'):
            if line.startswith(name + '{') or line.startswith(name + ' '):
                out[name] = out.get(name, 0.0) + float(line.rsplit(' ', 1)[1])
    return out


async def reset(http, root):
    return (await http.post(root + '/reset_prefix_cache', timeout=30)).status_code


class Load:
    """Back-to-back progress-sized talker calls (about 2,000 prompt tokens, 120 output tokens)."""

    def __init__(self, http):
        self.http, self.stop, self.n, self.ttft = http, False, 0, []

    async def run(self):
        filler = ' '.join(f'step {i} checks the intermediate relation again;' for i in range(330))
        while not self.stop:
            o = await stream(self.http, K, talker_body('Summarize for a listener in two sentences what is being done: '
                                                       + filler, 7 + self.n, 120))
            self.n += 1
            self.ttft.append(o['first_t'])


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--long', action='store_true')
    args = ap.parse_args()
    rep = {'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    pool = [json.loads(l) for l in open(PKG / 'v2/fixtures/dev_pool.jsonl')]
    used = {json.loads(l)['id'] for f in ('dev_A.jsonl', 'dev_B.jsonl') for l in open(PKG / 'v2/fixtures' / f)}
    cand = [r for r in pool if r['id'] not in used and r['source'] != 'voicebench_alpaca_eval']
    random.Random(20261008).shuffle(cand)
    items = cand[:args.n]
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=64)) as http:
        for name, root in (('thinker', T), ('talker', K)):
            rep[f'{name}_version'] = (await http.get(root + '/version')).json()
            models = (await http.get(root + '/v1/models')).json()
            rep[f'{name}_root_ok'] = REV in json.dumps(models)
            rep[f'{name}_reset_status'] = await reset(http, root)
        q = items[0]['question']
        solo = await stream(http, T, thinking_body(q, 1, 1500))
        rep['thinker_solo_decode_tok_s'] = rate(solo)
        tk = await stream(http, K, talker_body('Explain in three sentences why the sky is blue.', 3, 150))
        rep['talker_solo'] = {'ttft_s': tk['first_t'], 'decode_tok_s': rate(tk)}
        load = Load(http)
        lt = asyncio.create_task(load.run())
        await asyncio.sleep(1.0)
        busy = await stream(http, T, thinking_body(q, 1, 1500))
        load.stop = True
        await lt
        rep['thinker_decode_under_talker_load_tok_s'] = rate(busy)
        rep['talker_load_calls'] = load.n
        rep['talker_ttft_under_thinker_s_median'] = sorted(x for x in load.ttft if x)[len(load.ttft) // 2] if load.ttft else None
        rep['solo_vs_loaded_identical_1500'] = solo['reasoning'] + solo['content'] == busy['reasoning'] + busy['content']
        rows = []
        for it in items:
            seed = seed_of(20261008, 'bringup', it['id'])
            body = thinking_body(it['question'], seed, args.max_tokens)
            outs = []
            for mode in ('solo', 'solo2', 'loaded'):
                await reset(http, T)
                if mode == 'loaded':
                    load = Load(http)
                    lt = asyncio.create_task(load.run())
                    await asyncio.sleep(0.5)
                o = await stream(http, T, body)
                if mode == 'loaded':
                    load.stop = True
                    await lt
                outs.append(o)
            a, b, c = (o['reasoning'] + '\x00' + o['content'] for o in outs)
            rows.append({'id': it['id'], 'tokens': [o['usage']['completion_tokens'] for o in outs],
                         'finish': [o['finish'] for o in outs], 'chars': len(a),
                         'solo_solo_identical': a == b, 'solo_loaded_identical': a == c,
                         'solo_solo_prefix': common_prefix(a, b), 'solo_loaded_prefix': common_prefix(a, c),
                         'decode_solo': rate(outs[0]), 'decode_loaded': rate(outs[2])})
            print(json.dumps(rows[-1]), flush=True)
        rep['reproducibility'] = {'n': len(rows), 'solo_solo_identical': sum(r['solo_solo_identical'] for r in rows),
                                  'solo_loaded_identical': sum(r['solo_loaded_identical'] for r in rows), 'rows': rows}
        if args.long:
            m0 = {n: await metrics(http, r) for n, r in (('thinker', T), ('talker', K))}
            longb = thinking_body(items[0]['question'] * 4, 11, 17000)
            longb['ignore_eos'] = True
            load = Load(http)
            lt = asyncio.create_task(load.run())
            lr = await stream(http, T, longb)
            load.stop = True
            await lt
            m1 = {n: await metrics(http, r) for n, r in (('thinker', T), ('talker', K))}
            rep['long'] = {'completion_tokens': lr['usage']['completion_tokens'], 'elapsed_s': lr['elapsed'],
                           'decode_tok_s': rate(lr), 'talker_calls_alongside': load.n,
                           'preemptions_delta': {n: m1[n].get('vllm:num_preemptions_total', 0) - m0[n].get('vllm:num_preemptions_total', 0)
                                                 for n in m0}}
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    Path(args.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'reproducibility'}, indent=1))
    print(json.dumps({k: v for k, v in rep['reproducibility'].items() if k != 'rows'}))


if __name__ == '__main__':
    asyncio.run(main())
