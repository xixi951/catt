"""Listening rating with the full timeline (the paper's quality measure), judged by gpt-5.6-luna and
gemini-3.1-flash-lite.

Derived from the study's judge script without changing what the judges see: the same template
(prompts/judge_listen.md), prompt construction, score parser (an integer 1-10 in {"score": ..., "reason": ...}),
format reminder, reasoning_effort "medium", max_tokens 16000, default sampling, up to 3 attempts per unit, and a
spend guard. Release changes: credentials from environment variables (api_client.py) and the parser inlined.
`--check-prompts LABEL_DIR` re-renders the prompts of recorded labels and compares their sha256 with the recorded
`prompt_sha256` without calling any model.

Units are built from the run records by build_listening_units.py.

Usage: python judge_listen.py --units units_<set>.jsonl ... --out LAB_DIR [--judges gpt-5.6-luna,gemini-3.1-flash-lite]
                              [--workers 24] [--cap-usd 18]
       python judge_listen.py --units units_<set>.jsonl ... --out LAB_DIR --check-prompts
Output: LAB_DIR/listen__<judge>__<set>.jsonl, resumable by key (units with an ok record are skipped).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

JUDGES = ('gpt-5.6-luna', 'gemini-3.1-flash-lite')
PARAMS = {'reasoning_effort': 'medium'}
MAX_TOKENS = 16000
PRICES = {'gpt-5.6-luna': (1.25, 10.0), 'gemini-3.1-flash-lite': (0.50, 3.00)}   # conservative (in, out) USD/M
TEMPLATE = (Path(__file__).resolve().parent / 'prompts' / 'judge_listen.md').read_text(encoding='utf-8')
REMINDER = ('Your previous reply could not be used. Reply again with JSON only, exactly in the requested format, '
            'with one entry per numbered item.')


def build_prompt(u: dict) -> str:
    return TEMPLATE.format(question=u['question'], reference=u['reference'], timeline=u['timeline'])


def parse_score(text: Optional[str]) -> Optional[dict]:
    if text is None:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        s = int(obj.get('score'))
    except (TypeError, ValueError):
        return None
    if not 1 <= s <= 10:
        return None
    return {'score': s, 'reason': obj.get('reason', '')}


def call_cost(judge: str, usage: dict) -> float:
    pin, pout = PRICES[judge]
    return ((usage.get('prompt_tokens') or 0) * pin + (usage.get('completion_tokens') or 0) * pout) / 1e6


class Guard:
    def __init__(self, cap: float, start: float) -> None:
        self.cap, self.spent, self.lock, self.stopped = cap, start, threading.Lock(), False

    def add(self, x: float) -> None:
        with self.lock:
            self.spent += x
            if self.spent >= self.cap:
                self.stopped = True


def judge_unit(client, judge: str, u: dict, guard: Guard) -> Optional[dict]:
    if guard.stopped:
        return None
    prompt = build_prompt(u)
    messages = [{'role': 'user', 'content': prompt}]
    rec = {'task': 'listen', 'key': u['key'], 'judge': judge, 'params': PARAMS,
           'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'attempts': [], 'ok': False, 'result': None}
    for _ in range(3):
        try:
            r = client.chat(judge, messages, max_tokens=MAX_TOKENS, extra=dict(PARAMS))
        except Exception as exc:  # network/API failure after client retries
            rec['attempts'].append({'error': str(exc)[:300]})
            continue
        msg = (r.get('choices') or [{}])[0].get('message') or {}
        text = msg.get('content')
        usage = r.get('usage') or {}
        guard.add(call_cost(judge, usage))
        rec['attempts'].append({'text': text, 'usage': usage, 'model': r.get('model'), 'elapsed_s': r.get('_elapsed_s'),
                                'finish': (r.get('choices') or [{}])[0].get('finish_reason')})
        parsed = parse_score(text)
        if parsed is not None:
            rec['ok'], rec['result'] = True, parsed
            break
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': text or ''},
                    {'role': 'user', 'content': REMINDER}]
    rec['finished'] = time.time()
    return rec


def spent_so_far(out: Path) -> float:
    total = 0.0
    for p in out.glob('listen__*.jsonl'):
        judge = p.name.split('__')[1]
        for l in open(p, encoding='utf-8'):
            if l.strip():
                for a in json.loads(l).get('attempts', []):
                    total += call_cost(judge, a.get('usage') or {})
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--units', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--judges', default=','.join(JUDGES))
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--cap-usd', type=float, default=18.0)
    ap.add_argument('--check-prompts', action='store_true')
    args = ap.parse_args()
    sets = {}
    for f in args.units:
        name = Path(f).stem.replace('units_', '')
        sets[name] = [json.loads(l) for l in open(f, encoding='utf-8') if l.strip()]
    out = Path(args.out)
    if args.check_prompts:
        n = bad = 0
        for name, rows in sets.items():
            by = {u['key']: u for u in rows}
            for p in sorted(out.glob(f'listen__*__{name}.jsonl')):
                for l in open(p, encoding='utf-8'):
                    if l.strip():
                        r = json.loads(l)
                        if r['key'] in by:
                            n += 1
                            bad += hashlib.sha256(build_prompt(by[r['key']]).encode()).hexdigest() != r['prompt_sha256']
        print(json.dumps({'checked': n, 'mismatched': bad}))
        raise SystemExit(1 if bad or not n else 0)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import api_client  # noqa: E402
    client = api_client.ChatClient()
    out.mkdir(parents=True, exist_ok=True)
    guard = Guard(args.cap_usd, spent_so_far(out))
    print(f'template sha256 {hashlib.sha256(TEMPLATE.encode()).hexdigest()}; estimated spend so far USD {guard.spent:.3f}',
          flush=True)
    locks: Dict[Path, threading.Lock] = {}
    jobs: List[tuple] = []
    for name, rows in sets.items():
        for judge in args.judges.split(','):
            path = out / f'listen__{judge}__{name}.jsonl'
            locks[path] = threading.Lock()
            done = set()
            if path.exists():
                for l in open(path, encoding='utf-8'):
                    if l.strip():
                        r = json.loads(l)
                        if r.get('ok'):
                            done.add(r['key'])
            jobs += [(judge, u, path) for u in rows if u['key'] not in done]
    order = []   # interleave the two judges so a stall in one does not idle the pool
    a = [j for j in jobs if j[0] == JUDGES[0]]
    b = [j for j in jobs if j[0] != JUDGES[0]]
    for i in range(max(len(a), len(b))):
        order += a[i:i + 1] + b[i:i + 1]
    print(f'{len(order)} judge calls to make', flush=True)
    t0 = time.time()
    n_done = n_fail = n_skip = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_unit, client, judge, u, guard): (judge, u, path) for judge, u, path in order}
        for f in cf.as_completed(futs):
            judge, u, path = futs[f]
            rec = f.result()
            if rec is None:
                n_skip += 1
                continue
            with locks[path]:
                with open(path, 'a', encoding='utf-8') as h:
                    h.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_done += 1
            n_fail += not rec['ok']
            if n_done % 100 == 0 or n_done + n_skip == len(order):
                print(f'{n_done}/{len(order)} done, {n_fail} failed, {n_skip} skipped, '
                      f'est. USD {guard.spent:.2f}, {time.time() - t0:.0f}s', flush=True)
    if guard.stopped:
        print(f'SPEND CAP REACHED: est. USD {guard.spent:.2f} >= {args.cap_usd}', flush=True)


if __name__ == '__main__':
    main()
