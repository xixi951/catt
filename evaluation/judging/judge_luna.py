"""Substantive-content and intermediate-speech labels with gpt-5.6-luna, as used in the paper.

Derived from the study's frozen judge script without changing what the judge sees: the same prompt templates
(prompts/judge_substantive.md, prompts/judge_g2.md), the same prompt construction, the same parser and format
reminder, reasoning_effort "medium", max_tokens 16000, default sampling, up to 3 attempts per unit. Release changes:
only the two label tasks the paper reports, a single judge (gpt-5.6-luna), and credentials from environment
variables (api_client.py). `--check-prompts` re-renders the prompts of recorded labels and compares their sha256
with the recorded `prompt_sha256` without calling any model.

Units (one JSON object per line) are built from the run records by orchestration/code/make_units_v5.py; only the
tasks "substantive" and "g2" are judged here:
  substantive  {"task", "key", "question", "sentences": [...], "offset": 0}
  g2           {"task", "key", "question", "reference", "final", "updates": [...]}

Usage: python judge_luna.py UNITS.jsonl OUT_DIR [--workers 24] [--retest FRACTION --retest-seed 20261009]
       python judge_luna.py UNITS.jsonl LABEL_DIR --check-prompts
Output: OUT_DIR/<task>__gpt-5.6-luna.jsonl (or ...__retest.jsonl), resumable by (key, offset).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JUDGE = 'gpt-5.6-luna'
PARAMS = {'reasoning_effort': 'medium'}
MAX_TOKENS = 16000
TASKS = ('substantive', 'g2')
TEMPLATES = {t: (Path(__file__).resolve().parent / 'prompts' / f'judge_{t}.md').read_text(encoding='utf-8')
             for t in TASKS}
G2_LABELS = ('OK', 'WRONG', 'REVERSED', 'TENTATIVE_CHANGED')
REMINDER = ('Your previous reply could not be used. Reply again with JSON only, exactly in the requested format, '
            'with one entry per numbered item.')


def build_prompt(u: dict) -> Tuple[str, int]:
    t = u['task']
    if t == 'substantive':
        sents = u['sentences']
        numbered = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(sents))
        return TEMPLATES[t].format(question=u['question'], numbered=numbered), len(sents)
    if t == 'g2':
        ups = u['updates']
        numbered = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(ups))
        return TEMPLATES[t].format(question=u['question'], reference=u['reference'], final=u['final'] or '(none)',
                                   numbered=numbered), len(ups)
    raise ValueError(t)


def parse(task: str, text: str, n: int) -> Optional[Any]:
    if text is None:
        return None
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    labels = obj.get('labels')
    if not isinstance(labels, list) or len(labels) != n:
        return None
    out = []
    if task == 'substantive':
        for x in labels:
            x = str(x).strip().upper()
            if x not in ('Y', 'N'):
                return None
            out.append(x == 'Y')
        return out
    if task == 'g2':
        for x in labels:
            lab = str((x or {}).get('label', '')).strip().upper() if isinstance(x, dict) else str(x).strip().upper()
            if lab not in G2_LABELS:
                return None
            out.append({'label': lab, 'note': (x.get('note') if isinstance(x, dict) else '')})
        return out
    return None


def judge_unit(client, u: dict) -> dict:
    prompt, n = build_prompt(u)
    messages = [{'role': 'user', 'content': prompt}]
    rec = {'task': u['task'], 'key': u['key'], 'offset': u.get('offset', 0), 'judge': JUDGE, 'params': PARAMS,
           'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'n_items': n, 'attempts': [],
           'ok': False, 'result': None}
    for _ in range(3):
        try:
            r = client.chat(JUDGE, messages, max_tokens=MAX_TOKENS, extra=dict(PARAMS))
        except Exception as exc:  # network/API failure after client retries
            rec['attempts'].append({'error': str(exc)[:300]})
            continue
        msg = (r.get('choices') or [{}])[0].get('message') or {}
        text = msg.get('content')
        rec['attempts'].append({'text': text, 'usage': r.get('usage') or {}, 'model': r.get('model'),
                                'elapsed_s': r.get('_elapsed_s'), 'finish': (r.get('choices') or [{}])[0].get('finish_reason')})
        parsed = parse(u['task'], text, n)
        if parsed is not None:
            rec['ok'], rec['result'] = True, parsed
            break
        messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': text or ''},
                    {'role': 'user', 'content': REMINDER}]
    rec['finished'] = time.time()
    return rec


def check_prompts(units: List[dict], labels: Path) -> None:
    """Re-render prompts and compare with the prompt_sha256 recorded in existing label files (no model call)."""
    by = {(u['task'], u['key'], u.get('offset', 0)): u for u in units}
    n = bad = 0
    for task in TASKS:
        for p in sorted(labels.glob(f'{task}__{JUDGE}*.jsonl')):
            for line in open(p, encoding='utf-8'):
                if not line.strip():
                    continue
                r = json.loads(line)
                u = by.get((task, r['key'], r.get('offset', 0)))
                if u is None:
                    continue
                n += 1
                bad += hashlib.sha256(build_prompt(u)[0].encode()).hexdigest() != r['prompt_sha256']
    print(json.dumps({'checked': n, 'mismatched': bad}))
    raise SystemExit(1 if bad or not n else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('units')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--retest', type=float, default=0.0)
    ap.add_argument('--retest-seed', type=int, default=20261009)
    ap.add_argument('--check-prompts', action='store_true')
    args = ap.parse_args()
    units = [json.loads(l) for l in open(args.units, encoding='utf-8') if l.strip()]
    units = [u for u in units if u['task'] in TASKS]
    if args.check_prompts:
        check_prompts(units, Path(args.out))
    if args.retest:
        rng = random.Random(f'{args.retest_seed}-retest')
        by_task: Dict[str, List[dict]] = {}
        for u in units:
            if u.get('offset', 0) == 0:
                by_task.setdefault(u['task'], []).append(u)
        units = []
        for t, us in sorted(by_task.items()):
            us = sorted(us, key=lambda x: x['key'])
            units += rng.sample(us, max(1, round(args.retest * len(us))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import api_client  # noqa: E402
    client = api_client.ChatClient()
    suffix = '__retest' if args.retest else ''
    locks: Dict[Path, threading.Lock] = {}
    jobs = []
    for task in sorted({u['task'] for u in units}):
        path = out / f'{task}__{JUDGE}{suffix}.jsonl'
        locks[path] = threading.Lock()
        done = set()
        if path.exists():
            for l in open(path, encoding='utf-8'):
                if l.strip():
                    r = json.loads(l)
                    if r.get('ok'):
                        done.add((r['key'], r.get('offset', 0)))
        jobs += [(u, path) for u in units if u['task'] == task and (u['key'], u.get('offset', 0)) not in done]
    print(f'{len(jobs)} judge calls to make', flush=True)
    t0 = time.time()
    n_done = n_fail = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_unit, client, u): (u, path) for u, path in jobs}
        for f in cf.as_completed(futs):
            u, path = futs[f]
            rec = f.result()
            with locks[path]:
                with open(path, 'a', encoding='utf-8') as h:
                    h.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_done += 1
            n_fail += not rec['ok']
            if n_done % 50 == 0 or n_done == len(jobs):
                print(f'{n_done}/{len(jobs)} done, {n_fail} failed, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
