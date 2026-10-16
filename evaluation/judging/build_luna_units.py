"""Judge units for the gpt-5.6-luna labels (substantive content, intermediate-speech errors) from run records.

Derived from the study's frozen unit builder with the same units and the same judge input: the unit of judgment is
the played sentence (consecutive chunks of one dispatch merged until a chunk ends a sentence), judges see the
spoken form, substantive units hold the first 25 sentences of every turn (offset 0), and intermediate-error units
hold the bridge and progress updates of CATT turns routed to reasoning (closed-form turns only when the final answer
is strictly correct). Release changes: only the two label tasks the paper reports, and the follow-up rule reads
the luna labels only (a full-turn unit at offset 25 for turns where luna marked none of the first 25 sentences
substantive; in the study no turn needed one). Model-free.

Needs the run records (one directory per attempt with summary.json, as written by orchestration/code/run_v5.py)
and the benchmark fixture rebuilt as described in docs/BENCHMARKS.md.

Usage: python build_luna_units.py --runs RUN_DIR[:LABEL] ... --fixtures main.jsonl --out UNITS.jsonl [--followup LABEL_DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ORCH = Path(__file__).resolve().parents[2] / 'orchestration'
sys.path.insert(0, str(ORCH / 'code'))
import v5_common as C  # noqa: E402  (frozen helpers of the main experiment)

S = C.S
JUDGE = 'gpt-5.6-luna'
CATT = ('catt_v4_route', 'catt_v4_nvroute', 'catt_v4_allhard')
FIRST_N = 25
_OPT = re.compile(r'^\s*([A-D])[.)]\s+(.+?)\s*$', re.M)


def route_of(cell: Path, s: dict) -> str:
    if s['condition'] == 'catt_v4_allhard':
        return 'hard'
    if s['condition'] not in ('catt_v4_route', 'catt_v4_nvroute'):
        return 'single_call'
    ws = cell / s['ws_log'] if s.get('ws_log') else None
    if ws and ws.exists():
        for e in C.read_jsonl(ws):
            if e.get('kind') == 'route_decision':
                return e['data'].get('route')
    sp = s.get('speculative') or {}
    return sp.get('route') or 'unknown'


def reference(q: dict) -> str:
    if q['source'] in S.MCQ_SOURCES:
        opts = dict(_OPT.findall(q['question']))
        return f"{q['answer']}" + (f" ({opts[q['answer']]})" if q['answer'] in opts else '')
    if q['source'] in S.AIME_SOURCES:
        return str(q['answer'])
    return '(none: open-ended request; judge WRONG against established facts only)'


def iter_attempts(runs):
    for spec in runs:
        run, _, label = spec.partition(':')
        run = Path(run)
        label = label or run.parent.name
        for cell, s in C.iter_run(run):
            yield label, cell, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--fixtures', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--followup', default=None)
    args = ap.parse_args()
    fx = {r['id']: r for f in args.fixtures for r in C.read_jsonl(f)}
    need_full = None
    if args.followup:
        any_y = {}
        p = Path(args.followup) / f'substantive__{JUDGE}.jsonl'
        if p.exists():
            for r in C.read_jsonl(p):
                if r.get('ok') and r.get('offset', 0) == 0:
                    any_y[r['key']] = any_y.get(r['key'], False) or any(r['result'])
        need_full = {k for k, v in any_y.items() if not v}
    n = {'substantive': 0, 'g2': 0, 'g2_excluded_closed_wrong': 0}
    with open(args.out, 'w', encoding='utf-8') as out:
        for label, cell, s in iter_attempts(args.runs):
            q = fx[s['question_id']]
            key = f"{label}/{s['set']}/{s['attempt']:04d}"
            chunks = C.played(s)
            units = C.sentence_units(chunks)
            sents = [u['spoken'] for u in units]
            if need_full is not None:
                if key in need_full and len(sents) > FIRST_N:
                    out.write(json.dumps({'task': 'substantive', 'key': key, 'offset': FIRST_N, 'question': q['question'],
                                          'sentences': sents}, ensure_ascii=False) + '\n')
                    n['substantive'] += 1
                continue
            if sents:
                out.write(json.dumps({'task': 'substantive', 'key': key, 'offset': 0, 'question': q['question'],
                                      'sentences': sents[:FIRST_N]}, ensure_ascii=False) + '\n')
                n['substantive'] += 1
            if s['condition'] in CATT and route_of(cell, s) == 'hard' and s['status'] == 'COMPLETE':
                if q['source'] not in S.OPEN_SOURCES:
                    x, ok = S.score_chunks(q['source'], q['answer'], chunks, q['question'])
                    if ok is not True:
                        n['g2_excluded_closed_wrong'] += 1
                        continue
                order, groups = [], {}
                for u in units:
                    if u['mode'] in ('hard_first', 'hard_mid'):
                        if u['seq'] not in groups:
                            order.append(u['seq'])
                            groups[u['seq']] = []
                        groups[u['seq']].append(u['spoken'])
                if not order:
                    continue
                final = ' '.join(u['spoken'] for u in units if u['mode'] == 'hard_final')
                out.write(json.dumps({'task': 'g2', 'key': key, 'question': q['question'], 'reference': reference(q),
                                      'final': final, 'updates': [' '.join(groups[k]) for k in order],
                                      'update_seqs': order}, ensure_ascii=False) + '\n')
                n['g2'] += 1
    print(json.dumps(n))


if __name__ == '__main__':
    main()
