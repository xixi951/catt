"""EXP-20261015-001: judge units for the listener-experience (timeline) rating. Model-free; runs on CPU.

Reads EXP-20261009-001's raw formal run (read-only) and its frozen helpers (pkg/code/v5_common.py, which imports
pkg/v3/code/scorer_v4.py). Writes (BRIEF sections 2-4):
  units_main.jsonl    one unit per (item, condition) for the 738-item reduced set and the conditions
                      catt_v4_route, thinking, non_thinking (2,214 units);
  units_probeA.jsonl  validity probe A (initial wait): thinking-baseline replies of 60 items, first sentence moved to
                      D in {0.5, 3, 10, 30, 60} s, later sentences keep their recorded offsets (300 units);
  units_probeB.jsonl  validity probe B (silence after the first sentence): CATT router hard turns of 40 items whose
                      first sentence is the acknowledgment, with +0 (recorded), +10 or +30 s inserted after it (120 units);
  units_retest.jsonl  a seeded 10% sample of units_main for test-retest;
  keymap.jsonl        key -> question id, condition, item kind, route and timeline summary (never shown to judges).
A unit holds only what the judge sees (question, reference, rendered timeline) plus its key and the raw sentence
list for audit; the condition is kept out of the unit.

Usage: python build_units.py --run RUN_DIR --pkg PKG_DIR --exclude EXCLUDED_IDS.txt --out OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CONDITIONS = ('catt_v4_route', 'thinking', 'non_thinking')
SEED = 20261009
PROBE_A_LEVELS = (0.5, 3.0, 10.0, 30.0, 60.0)
PROBE_B_LEVELS = (0.0, 10.0, 30.0)
PROBE_A_N = {'closed': 30, 'open': 30}
PROBE_B_N = {'closed': 20, 'open': 20}
RETEST_FRACTION = 0.10
SILENCE_MIN = 0.5        # an inter-sentence gap of at least this many seconds gets its own silence line
_OPT = re.compile(r'^\s*([A-D])[.)]\s+(.+?)\s*$', re.M)
OPEN_REFERENCE = '(none: open-ended request)'


def fmt(t: float) -> str:
    return f'{t:.1f}'


def render(sents: List[Tuple[float, float, str]]) -> str:
    """Timeline text exactly as heard: every played sentence with its start and end time (seconds after the user
    finished speaking), an explicit silence line before the first sentence and for every later gap >= SILENCE_MIN,
    and the end of the reply. Times are rounded to 0.1 s; nothing about the producing system is shown."""
    if not sents:
        return '(no speech: the user heard nothing)'
    lines: List[str] = []
    t = 0.0
    for i, (start, end, text) in enumerate(sents):
        gap = start - t
        if (i == 0 and round(start, 1) > 0.0) or (i > 0 and gap >= SILENCE_MIN):
            lines.append(f'{fmt(t)} - {fmt(start)} s   (silence, {fmt(gap)} s)')
        lines.append(f'{fmt(start)} - {fmt(end)} s   "{text}"')
        t = max(t, end)
    lines.append(f'{fmt(t)} s   (end of reply)')
    return '\n'.join(lines)


def reference(S, q: dict) -> str:
    """Same reference string as EXP-20261009-001's make_units_v5 (letter plus option text; AIME integer)."""
    if q['source'] in S.MCQ_SOURCES:
        opts = dict(_OPT.findall(q['question']))
        return f"{q['answer']}" + (f" ({opts[q['answer']]})" if q['answer'] in opts else '')
    if q['source'] in S.AIME_SOURCES:
        return str(q['answer'])
    return OPEN_REFERENCE


def unit(key: str, q: dict, ref: str, sents: List[Tuple[float, float, str]]) -> dict:
    return {'task': 'listen', 'key': key, 'question': q['question'], 'reference': ref, 'timeline': render(sents),
            'sentences': [[round(a, 3), round(b, 3), s] for a, b, s in sents]}


def route_of(C, cell: Path, s: dict) -> str:
    if s['condition'] != 'catt_v4_route':
        return 'single_call'
    ws = cell / s['ws_log'] if s.get('ws_log') else None
    if ws and ws.exists():
        for e in C.read_jsonl(ws):
            if e.get('kind') == 'route_decision':
                return e['data'].get('route')
    sp = s.get('speculative') or {}
    return sp.get('route') or 'unknown'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pkg', required=True)
    ap.add_argument('--exclude', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg) / 'code'))
    import v5_common as C  # noqa: E402  (frozen v4 helpers, read-only)
    S = C.S
    fx = {r['id']: r for r in C.read_jsonl(Path(args.pkg) / 'v3' / 'fixtures' / 'main.jsonl')}
    excluded = {l.strip() for l in open(args.exclude, encoding='utf-8') if l.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    turns: Dict[str, Dict[str, dict]] = {}
    for cell, s in C.iter_run(Path(args.run)):
        if s['condition'] not in CONDITIONS or s['question_id'] in excluded:
            continue
        q = fx[s['question_id']]
        units = C.sentence_units(C.played(s))
        sents = [(u['playback_start'], u['playback_end'], u['spoken']) for u in units]
        kind = 'open' if q['source'] in S.OPEN_SOURCES else 'closed'
        turns.setdefault(s['question_id'], {})[s['condition']] = {
            'key': f"formal_v4/{s['set']}/{s['attempt']:04d}", 'question_id': s['question_id'], 'condition': s['condition'],
            'kind': kind, 'source': q['source'], 'status': s['status'], 'route': route_of(C, cell, s),
            'first_mode': units[0]['mode'] if units else None, 'sents': sents,
            'first_audio': sents[0][0] if sents else None, 'reply_end': max((b for _, b, _ in sents), default=None),
            'n_sentences': len(sents), 'silence_total': sum(max(0.0, b[0] - a[1]) for a, b in zip(sents, sents[1:]))}
    qids = sorted(turns)
    missing = [(qid, c) for qid in qids for c in CONDITIONS if c not in turns[qid]]
    kinds = {k: sum(1 for qid in qids if turns[qid][CONDITIONS[1]]['kind'] == k) for k in ('closed', 'open')}
    assert len(qids) == 738 and not missing and kinds == {'closed': 214, 'open': 524}, (len(qids), missing[:5], kinds)

    main_units, keymap = [], []
    for qid in qids:
        q = fx[qid]
        ref = reference(S, q)
        for c in CONDITIONS:
            t = turns[qid][c]
            main_units.append(unit(t['key'], q, ref, t['sents']))
            keymap.append({k: v for k, v in t.items() if k != 'sents'} | {'set': 'main', 'reference': ref})

    rng = random.Random(SEED)
    probe_a, probe_b = [], []
    for kind, n in PROBE_A_N.items():
        pool = [qid for qid in qids if turns[qid]['thinking']['kind'] == kind and turns[qid]['thinking']['sents']]
        for qid in sorted(rng.sample(pool, n)):
            q, t = fx[qid], turns[qid]['thinking']
            o = t['sents'][0][0]
            for d in PROBE_A_LEVELS:
                sents = [(a - o + d, b - o + d, s) for a, b, s in t['sents']]
                key = f'probeA/{qid}/D{d:g}'
                probe_a.append(unit(key, q, reference(S, q), sents))
                keymap.append({'key': key, 'set': 'probeA', 'question_id': qid, 'kind': kind, 'level': d,
                               'source_key': t['key'], 'source_condition': 'thinking'})
    for kind, n in PROBE_B_N.items():
        pool = [qid for qid in qids if turns[qid]['catt_v4_route']['kind'] == kind
                and turns[qid]['catt_v4_route']['route'] == 'hard' and turns[qid]['catt_v4_route']['first_mode'] == 'hard_first'
                and len(turns[qid]['catt_v4_route']['sents']) >= 2]
        for qid in sorted(rng.sample(pool, n)):
            q, t = fx[qid], turns[qid]['catt_v4_route']
            for x in PROBE_B_LEVELS:
                sents = [t['sents'][0]] + [(a + x, b + x, s) for a, b, s in t['sents'][1:]]
                key = f'probeB/{qid}/S{x:g}'
                probe_b.append(unit(key, q, reference(S, q), sents))
                keymap.append({'key': key, 'set': 'probeB', 'question_id': qid, 'kind': kind, 'level': x,
                               'source_key': t['key'], 'source_condition': 'catt_v4_route'})

    rr = random.Random(f'{SEED}-retest')
    retest = sorted(rr.sample(sorted(main_units, key=lambda u: u['key']), round(RETEST_FRACTION * len(main_units))),
                    key=lambda u: u['key'])

    def write(name: str, rows: List[dict], shuffle_seed: Optional[str]) -> None:
        rows = list(rows)
        if shuffle_seed is not None:   # interleave conditions and levels in submission order
            random.Random(shuffle_seed).shuffle(rows)
        with open(out / name, 'w', encoding='utf-8') as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + '\n')

    write('units_main.jsonl', main_units, f'{SEED}-order-main')
    write('units_probeA.jsonl', probe_a, f'{SEED}-order-probeA')
    write('units_probeB.jsonl', probe_b, f'{SEED}-order-probeB')
    write('units_retest.jsonl', retest, f'{SEED}-order-retest')
    write('keymap.jsonl', keymap, None)
    chars = {name: sum(len(u['question']) + len(u['timeline']) + len(u['reference']) for u in rows)
             for name, rows in (('main', main_units), ('probeA', probe_a), ('probeB', probe_b), ('retest', retest))}
    print(json.dumps({'items': len(qids), 'kinds': kinds, 'main': len(main_units), 'probeA': len(probe_a),
                      'probeB': len(probe_b), 'retest': len(retest), 'unit_chars': chars,
                      'max_timeline_chars': max(len(u['timeline']) for u in main_units)}))


if __name__ == '__main__':
    main()
