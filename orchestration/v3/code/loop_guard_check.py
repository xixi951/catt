"""EXP-20261007-001: offline replay of the loop guard (BRIEF 3.1 change 3; thresholds fixed in BRIEF v0) on
development reasoning streams (deltas.jsonl 'r' channel of thinking calls). Reports, per stream, whether the
guard would have fired and at which reasoning token, and how the stream actually ended.
Usage: python loop_guard_check.py OUT.json RUN_DIR [RUN_DIR ...]   (run dirs containing attempt-*/)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catt_v3 import (LOOP_EVERY_TOKENS, LOOP_MIN_COUNT, LOOP_PROBE_CHARS, LOOP_START_TOKENS,  # noqa: E402
                     LOOP_WINDOW_CHARS, repeated_tail)

rows = []
for run in sys.argv[2:]:
    for cell in sorted(Path(run).glob('attempt-*')):
        d = cell / 'deltas.jsonl'
        sp = cell / 'summary.json'
        if not d.exists() or not sp.exists():
            continue
        s = json.loads(sp.read_text())
        streams = {}
        for line in open(d, encoding='utf-8'):
            r = json.loads(line)
            if r.get('k') == 'r':
                streams.setdefault(r['c'], []).append((r['x'], r.get('n')))
        for call, parts in streams.items():
            info = (s.get('calls') or {}).get(call, {})
            text, nxt, fired_at = '', LOOP_START_TOKENS, None
            for x, n in parts:
                text += x
                if n is not None and n >= nxt:
                    nxt = n + LOOP_EVERY_TOKENS
                    if fired_at is None and repeated_tail(text, LOOP_PROBE_CHARS, LOOP_WINDOW_CHARS, LOOP_MIN_COUNT):
                        fired_at = n
            rows.append({'run': run, 'attempt': cell.name, 'condition': s.get('condition'), 'question_id': s.get('question_id'),
                         'call': call, 'reasoning_tokens': parts[-1][1], 'chars': len(text),
                         'budget_forced': bool(info.get('budget_forced')), 'finish': info.get('finish_reason'),
                         'guard_fired_at': fired_at})
summary = {'streams': len(rows), 'over_2000_tokens': sum(1 for r in rows if (r['reasoning_tokens'] or 0) >= 2000),
           'guard_fired': sum(1 for r in rows if r['guard_fired_at'] is not None),
           'guard_fired_on_naturally_ended': sum(1 for r in rows if r['guard_fired_at'] is not None and not r['budget_forced']),
           'budget_forced_streams': sum(1 for r in rows if r['budget_forced']),
           'budget_forced_and_fired': sum(1 for r in rows if r['budget_forced'] and r['guard_fired_at'] is not None),
           'max_tokens': max((r['reasoning_tokens'] or 0) for r in rows) if rows else None}
json.dump({'summary': summary, 'rows': rows}, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    if r['guard_fired_at'] is not None or r['budget_forced']:
        print(json.dumps(r))
