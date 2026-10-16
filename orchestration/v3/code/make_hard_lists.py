"""EXP-20261007-001: hard-turn lists at tau* from the router pre-pass (BRIEF 1). Also checks router
identity against EXP-20261005-001's pre-pass for the 328 items both EXPs share.
Usage: python make_hard_lists.py PREPASS.jsonl V2_HARD_LISTS.json V2_PREPASS.jsonl OUT.json"""
import json
import sys
from collections import Counter

TAU = 0.5444878691642917
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v2 = json.load(open(sys.argv[2]))['hard_lists']
v2_hard = set().union(*v2.values())
v2_scores = {}
try:
    v2_scores = {r['question_id']: r['strong_win_prob'] for r in (json.loads(l) for l in open(sys.argv[3]) if l.strip())}
except FileNotFoundError:
    pass
main = [r for r in rows if r['fixture'] == 'main.jsonl']
dev = [r for r in rows if r['fixture'] == 'dev_pool.jsonl']
hard_main = sorted(r['question_id'] for r in main if r['strong_win_prob'] >= TAU)
hard_dev = sorted(r['question_id'] for r in dev if r['strong_win_prob'] >= TAU)
shared = [r for r in main if r['question_id'] in v2_scores]
diffs = [abs(r['strong_win_prob'] - v2_scores[r['question_id']]) for r in shared]
v2_ids_main = [r['question_id'] for r in main if not r['source'].startswith('voicebench_alpacaeval_full')]
route_agree = sum(((r['strong_win_prob'] >= TAU) == (r['question_id'] in v2_hard)) for r in main
                  if r['question_id'] in set(v2_ids_main))
out = {'tau': TAU, 'n_main': len(main), 'n_dev': len(dev),
       'hard_main_by_source': dict(Counter(r['source'] for r in main if r['strong_win_prob'] >= TAU)),
       'easy_main_by_source': dict(Counter(r['source'] for r in main if r['strong_win_prob'] < TAU)),
       'v2_shared_items': len(v2_ids_main), 'v2_route_agreement': route_agree,
       'v2_score_max_abs_diff': max(diffs) if diffs else None, 'v2_scores_compared': len(diffs),
       'closest_to_tau_main': sorted(abs(r['strong_win_prob'] - TAU) for r in main)[:5],
       'hard_lists': {'MAIN': hard_main, 'DEV': hard_dev}}
json.dump(out, open(sys.argv[4], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'hard_lists'}, indent=1))
