"""EXP-20261007-001: build the main set and audit its overlap with every development / calibration source.

Main set = EXP-20260930-003's frozen 220-item fixture (AIME 2026 30, GPQA Diamond 90, VoiceBench MMSU 50,
VoiceBench AlpacaEval 50) + EXP-20261005-001's 108 confirmation GPQA Diamond items (same formatting and
option orders) + clean open-ended items from VoiceBench 'alpacaeval_full' (636 prompts, text only).

An alpacaeval_full prompt is clean iff it is neither an exact normalized match nor a near duplicate
(word-trigram Jaccard >= 0.5, both texts >= 10 words; EXP-20261005-001's rule) of any prompt in:
the 220-item test fixture, the router-calibration fixture, the development pool (Dev-A/Dev-B included),
the router reference bank questions, the older scripts/data fixtures; nor of an earlier-kept extra.
Model-free. Usage: python build_main_fixture.py PKG OUT_DIR
"""
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

PKG = Path(sys.argv[1])
OUT = Path(sys.argv[2])
VB = Path('/data/userhome/xi/.cache/huggingface/hub/datasets--hlt-lab--voicebench/snapshots/'
          'b02edcef1330480be3a11bd6f7434ac32f05ad08')
TEST = PKG / 'v2/exp003/snapshot/scripts/data/router_sweep_questions_paper_main.jsonl'
TEST_SHA = '3a1cdac1162a9b236b62222b453564955305be1f7f02cbba8427defd506f5276'
CONFIRM = PKG / 'v2/fixtures/confirm_gpqa108.jsonl'
CONFIRM_SHA = 'c96aa0a588de5621ceb0d3c8ba2707a3dee4fbe6f8ebbf4eb8ca54bf18175fb1'
OTHERS = {'calibration': PKG / 'v2/exp003/snapshot/scripts/data/router_sweep_questions_paper_train.jsonl',
          'dev_pool': PKG / 'v2/fixtures/dev_pool.jsonl',
          'router_bank': PKG / 'inputs/router_train_questions.jsonl'}
OTHERS.update({f'scripts_data/{p.name}': p for p in sorted((PKG / 'inputs/scripts_data').glob('*.jsonl'))})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    with open(p, encoding='utf-8') as h:
        return [json.loads(l) for l in h if l.strip()]


def norm(s):
    s = unicodedata.normalize('NFKC', s or '').lower()
    return ' '.join(re.sub(r'[^0-9a-z]+', ' ', s).split())


def tri(s):
    w = s.split()
    return set(tuple(w[i:i + 3]) for i in range(len(w) - 2)), len(w)


def jac(a, b):
    (ta, na), (tb, nb) = a, b
    if na < 10 or nb < 10 or not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def text_of(r):
    return r.get('raw_problem') or r.get('question') or r.get('prompt') or ''


def main():
    assert sha(TEST) == TEST_SHA and sha(CONFIRM) == CONFIRM_SHA
    test, confirm = read_jsonl(TEST), read_jsonl(CONFIRM)
    refs = []  # (source name, id, normalized text, trigrams)
    for r in test:
        refs.append(('test', r['id'], norm(text_of(r))))
    for name, p in OTHERS.items():
        for r in read_jsonl(p):
            refs.append((name, r.get('id', ''), norm(text_of(r))))
    full = pq.read_table(VB / 'alpacaeval_full/test-00000-of-00001.parquet', columns=['prompt']).to_pylist()
    sub199 = pq.read_table(VB / 'alpacaeval/test-00000-of-00001.parquet', columns=['prompt']).to_pylist()
    sub_norm = {norm(r['prompt']) for r in sub199}
    ref_tri = [(n, i, t, tri(t)) for n, i, t in refs]
    exact = {}
    for n, i, t, _ in ref_tri:
        exact.setdefault(t, []).append((n, i))
    kept, audit = [], []
    kept_tri = []
    for idx, row in enumerate(full):
        t = norm(row['prompt'])
        tt = tri(t)
        reason = None
        if t in exact:
            reason = {'exact': exact[t][:3]}
        else:
            best = max(((jac(tt, rt), n, i) for n, i, _, rt in ref_tri), default=(0.0, '', ''))
            if best[0] >= 0.5:
                reason = {'near': [round(best[0], 3), best[1], best[2]]}
            else:
                bk = max(((jac(tt, kt), kid) for kid, kt in kept_tri), default=(0.0, ''))
                if bk[0] >= 0.5:
                    reason = {'near_kept_extra': [round(bk[0], 3), bk[1]]}
        rid = f'voicebench_alpacaeval_full/{idx:03d}'
        audit.append({'id': rid, 'in_199_subset': t in sub_norm, 'excluded': reason})
        if reason is None:
            kept.append({'id': rid, 'source': 'voicebench_alpacaeval_full', 'subject': 'general',
                         'difficulty_hint': 'open_ended', 'question': row['prompt'], 'answer': row['prompt'],
                         'raw_problem': row['prompt']})
            kept_tri.append((rid, tt))
    main_rows = test + confirm + kept
    ids = [r['id'] for r in main_rows]
    assert len(ids) == len(set(ids))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'main.jsonl', 'w', encoding='utf-8') as h:
        for r in main_rows:
            h.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(OUT / 'open_extras.jsonl', 'w', encoding='utf-8') as h:
        for r in kept:
            h.write(json.dumps(r, ensure_ascii=False) + '\n')
    from collections import Counter
    summary = {'alpacaeval_full_rows': len(full), 'alpacaeval_199_rows': len(sub199),
               'full_rows_in_199_subset': sum(a['in_199_subset'] for a in audit),
               'excluded_exact': sum(1 for a in audit if a['excluded'] and 'exact' in a['excluded']),
               'excluded_near': sum(1 for a in audit if a['excluded'] and 'near' in a['excluded']),
               'excluded_near_kept_extra': sum(1 for a in audit if a['excluded'] and 'near_kept_extra' in a['excluded']),
               'kept_extras': len(kept), 'kept_extras_in_199_subset': sum(1 for a in audit if a['excluded'] is None and a['in_199_subset']),
               'main_counts': dict(Counter(r['source'] for r in main_rows)), 'main_n': len(main_rows),
               'main_sha256': sha(OUT / 'main.jsonl'), 'reference_sources': {k: str(v) for k, v in OTHERS.items()},
               'test_sha256': TEST_SHA, 'confirm_sha256': CONFIRM_SHA}
    json.dump({'summary': summary, 'rows': audit}, open(OUT / 'open_extras_audit.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))


main()
