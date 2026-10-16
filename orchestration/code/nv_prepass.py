"""EXP-20261009-001 (change 7): NVIDIA complexity scores on CPU and the test-free threshold tau_nv.

tau_nv uses EXP-20261002-001's rule T1 (code/analysis.py tstar, copied verbatim below) on the same primary
calibration set as tau* (the 318 historical calibration items left after the overlap exclusion, repaired
gains g_repaired from EXP-20261002-001 inputs/calibration.csv). No main-set outcome is used. Then the frozen
tau_nv is applied to the main set and the development pool (inputs only) to fix the hard list of the
nvroute configuration. Every score is computed twice; the run-to-run max abs difference is recorded.

Usage (newsrv, CPU): CUDA_VISIBLE_DEVICES="" python nv_prepass.py --cal-questions Q.jsonl --cal-csv calibration.csv
                     --sets main=... devpool=... --ckpt CKPT --deberta-cfg CFG --out inputs/nv_prepass.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

assert os.environ.get('CUDA_VISIBLE_DEVICES', 'x') == '', 'must run with CUDA_VISIBLE_DEVICES=""'
sys.path.insert(0, str(Path(__file__).resolve().parent))


def tstar(scores, gains, lam=0.0) -> dict:
    """EXP-20261002-001 rule T1 (verbatim). Items with score >= tau go hard; the cut between adjacent distinct
    calibration scores maximizing the summed gain of hard-routed items; ties to the fewest hard routes;
    tau is the midpoint of the two adjacent distinct scores."""
    s = np.asarray(scores, dtype=float)
    g = np.asarray(gains, dtype=float) - lam
    order = np.argsort(s, kind="mergesort")
    ss, gs = s[order], g[order]
    n = len(ss)
    suffix = np.concatenate([np.cumsum(gs[::-1])[::-1], [0.0]])
    uniq, starts = np.unique(ss, return_index=True)
    cuts = list(starts.tolist()) + [n]
    vals = np.array([suffix[c] for c in cuts])
    best = vals.max()
    chosen = max(c for c, v in zip(cuts, vals) if v >= best - 1e-9)
    if chosen == 0:
        tau = -math.inf
    elif chosen == n:
        tau = math.inf
    else:
        tau = 0.5 * (ss[chosen - 1] + ss[chosen])
    return {"tau": tau, "n_hard": int(n - chosen), "n": int(n),
            "gain_sum": float(best + lam * (n - chosen)), "objective": float(best)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cal-questions', required=True)
    ap.add_argument('--cal-csv', required=True)
    ap.add_argument('--sets', nargs='+', required=True, help='name=path.jsonl')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--deberta-cfg', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from catt_v4 import NvScorer
    t0 = time.time()
    scorer = NvScorer(Path(args.ckpt), Path(args.deberta_cfg), threads=args.threads)
    qtext = {}
    for line in open(args.cal_questions, encoding='utf-8'):
        r = json.loads(line)
        qtext[r['id']] = r['question']
    cal = list(csv.DictReader(open(args.cal_csv)))
    cal = [r for r in cal if r['excluded_overlap_test'] == '0']
    out = {'checkpoint': 'nvidia/prompt-task-and-complexity-classifier@fea11215', 'rule': 'EXP-20261002-001 T1',
           'cal_csv_sha256': hashlib.sha256(Path(args.cal_csv).read_bytes()).hexdigest(),
           'cal_questions_sha256': hashlib.sha256(Path(args.cal_questions).read_bytes()).hexdigest(),
           'n_cal': len(cal)}
    s1 = [scorer.score(qtext[r['cal_id']]) for r in cal]
    s2 = [scorer.score(qtext[r['cal_id']]) for r in cal]
    gains = [float(r['g_repaired']) for r in cal]
    t = tstar(s1, gains)
    out.update(tau_nv=t['tau'], cal_n_hard=t['n_hard'], cal_gain_sum=t['gain_sum'],
               cal_rerun_maxabs=float(np.max(np.abs(np.array(s1) - np.array(s2)))),
               cal_scores={r['cal_id']: s for r, s in zip(cal, s1)})
    tau = t['tau']
    sets = {}
    for spec in args.sets:
        name, path = spec.split('=', 1)
        rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
        a = [scorer.score(r['question']) for r in rows]
        b = [scorer.score(r['question']) for r in rows]
        sets[name] = {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      'rerun_maxabs': float(np.max(np.abs(np.array(a) - np.array(b)))),
                      'scores': {r['id']: s for r, s in zip(rows, a)},
                      'hard': sorted(r['id'] for r, s in zip(rows, a) if s >= tau),
                      'near_tau_0.0005': sum(1 for s in a if abs(s - tau) < 0.0005)}
    out['sets'] = sets
    out['seconds'] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'tau_nv': tau, 'cal_n_hard': t['n_hard'], 'n_cal': len(cal), 'cal_rerun_maxabs': out['cal_rerun_maxabs'],
                      **{k: {'n': len(v['scores']), 'hard': len(v['hard']), 'rerun_maxabs': v['rerun_maxabs'],
                             'near_tau': v['near_tau_0.0005']} for k, v in sets.items()},
                      'seconds': round(out['seconds'], 1)}, indent=1))


if __name__ == '__main__':
    main()
