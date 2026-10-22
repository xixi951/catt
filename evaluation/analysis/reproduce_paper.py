"""Recompute the main reported values (see README) from the released per-item data (738 evaluated items).

Inputs (all under --data):
  v4/turns.csv                                  one row per item and condition (5 conditions x 738 items)
  v4/luna_labels.jsonl.gz                       gpt-5.6-luna labels (first substantive sentence, intermediate errors)
  v4/luna_labels_retest.jsonl.gz                the same labels for the re-judged sample (test-retest agreement)
  v4/luna_labels_wrong_final.jsonl.gz           gpt-5.6-luna intermediate-error labels of the 65 closed-form turns of
                                                the router configuration routed to reasoning whose final answer is not
                                                strictly correct (not counted by g2_included)
  v4/luna_labels_wrong_final_retest.jsonl.gz    the same labels for their re-judged sample (6 turns)
  listening_rating/exp20261015001_scores.csv    listening rating: CATT router and both baselines, probes, retest
  listening_rating/exp20261015002_scores.csv    listening rating: NVIDIA-router and all-thinking configurations,
                                                ReVerT, second retest (plus the first study's main scores as main001)
  revert/turns.jsonl                            ReVerT per-turn metrics (139 items)
  transfer/turns.csv                            transfer study per-turn metrics (3 hosted models, 139 items)

Statistics (identical to the study's analysis code): pairs are formed per item and sorted by question id; the point
estimate is the median (times) or mean (accuracy, ratings) of the paired differences; the 95% interval is the
2.5/97.5 percentile of 10,000 bootstrap resamples drawn with numpy.random.default_rng(20261009); a missing value
drops the pair. Rates use the Wilson interval. Timing values of 1200 s mean "not reached within the turn watchdog".

Substantive-content timing and intermediate-speech errors use the gpt-5.6-luna labels; their test-retest agreement
compares the re-judged labels with the main ones. Intermediate-speech errors are given for the turns counted by
g2_included (closed-form turns only when the final answer is strictly correct) and, for the router configuration,
over all turns routed to reasoning. The listening rating uses both judges (gpt-5.6-luna,
gemini-3.1-flash-lite) and their per-turn mean.

Usage:
  python reproduce_paper.py --data ../../data --out results            # writes results/paper_numbers.json
  python reproduce_paper.py --data ../../data --out results --check expected/paper_numbers_expected.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SEED, DRAWS, WATCHDOG = 20261009, 10000, 1200.0
CATT = ('catt_v4_route', 'catt_v4_nvroute', 'catt_v4_allhard')
BASES = ('thinking', 'non_thinking')
JUDGES = {'luna': 'score_luna', 'flash_lite': 'score_flash_lite'}


# ----------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------

def bootstrap(values, stat=np.median, seed: int = SEED) -> dict:
    d = np.asarray(values, dtype=float)
    if len(d) == 0:
        return {'n': 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(DRAWS, len(d)))
    draws = stat(d[idx], axis=1)
    return {'n': int(len(d)), 'estimate': float(stat(d)), 'ci_low': float(np.percentile(draws, 2.5)),
            'ci_high': float(np.percentile(draws, 97.5))}


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def value(x: str):
    if x in ('', 'None', None):
        return None
    if x in ('True', 'False'):
        return x == 'True'
    try:
        return float(x)
    except ValueError:
        return x


# ----------------------------------------------------------------------------------------------------------------
# v4 main experiment (738 items)
# ----------------------------------------------------------------------------------------------------------------

def load_turns(path: Path) -> Dict[str, Dict[str, dict]]:
    by: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            by[r['question_id']][r['condition']] = {k: value(v) for k, v in r.items()}
    return by


def paired(by, cond: str, base: str, metric: str, subset: Optional[Callable] = None) -> List[float]:
    out = []
    for qid in sorted(by):
        d = by[qid]
        if cond not in d or base not in d:
            continue
        if subset is not None and not subset(d[cond]):
            continue
        a, b = d[cond].get(metric), d[base].get(metric)
        if a is None or b is None:
            continue
        out.append(a - b)
    return out


def hard_of(cond: str) -> Callable:
    if cond == 'catt_v4_allhard':
        return lambda r: True
    c = cond.replace('catt_v4_', '')
    return lambda r: bool(r[f'hard_item|{c}'])


def v4_numbers(by) -> dict:
    res: dict = {}
    for cond in CATT:
        hs = hard_of(cond)
        subsets = {'all': None, 'closed': lambda r: r['closed'], 'open': lambda r: r['open'], 'hard': hs,
                   'closed_hard': lambda r, h=hs: r['closed'] and h(r), 'open_hard': lambda r, h=hs: r['open'] and h(r)}
        if cond != 'catt_v4_allhard':
            subsets['easy'] = lambda r, h=hs: not h(r)
            subsets['closed_easy'] = lambda r, h=hs: r['closed'] and not h(r)
            subsets['open_easy'] = lambda r, h=hs: r['open'] and not h(r)
        c: dict = {}
        for metric in ('t_first_audio', 't_first_substantive', 't_first_candidate', 't_final_answer', 'playback_completion',
                       'gap_total', 'wall_seconds', 'llm_calls', 'tokens_prompt', 'tokens_completion', 'tokens_reasoning',
                       'gpu_energy_j'):
            for base in BASES:
                for sname, sub in subsets.items():
                    x = bootstrap(paired(by, cond, base, metric, sub), np.median)
                    if x['n']:
                        c[f'{metric}|{cond}-{base}|{sname}'] = x
        for base in BASES:
            c[f'correct_strict|{cond}-{base}|closed(mean)'] = bootstrap(
                paired(by, cond, base, 'correct_strict', lambda r: r['closed']), np.mean)
        # continuity: total silence after first audio on turns routed to reasoning
        # in run order (attempt number), as in the study's analysis
        hard_turns = sorted((d[cond] for d in by.values() if cond in d and d[cond]['route'] == 'hard'
                             and d[cond].get('gap_total') is not None and hs(d[cond])), key=lambda r: r['attempt'])
        gaps = [r['gap_total'] for r in hard_turns]
        g = np.asarray(gaps, dtype=float)
        x = bootstrap(g, np.median)
        c['continuity_hard_turns'] = {'n': x['n'], 'median': x['estimate'], 'median_ci_low': x['ci_low'],
                                      'median_ci_high': x['ci_high'], 'p90': float(np.percentile(g, 90)),
                                      'max': float(g.max()), 'share_zero': float(np.mean(g == 0)),
                                      'share_over_1s': float(np.mean(g > 1.0)), 'n_over_2s': int(np.sum(g > 2.0))}
        # intermediate-speech errors (luna), turns routed to reasoning; closed-form turns only when answered correctly
        turns = [d[cond] for d in by.values() if cond in d and d[cond]['route'] == 'hard' and d[cond].get('g2_included')]
        for scope, f in (('all', lambda r: True), ('closed_only', lambda r: r['closed']), ('open_only', lambda r: r['open'])):
            vals = [r['intermediate_error'] for r in turns if f(r) and r.get('intermediate_error') is not None]
            k, n = int(sum(vals)), len(vals)
            lo, hi = wilson(k, n)
            c[f'intermediate_error|{scope}'] = {'n_turns': n, 'n_error_turns': k, 'rate': (k / n) if n else None,
                                                'wilson_low': lo, 'wilson_high': hi}
        c['intermediate_error|excluded_closed_wrong'] = sum(
            1 for d in by.values() if cond in d and d[cond]['route'] == 'hard' and d[cond]['closed']
            and d[cond].get('correct_strict') == 0)
        # reasoning identity with the thinking baseline on turns routed to reasoning
        same = tot = 0
        for d in by.values():
            if cond in d and 'thinking' in d and d[cond]['route'] == 'hard' and d[cond].get('reasoning_sha256'):
                tot += 1
                same += d[cond]['reasoning_sha256'] == d['thinking'].get('reasoning_sha256')
        c['reasoning_identity'] = {'identical': same, 'total': tot}
        rs = [d[cond] for d in by.values() if cond in d]
        c['routing'] = {'hard_turns': sum(r['route'] == 'hard' for r in rs), 'easy_turns': sum(r['route'] == 'easy' for r in rs),
                        'closed_routed_hard': sum(r['route'] == 'hard' and r['closed'] for r in rs),
                        'open_routed_hard': sum(r['route'] == 'hard' and r['open'] for r in rs)}
        res[cond] = c
    # per-condition descriptives: accuracy by dataset, calls per turn, budget-forced turns
    desc: dict = {}
    conds = CATT + BASES
    for cond in conds:
        rs = [d[cond] for d in by.values() if cond in d]
        e = {'turns': len(rs), 'complete': sum(r['status'] == 'COMPLETE' for r in rs),
             'llm_calls_mean': float(np.mean([r['llm_calls'] for r in rs])),
             'budget_forced_turns': sum(1 for r in rs if (r.get('forced') or 0) > 0)}
        for src in sorted({r['source'] for r in rs}):
            sc = [r['correct_strict'] for r in rs if r['source'] == src and r.get('correct_strict') is not None]
            if sc:
                e[f'accuracy|{src}'] = {'n': len(sc), 'correct': int(sum(sc)), 'rate': float(np.mean(sc))}
        sc = [r['correct_strict'] for r in rs if r['closed'] and r.get('correct_strict') is not None]
        e['accuracy|closed'] = {'n': len(sc), 'correct': int(sum(sc)), 'rate': float(np.mean(sc))}
        desc[cond] = e
    res['descriptives'] = desc
    return res


# ----------------------------------------------------------------------------------------------------------------
# listening rating
# ----------------------------------------------------------------------------------------------------------------

def load_listen(path: Path) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return [{k: value(v) if k not in ('key', 'question_id', 'set', 'condition') else v for k, v in r.items()}
                for r in csv.DictReader(f)]


def with_mean(r: dict) -> dict:
    a, b = r.get('score_luna'), r.get('score_flash_lite')
    r = dict(r)
    r['score_mean2'] = (a + b) / 2 if a is not None and b is not None else None
    return r


SCORES = ('score_luna', 'score_flash_lite', 'score_mean2')


def paired_scores(by_q, a: str, b: str, qids, s: str) -> List[float]:
    out = []
    for q in sorted(qids):
        x, y = by_q[q].get(a, {}).get(s), by_q[q].get(b, {}).get(s)
        if x is not None and y is not None:
            out.append(x - y)
    return out


def listening_numbers(rows1: List[dict], rows2: List[dict], revert_conditions) -> dict:
    res: dict = {}
    main = [with_mean(r) for r in rows1 if r['set'] == 'main']
    cfg = [with_mean(r) for r in rows2 if r['set'] in ('cfg', 'revert')]
    by_q: Dict[str, Dict[str, dict]] = defaultdict(dict)
    kind: Dict[str, str] = {}
    route: Dict[str, str] = {}
    for r in main + cfg:
        by_q[r['question_id']][r['condition']] = r
        kind[r['question_id']] = r['kind']
        if r['condition'] == 'catt_v4_route':
            route[r['question_id']] = r['route']
    qsets = {'closed': [q for q in kind if kind[q] == 'closed'], 'open': [q for q in kind if kind[q] == 'open'],
             'all': list(kind)}
    for s in SCORES:
        for k in ('closed', 'open', 'all'):
            for a in CATT + ('thinking',):
                for b in BASES:
                    if a == b:
                        continue
                    res[f'{s}|{k}|{a}-{b}'] = bootstrap(paired_scores(by_q, a, b, qsets[k], s), np.mean)
            for rt in ('hard', 'easy'):
                qs = [q for q in qsets[k] if route.get(q) == rt]
                for b in BASES:
                    res[f'{s}|{k}|route={rt}|catt_v4_route-{b}'] = bootstrap(paired_scores(by_q, 'catt_v4_route', b, qs, s), np.mean)
            for arm in revert_conditions:
                res[f'{s}|{k}|{arm}-catt_v4_route'] = bootstrap(paired_scores(by_q, arm, 'catt_v4_route', qsets[k], s), np.mean)
                for b in BASES:
                    res[f'{s}|{k}|{arm}-{b}'] = bootstrap(paired_scores(by_q, arm, b, qsets[k], s), np.mean)
            for c in CATT + BASES + tuple(revert_conditions):
                vals = [by_q[q][c][s] for q in qsets[k] if c in by_q[q] and by_q[q][c].get(s) is not None]
                if vals:
                    res[f'mean|{s}|{k}|{c}'] = {'n': len(vals), 'mean': float(np.mean(vals))}
    # validity probes: score change when the first sentence is delayed (A) or a silence is inserted (B)
    for probe, lo in (('probeA', 0.5), ('probeB', 0.0)):
        rows = [with_mean(r) for r in rows1 if r['set'] == probe]
        levels = sorted({r['level'] for r in rows})
        for s in SCORES:
            by: Dict[str, Dict[float, float]] = defaultdict(dict)
            for r in rows:
                if r.get(s) is not None:
                    by[r['question_id']][r['level']] = r[s]
            complete = sorted(q for q, d in by.items() if len(d) == len(levels))
            for lv in levels:
                res[f'{probe}|{s}|mean|level={lv:g}'] = float(np.mean([by[q][lv] for q in complete]))
                if lv != lo:
                    res[f'{probe}|{s}|{lv:g}-{lo:g}'] = bootstrap([by[q][lv] - by[q][lo] for q in complete], np.mean)
    # test-retest (same timelines judged again) and inter-judge agreement on the main turns
    main_by_key = {r['key']: r for r in main}
    for name, rows in (('retest', [with_mean(r) for r in rows1 if r['set'] == 'retest']),
                       ('retest2', [with_mean(r) for r in rows2 if r['set'] == 'retest2'])):
        ref = main_by_key if name == 'retest' else {r['key']: with_mean(r) for r in rows2 if r['set'] in ('cfg', 'main001')}
        for s in SCORES:
            pairs = [(ref[r['key']][s], r[s]) for r in rows if r['key'] in ref and ref[r['key']].get(s) is not None
                     and r.get(s) is not None]
            if pairs:
                res[f'{name}|{s}'] = {'turns': len(pairs), 'exact_agreement': float(np.mean([a == b for a, b in pairs])),
                                      'mean_abs_diff': float(np.mean([abs(a - b) for a, b in pairs]))}
    both = [(r['score_luna'], r['score_flash_lite']) for r in main
            if r.get('score_luna') is not None and r.get('score_flash_lite') is not None]
    res['inter_judge|main'] = {'turns': len(both), 'mean_luna': float(np.mean([a for a, _ in both])),
                               'mean_flash_lite': float(np.mean([b for _, b in both])),
                               'exact_agreement': float(np.mean([a == b for a, b in both]))}
    return res


# ----------------------------------------------------------------------------------------------------------------
# test-retest agreement of the gpt-5.6-luna labels (re-judged sample of the 738 items)
# ----------------------------------------------------------------------------------------------------------------

ERROR_LABELS = ('WRONG', 'REVERSED')


def load_labels(path: Path) -> Dict[str, dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return {r['key']: r for r in (json.loads(line) for line in f if line.strip())}


def first_substantive(labels: List[str]) -> Optional[int]:
    return next((i for i, x in enumerate(labels) if x == 'Y'), None)


def luna_retest_numbers(main_path: Path, retest_path: Path) -> dict:
    """Agreement of the re-judged labels with the main labels, as in the study's analysis: the same first substantive
    sentence (position of the first Y; no Y at all also counts as a position) and the same intermediate-error flag (any
    update WRONG or REVERSED). A turn without a main label of that task is skipped."""
    main, retest = load_labels(main_path), load_labels(retest_path)
    res: dict = {}
    n = a = 0
    for k, r in sorted(retest.items()):
        m = main.get(k)
        if r['substantive'] is None or m is None or not m['substantive'] or None in m['substantive']:
            continue
        n += 1
        a += first_substantive(r['substantive']) == first_substantive(m['substantive'])
    res['first_substantive'] = {'turns': n, 'agreeing': a, 'agreement': (a / n) if n else None}
    n = a = 0
    for k, r in sorted(retest.items()):
        m = main.get(k)
        if r['intermediate'] is None or m is None or m['intermediate'] is None:
            continue
        n += 1
        a += any(x in ERROR_LABELS for x in r['intermediate']) == any(x in ERROR_LABELS for x in m['intermediate'])
    res['intermediate_error_flag'] = {'turns': n, 'agreeing': a, 'agreement': (a / n) if n else None}
    return res


# ----------------------------------------------------------------------------------------------------------------
# intermediate-speech errors over all turns routed to reasoning (router configuration)
# ----------------------------------------------------------------------------------------------------------------

def rate_wilson(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {'n_turns': n, 'n_error_turns': k, 'rate': (k / n) if n else None, 'wilson_low': lo, 'wilson_high': hi}


def intermediate_all_hard_numbers(by, wrong_final_labels: Path) -> dict:
    """Intermediate-speech errors of the router configuration over all 410 turns routed to reasoning, with the same
    judge, prompt and rule (an error turn has at least one update labelled WRONG or REVERSED). Turns counted by
    g2_included take intermediate_error from turns.csv; the closed-form turns whose final answer is not strictly
    correct, which g2_included leaves out, take their labels from luna_labels_wrong_final.jsonl.gz. Also the split of
    the closed-form turns by final answer (strictly correct or not) and intermediate error, each part as a share of
    the closed-form turns with its Wilson interval."""
    cond = 'catt_v4_route'
    new = load_labels(wrong_final_labels)
    hard = [d[cond] for d in by.values() if cond in d and d[cond]['route'] == 'hard']
    wrong_final = {r['key'] for r in hard if r['closed'] and r.get('correct_strict') != 1}
    turns = []      # (closed, correct final, intermediate error)
    for r in hard:
        if r.get('g2_included') and r.get('intermediate_error') is not None and r['key'] not in wrong_final:
            err = bool(r['intermediate_error'])
        elif r['key'] in wrong_final and r['key'] in new:
            err = any(x in ERROR_LABELS for x in new[r['key']]['intermediate'])
        else:   # e.g. a rerun's turns.csv: the released labels do not cover its turns
            print(f'intermediate errors over all turns routed to reasoning not computed: no label for {r["key"]}')
            return {}
        turns.append((bool(r['closed']), r['key'] not in wrong_final, err))
    res: dict = {}
    for scope, f in (('all', lambda t: True), ('closed_only', lambda t: t[0]),
                     ('wrong_final_closed', lambda t: t[0] and not t[1])):
        sel = [t for t in turns if f(t)]
        res[f'intermediate_error_all_hard|{scope}'] = rate_wilson(sum(t[2] for t in sel), len(sel))
    closed = [t for t in turns if t[0]]
    split: dict = {'n_turns': len(closed)}
    for name, ok, err in (('correct_final|no_error', True, False), ('correct_final|error', True, True),
                          ('wrong_final|error', False, True), ('wrong_final|no_error', False, False)):
        k = sum(1 for t in closed if t[1] == ok and t[2] == err)
        lo, hi = wilson(k, len(closed))
        split[name] = {'n_turns': k, 'share': k / len(closed), 'wilson_low': lo, 'wilson_high': hi}
    res['intermediate_error_all_hard|closed_split'] = split
    return res


# ----------------------------------------------------------------------------------------------------------------
# ReVerT comparison (139 items)
# ----------------------------------------------------------------------------------------------------------------

def revert_numbers(path: Path, by) -> dict:
    """ReVerT arms (released per-turn metrics) against CATT v4 router and the baselines (released v4 turns)."""
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    arms = sorted({r['condition'] for r in rows})
    rv: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        rv[r['question_id']][r['condition']] = r
    ids = sorted(rv)
    comp = {}
    for q in ids:
        for c in ('catt_v4_route', 'thinking', 'non_thinking'):
            v = by[q][c]
            comp.setdefault(q, {})[c] = {'t_first_audio': v['t_first_audio'], 't_first_sub': v['t_first_substantive'],
                                         't_final_answer': v['t_final_answer'], 'gap_total': v['gap_total'],
                                         'playback_completion': v['playback_completion'], 'correct': v['correct_strict'],
                                         'llm_calls': v['llm_calls']}
    subsets = {'all': lambda r: True, 'closed': lambda r: r['closed'], 'open': lambda r: r['open'],
               'hard': lambda r: r['hard'], 'hard_closed': lambda r: r['hard'] and r['closed'],
               'hard_open': lambda r: r['hard'] and r['open'], 'easy': lambda r: not r['hard']}
    metrics = {'t_first_audio': ('t_first_audio', np.median, None), 't_first_sub_luna': ('t_first_sub', np.median, None),
               't_final_answer': ('t_final_answer', np.median, 'closed'), 'gap_total': ('gap_total', np.median, None),
               'playback_completion': ('playback_completion', np.median, None),
               'accuracy': ('correct', np.mean, 'closed'), 'llm_calls': ('llm_calls', np.mean, None)}
    res: dict = {'arms': arms, 'contrasts': {}, 'levels': {}}
    for mname, (f, stat, need) in metrics.items():
        for sname, sub in subsets.items():
            if need == 'closed' and sname in ('open', 'hard_open'):
                continue
            for x in arms:
                vals = [rv[q][x].get(f) for q in ids if x in rv[q] and sub(rv[q][x]) and (need is None or rv[q][x][need])]
                vals = [v for v in vals if v is not None]
                if vals:
                    res['levels'][f'{mname}|{x}|{sname}'] = bootstrap(vals, stat)
                for y in ('catt_v4_route', 'thinking', 'non_thinking') + tuple(z for z in arms if z != x):
                    d = []
                    for q in ids:
                        if x in rv[q] and sub(rv[q][x]) and (need is None or rv[q][x][need]):
                            other = rv[q].get(y) if y in arms else comp[q][y]
                            if other is None:
                                continue
                            u, v = rv[q][x].get(f), other.get(f)
                            if u is not None and v is not None:
                                d.append(u - v)
                    if d:
                        res['contrasts'][f'{mname}|{x}-{y}|{sname}'] = bootstrap(d, stat)
    return res


# ----------------------------------------------------------------------------------------------------------------
# transfer study: CATT on three hosted models (139 items), statistics of that study (bootstrap seed 20261011)
# ----------------------------------------------------------------------------------------------------------------

TRANSFER_SEED = 20261011
T_CATT, T_TH, T_NT = 'catt_v4_route', 'thinking', 'non_thinking'
STATES = ('after_reasoning_end', 'in_flight_then_text', 'in_flight_then_skip', 'after_skip', 'no_progress_dispatched')


def t_contrast(by, a, b, metric, subset=None, stat=np.median):
    diffs = []
    for qid in sorted(by):
        d = by[qid]
        if a in d and b in d and (subset is None or subset(d)):
            x, y = d[a].get(metric), d[b].get(metric)
            if x is not None and y is not None:
                diffs.append(float(x) - float(y))
    return bootstrap(diffs, stat, TRANSFER_SEED) if diffs else None


def t_summarize(by: Dict[str, Dict[str, dict]], with_components: bool = True) -> dict:
    """`by` is ordered as the study's loader ordered its items (the continuity bootstrap depends on that order)."""
    hard = lambda d: T_CATT in d and d[T_CATT]['route'] == 'hard'  # noqa: E731
    closed = lambda d: T_CATT in d and d[T_CATT]['closed']  # noqa: E731
    out: dict = {'n_items': len(by), 'n_complete': {c: sum(1 for d in by.values() if c in d) for c in (T_CATT, T_TH, T_NT)},
                 'routes': dict(Counter(d[T_CATT]['route'] for d in by.values() if T_CATT in d))}
    out['contrasts'] = {
        'first_audio': t_contrast(by, T_CATT, T_TH, 't_first_audio'),
        'first_audio_vs_nonthinking': t_contrast(by, T_CATT, T_NT, 't_first_audio'),
        'first_substantive_all': t_contrast(by, T_CATT, T_TH, 't_first_substantive'),
        'first_substantive_all_vs_nonthinking': t_contrast(by, T_CATT, T_NT, 't_first_substantive'),
        'first_substantive_hard': t_contrast(by, T_CATT, T_TH, 't_first_substantive', hard),
        'first_substantive_hard_closed': t_contrast(by, T_CATT, T_TH, 't_first_substantive', lambda d: hard(d) and closed(d)),
        'first_substantive_hard_open': t_contrast(by, T_CATT, T_TH, 't_first_substantive', lambda d: hard(d) and not closed(d)),
        'final_answer_closed': t_contrast(by, T_CATT, T_TH, 't_final_answer', closed),
        'final_answer_hard_closed': t_contrast(by, T_CATT, T_TH, 't_final_answer', lambda d: hard(d) and closed(d)),
        'accuracy_closed': t_contrast(by, T_CATT, T_TH, 'correct', closed, np.mean),
        'accuracy_closed_vs_nonthinking': t_contrast(by, T_CATT, T_NT, 'correct', closed, np.mean),
        'calls': t_contrast(by, T_CATT, T_TH, 'llm_calls', None, np.mean)}
    gaps = np.array([d[T_CATT]['gap_total'] for d in by.values() if hard(d) and d[T_CATT].get('gap_total') is not None], dtype=float)
    if len(gaps):
        x = bootstrap(gaps, np.median, TRANSFER_SEED)
        out['continuity_hard'] = {'n': x['n'], 'median': x['estimate'], 'ci_low': x['ci_low'], 'ci_high': x['ci_high'],
                                  'p90': float(np.percentile(gaps, 90)), 'max': float(gaps.max())}
        tg = [d[T_TH]['gap_total'] for d in by.values() if hard(d) and T_TH in d and d[T_TH].get('gap_total') is not None]
        out['continuity_hard_thinking_baseline'] = {'median': float(np.median(tg)) if tg else None}
        if with_components:
            comp = {}
            for k in ('bridge_to_progress', 'between_progress', 'before_final', 'within_dispatch', 'other', 'max_single'):
                v = [d[T_CATT][f'gap_{k}'] for d in by.values() if hard(d) and d[T_CATT].get(f'gap_{k}') is not None]
                comp[k] = {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'p90': float(np.percentile(v, 90))} if v else None
            for k in ('t_first_progress', 'bridge_audio_s', 'n_progress_played'):
                v = [d[T_CATT][k] for d in by.values() if hard(d) and d[T_CATT].get(k) is not None]
                comp[k] = {'median': float(np.median(v)), 'n': len(v)} if v else None
            out['continuity_hard_components'] = comp
    out['accuracy_by_condition'] = {}
    out['medians_by_condition'] = {}
    for c in (T_CATT, T_TH, T_NT):
        v = [d[c]['correct'] for d in by.values() if c in d and d[c]['closed']]
        out['accuracy_by_condition'][c] = {'n': len(v), 'accuracy': float(np.mean(v)) if v else None}
        for m in ('t_first_audio', 't_first_substantive'):
            v = [d[c][m] for d in by.values() if c in d]
            out['medians_by_condition'][f'{c}|{m}'] = float(np.median(v)) if v else None
        v = [d[c]['t_final_answer'] for d in by.values() if c in d and d[c]['closed']]
        out['medians_by_condition'][f'{c}|t_final_answer_closed'] = float(np.median(v)) if v else None
        out['medians_by_condition'][f'{c}|llm_calls_mean'] = float(np.mean([d[c]['llm_calls'] for d in by.values() if c in d] or [np.nan]))
    g2 = [d[T_CATT]['intermediate_error'] for d in by.values()
          if hard(d) and d[T_CATT].get('g2_included') and d[T_CATT].get('intermediate_error') is not None]
    if g2:
        k = int(sum(g2))
        out['intermediate_errors'] = {'k': k, 'n': len(g2), 'rate': k / len(g2), 'wilson': list(wilson(k, len(g2)))}
    if with_components:
        modes = Counter(d[T_CATT]['first_sub_mode'] for d in by.values() if T_CATT in d)
        out['first_sub_mode_catt'] = {('null' if k is None else k): v for k, v in modes.items()}   # JSON key of None
        out['labels_missing'] = sum(1 for d in by.values() for r in d.values() if not r['labels_present'])
    return out


def t_stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.array(v, dtype=float)
    return {'n': len(v), 'median': float(np.median(a)), 'mean': float(a.mean()), 'p90': float(np.percentile(a, 90)),
            'max': float(a.max())}


def t_diagnostics(by: Dict[str, Dict[str, dict]]) -> dict:
    """Further measures of the transfer study: silence on turns routed to reasoning attributed to the thinker/talker
    state, progress calls that returned SKIP, and the final answer measured from each condition's own reasoning end."""
    att = [d[T_CATT] for d in by.values() if T_CATT in d and d[T_CATT]['route'] == 'hard']
    tot = [a['silence_attributed_total'] for a in att]
    out: dict = {'n_hard': len(att), 'silence_total': t_stats(tot),
                 'silence_by_state_mean': {st: float(np.mean([a[f'silence_{st}'] for a in att])) for st in STATES},
                 'silence_by_state_median': {st: float(np.median([a[f'silence_{st}'] for a in att])) for st in STATES},
                 'silence_by_state_share_of_total': {st: float(sum(a[f'silence_{st}'] for a in att) / max(1e-9, sum(tot)))
                                                     for st in STATES},
                 'turns_with_gap_over_5s': sum(1 for x in tot if x > 5)}
    big = [a for a in att if a['silence_attributed_total'] > 5]
    out['over_5s_by_state_share'] = ({st: float(sum(a[f'silence_{st}'] for a in big) / sum(a['silence_attributed_total'] for a in big))
                                      for st in STATES} if big else None)
    out['progress_calls_per_turn'] = t_stats([a['n_progress_calls'] for a in att])
    npc = sum(a['n_progress_calls'] for a in att)
    out['progress_skip_share'] = float(sum(a['n_progress_skip'] for a in att) / npc) if npc else None
    out['progress_cancelled_share'] = float(sum(a['n_progress_cancelled'] for a in att) / npc) if npc else None
    out['reasoning_tokens_hard'] = t_stats([a['reasoning_tokens'] for a in att])
    out['thinker_decode_tok_s_hard'] = t_stats([a['thinker_decode_tok_s'] for a in att])
    out['by_kind'] = {}
    for kind, sel in (('closed', lambda a: a['closed']), ('open', lambda a: not a['closed'])):
        sub = [a for a in att if sel(a)]
        out['by_kind'][kind] = {'n': len(sub),
                                'silence_total_median': float(np.median([a['silence_attributed_total'] for a in sub])) if sub else None,
                                'by_state_mean': {st: float(np.mean([a[f'silence_{st}'] for a in sub])) for st in STATES} if sub else None}
    fa = []
    for qid, d in sorted(by.items()):
        if T_CATT in d and T_TH in d and d[T_CATT]['closed']:
            c, t = d[T_CATT], d[T_TH]
            rc, rt = c.get('t_reasoning_end'), t.get('t_reasoning_end')
            hc, ht, tc, tt = c.get('reasoning_sha256'), t.get('reasoning_sha256'), c.get('reasoning_tokens'), t.get('reasoning_tokens')
            cens_c, cens_t = c['t_final_answer'] >= WATCHDOG, t['t_final_answer'] >= WATCHDOG
            fa.append({'diff': c['t_final_answer'] - t['t_final_answer'], 'cens_c': cens_c, 'cens_t': cens_t, 'Rc': rc, 'Rt': rt,
                       'same_reasoning': (hc == ht) if (hc and ht) else (None if (tc is None or tt is None) else abs(tc - tt) <= 1),
                       'tok_c': tc, 'tok_t': tt,
                       'overhead_c': (c['t_final_answer'] - rc) if (rc is not None and not cens_c) else None,
                       'overhead_t': (t['t_final_answer'] - rt) if (rt is not None and not cens_t) else None})
    both = [x for x in fa if not x['cens_c'] and not x['cens_t']]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    f = {'n_pairs': len(fa), 'censored_catt': sum(x['cens_c'] for x in fa), 'censored_thinking': sum(x['cens_t'] for x in fa),
         'censored_both': sum(1 for x in fa if x['cens_c'] and x['cens_t']),
         'median_diff_all': med([x['diff'] for x in fa]), 'n_negative': sum(1 for x in fa if x['diff'] < 0),
         'n_positive': sum(1 for x in fa if x['diff'] > 0), 'median_diff_uncensored': med([x['diff'] for x in both]),
         'n_uncensored': len(both),
         'median_reasoning_end_diff': med([x['Rc'] - x['Rt'] for x in both if x['Rc'] is not None and x['Rt'] is not None]),
         'median_overhead_diff': med([x['overhead_c'] - x['overhead_t'] for x in both if x['overhead_c'] is not None and x['overhead_t'] is not None]),
         'median_overhead_catt': med([x['overhead_c'] for x in both if x['overhead_c'] is not None]),
         'median_overhead_thinking': med([x['overhead_t'] for x in both if x['overhead_t'] is not None]),
         'same_reasoning_share': (float(np.mean([bool(x['same_reasoning']) for x in fa if x['same_reasoning'] is not None]))
                                  if any(x['same_reasoning'] is not None for x in fa) else None),
         'median_abs_token_diff': med([abs(x['tok_c'] - x['tok_t']) for x in fa if x['tok_c'] is not None and x['tok_t'] is not None])}
    dd = [x['diff'] for x in both]
    if dd:
        b = bootstrap(dd, np.median, TRANSFER_SEED)
        f['median_diff_uncensored_ci'] = [b['ci_low'], b['ci_high']]
    od = [x['overhead_c'] - x['overhead_t'] for x in both if x['overhead_c'] is not None and x['overhead_t'] is not None]
    if od:
        b = bootstrap(od, np.median, TRANSFER_SEED)
        f['median_overhead_diff_ci'] = [b['ci_low'], b['ci_high']]
        f['n_overhead'] = len(od)
    out['final_answer_pairing'] = f
    return out


def transfer_numbers(path: Path, v4_by) -> dict:
    runs: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    order: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rec = {k: value(v) if k not in ('run', 'key', 'question_id', 'condition', 'source', 'route', 'first_sub_mode') else (v or None)
                   for k, v in r.items()}
            runs[r['run']][r['question_id']][r['condition']] = rec
            order[r['run']][r['question_id']] = int(r['analysis_order'])
    res: dict = {}
    items = None
    for run, by in runs.items():
        ordered = {q: by[q] for q in sorted(by, key=lambda q: order[run][q])}
        res[run] = {'model_result': t_summarize(ordered), 'diagnostics': t_diagnostics(ordered)}
        items = set(by)
    # reference: the main experiment's local Qwen3-32B on the same items, with the same statistics
    ref = {}
    for q in items:
        d = {}
        for c in (T_CATT, T_TH, T_NT):
            t = v4_by[q][c]
            d[c] = {'route': t['route'], 'closed': t['closed'], 't_first_audio': t['t_first_audio'],
                    't_first_substantive': t['t_first_substantive'], 'correct': t['correct_strict'],
                    't_final_answer': t['t_final_answer'], 'llm_calls': t['llm_calls'], 'gap_total': t['gap_total'],
                    'g2_included': t.get('g2_included'), 'intermediate_error': t.get('intermediate_error'),
                    'attempt': t['attempt']}
        ref[q] = d
    ref = {q: ref[q] for q in sorted(ref, key=lambda q: min(ref[q][c]['attempt'] for c in ref[q]))}
    res['reference_qwen3_32b'] = {'model_result': t_summarize(ref, with_components=False)}
    return res


# ----------------------------------------------------------------------------------------------------------------

def compare(got: dict, exp: dict, prefix: str = '', tol: float = 1e-9, out: Optional[list] = None) -> list:
    out = [] if out is None else out
    for k, v in exp.items():
        p = f'{prefix}{k}'
        if k not in got:
            out.append((p, 'missing in recomputation'))
        elif isinstance(v, dict):
            if isinstance(got[k], dict):
                compare(got[k], v, p + ' / ', tol, out)
            else:
                out.append((p, f'type differs: {got[k]!r}'))
        elif isinstance(v, (int, float)) and isinstance(got[k], (int, float)) and v is not None:
            if not (math.isnan(v) and math.isnan(got[k])) and abs(got[k] - v) > tol * max(1.0, abs(v)):
                out.append((p, f'expected {v!r}, got {got[k]!r}'))
        elif got[k] != v:
            out.append((p, f'expected {v!r}, got {got[k]!r}'))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', default=None, help='JSON of expected values; every expected key must be reproduced')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by = load_turns(data / 'v4' / 'turns.csv')
    rv_rows = [json.loads(l) for l in open(data / 'revert' / 'turns.jsonl', encoding='utf-8') if l.strip()]
    res = {'v4': v4_numbers(by),
           'luna_retest': luna_retest_numbers(data / 'v4' / 'luna_labels.jsonl.gz', data / 'v4' / 'luna_labels_retest.jsonl.gz'),
           'listening_rating': listening_numbers(load_listen(data / 'listening_rating' / 'exp20261015001_scores.csv'),
                                                 load_listen(data / 'listening_rating' / 'exp20261015002_scores.csv'),
                                                 sorted({r['condition'] for r in rv_rows if r['condition'].startswith('revert')})),
           'revert': revert_numbers(data / 'revert' / 'turns.jsonl', by),
           'transfer': transfer_numbers(data / 'transfer' / 'turns.csv', by)}
    # router configuration: intermediate-speech errors over all turns routed to reasoning (wrong-final labels added)
    res['v4']['catt_v4_route'].update(intermediate_all_hard_numbers(by, data / 'v4' / 'luna_labels_wrong_final.jsonl.gz'))
    res['luna_retest']['intermediate_error_flag|wrong_final'] = luna_retest_numbers(
        data / 'v4' / 'luna_labels_wrong_final.jsonl.gz',
        data / 'v4' / 'luna_labels_wrong_final_retest.jsonl.gz')['intermediate_error_flag']
    (out / 'paper_numbers.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(f'written {out / "paper_numbers.json"}')
    if a.check:
        exp = json.load(open(a.check, encoding='utf-8'))
        diffs = compare(res, exp)
        n = sum(1 for _ in _leaves(exp))
        print(f'check: {n} expected values, {len(diffs)} differences')
        for p, m in diffs[:50]:
            print(f'  {p}: {m}')
        raise SystemExit(1 if diffs else 0)


def _leaves(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _leaves(v)
        else:
            yield v


if __name__ == '__main__':
    main()
