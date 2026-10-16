"""Build the per-turn table (`data/v4/turns.csv`) from run records, fixture, gpt-5.6-luna labels and GPU samples.

Derived from the study's frozen per-attempt code without changing any computed value: played chunks and sentence
units (orchestration/code/v5_common.py), strict scoring and answer timing (scorer v4), silence components, cost and
energy. Release changes: only gpt-5.6-luna labels are read (first substantive content and intermediate-speech
errors are luna's, as in the paper), only the released columns are written, and the extracted answer of GPQA and
AIME items is left empty. Rows are sorted by item and condition; the 64 items that set the routing threshold are skipped with
--exclude. Run on the study's records, the output equals `data/v4/turns.csv` byte for byte.

Usage: python build_turns.py --runs RUN/MAIN --run-label formal_v4 --fixtures main.jsonl --hard hard_lists.json
                             --labels LABEL_DIR [--sampler gpu_*.csv ...] [--exclude excluded_ids.txt] --out turns.csv
`--hard` is a JSON object with the item lists routed to reasoning by the pre-pass: {"route": [...], "nvroute": [...]}
(`data/items.csv` columns route_router and route_nvidia hold the same lists).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ORCH = Path(__file__).resolve().parents[2] / 'orchestration'
sys.path.insert(0, str(ORCH / 'code'))
import v5_common as C  # noqa: E402  (frozen helpers of the main experiment)

S = C.S
WATCHDOG = 1200.0
LUNA = 'gpt-5.6-luna'
CATT = ('catt_v4_route', 'catt_v4_nvroute', 'catt_v4_allhard')
COLUMNS = [
    'key', 'attempt', 'question_id', 'source', 'condition', 'status', 'kind', 'closed', 'open',
    'route', 'route_confidence', 'route_decided_s', 'hard_item', 'hard_item|route', 'hard_item|nvroute',
    't_first_audio', 't_first_progress', 't_first_candidate', 't_final_answer', 'playback_completion',
    'correct_strict', 'extracted', 'extract_method',
    'gap_total', 'gap_before_final', 'gap_between_progress', 'gap_bridge_to_progress', 'gap_max_single', 'gap_other',
    'gap_within_dispatch',
    'n_played', 'n_units', 'n_dropped', 'interleaved', 'preempted', 'cut_mid_sentence',
    'progress_first', 'progress_regular', 'progress_low_queue', 'final_prompt', 'easy_prompt',
    'bridge_audio_s', 'bridge_played_before_easy', 'bridge_guard_v4', 'norm_changed_chunks', 'norm_long_formula',
    'g2_included', 'llm_calls', 'calls_thinker', 'calls_talker', 'peak_concurrent_llm_calls', 'budget_forced_calls',
    'forced', 'tokens_prompt', 'tokens_completion', 'tokens_prompt_thinker', 'tokens_prompt_talker',
    'tokens_completion_thinker', 'tokens_completion_talker', 'tokens_reasoning', 'reasoning_sha256', 'wall_seconds',
    'gpu_energy_j', 't_first_substantive', 'first_substantive_unit', 'intermediate_error']


class LunaLabels:
    """gpt-5.6-luna substantive labels (offset 0: first 25 sentences; offset 25: full turn) and update labels."""

    def __init__(self, label_dir: Path) -> None:
        self.sub: Dict[str, List[bool]] = {}
        self.sub_full: Dict[str, List[bool]] = {}
        self.g2: Dict[str, List[str]] = {}
        for task in ('substantive', 'g2'):
            p = Path(label_dir) / f'{task}__{LUNA}.jsonl'
            if not p.exists():
                continue
            for r in C.read_jsonl(p):
                if not r.get('ok'):
                    continue
                if task == 'substantive':
                    (self.sub_full if r.get('offset', 0) else self.sub)[r['key']] = r['result']
                else:
                    self.g2[r['key']] = [x['label'] for x in r['result']]

    def sub_labels(self, key: str) -> Optional[List[bool]]:
        return self.sub_full.get(key) or self.sub.get(key)


class Energy:
    """Gross energy of the GPUs over a time window, from NVML samples (columns epoch, index, energy_mj)."""

    def __init__(self, paths: List[Path]) -> None:
        rows: Dict[int, list] = defaultdict(list)
        for path in paths:
            with Path(path).open() as handle:
                for row in csv.DictReader(handle):
                    if row.get('epoch') in (None, 'epoch') or not row.get('energy_mj'):
                        continue
                    rows[int(row['index'])].append((float(row['epoch']), float(row['energy_mj'])))
        self.series = {i: np.array(sorted(v)) for i, v in rows.items()}

    def joules(self, start: float, end: float) -> Optional[float]:
        total = 0.0
        for arr in self.series.values():
            if arr.size == 0 or start < arr[0, 0] or end > arr[-1, 0]:
                return None
            total += (np.interp(end, arr[:, 0], arr[:, 1]) - np.interp(start, arr[:, 0], arr[:, 1])) / 1000.0
        return total


def first_true(labels: Optional[List[bool]]) -> Optional[int]:
    if labels is None:
        return None
    return next((i for i, v in enumerate(labels) if v), None)


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


def turn_record(run: Path, row: dict, label: str, fixture: Dict[str, dict], hard: Dict[str, set], L: LunaLabels,
                energy: Optional[Energy]) -> dict:
    cell = run / f"attempt-{row['attempt']:04d}"
    s = json.loads((cell / 'summary.json').read_text())
    key = f"{label}/{s['set']}/{s['attempt']:04d}"
    q = fixture[s['question_id']]
    cond = s['condition']
    chunks = C.played(s)
    units = C.sentence_units(chunks)
    complete = s['status'] == 'COMPLETE'
    cfg_key = cond.replace('catt_v4_', '') if cond in CATT else None
    rec: Dict[str, Any] = {'key': key, 'attempt': s['attempt'], 'condition': cond, 'question_id': s['question_id'],
                           'source': q['source'], 'status': s['status'], 'kind': s.get('kind'),
                           'closed': q['source'] in C.CLOSED, 'open': q['source'] in S.OPEN_SOURCES,
                           'n_played': len(chunks), 'n_units': len(units),
                           'n_dropped': len(s.get('dropped_chunks') or []), 'llm_calls': s['llm_calls'],
                           'budget_forced_calls': s['budget_forced_calls'],
                           'peak_concurrent_llm_calls': s['peak_concurrent_llm_calls']}
    for c in ('route', 'nvroute'):
        rec[f'hard_item|{c}'] = s['question_id'] in hard[c]
    rec['hard_item'] = rec.get(f'hard_item|{cfg_key}', True) if cfg_key in ('route', 'nvroute') else True
    rec['route'] = route_of(cell, s)
    sp = s.get('speculative') or {}
    rec['route_confidence'] = sp.get('confidence')
    rec['route_decided_s'] = sp.get('route_ts')
    rec['bridge_played_before_easy'] = int(bool(sp.get('route') == 'easy' and sp.get('bridge_kept_n')))
    rec['t_first_audio'] = chunks[0]['playback_start'] if chunks else WATCHDOG
    idx = first_true(L.sub_labels(key))
    rec['first_substantive_unit'] = idx
    rec['t_first_substantive'] = (units[idx]['playback_start'] if (complete and idx is not None and idx < len(units))
                                  else WATCHDOG)
    if rec['closed']:
        x, ok = S.score_chunks(q['source'], q['answer'], chunks, q['question'])
        times = S.answer_times(q['source'], q['answer'], q['question'], chunks)
        rec.update(extracted='' if q['source'] in ('gpqa_diamond', 'aime_2026') else x.answer, extract_method=x.method,
                   correct_strict=int(complete and ok is True),
                   t_first_candidate=times['first_candidate'] if complete else None,
                   t_final_answer=(times['final_answer'] if (complete and times['final_answer'] is not None) else WATCHDOG))
    else:
        rec.update(extracted='', extract_method='not_scored', correct_strict=None, t_first_candidate=None, t_final_answer=None)
    g = C.gaps(chunks) if chunks else None
    rec['gap_total'] = g['total'] if g else None
    if g:
        for k, v in g.items():
            if k != 'total':
                rec[f'gap_{k}'] = v
    rec['playback_completion'] = max((c['playback_end'] for c in chunks), default=None)
    modes = [c['mode'] for c in chunks]
    first_final = next((i for i, m in enumerate(modes) if m == 'hard_final'), None)
    rec['interleaved'] = int(first_final is not None and any(m == 'hard_mid' for m in modes[first_final:]))
    mids = [c['playback_start'] for c in chunks if c['mode'] == 'hard_mid']
    rec['t_first_progress'] = min(mids) if mids else None
    bridge = [c for c in chunks if c['mode'] == 'hard_first']
    rec['bridge_audio_s'] = sum(c['playback_end'] - c['playback_start'] for c in bridge) if bridge else None
    pre = s.get('preemption') or {}
    rec['preempted'] = int(bool(pre))
    mid_units = [u for u in units if u['mode'] == 'hard_mid']
    rec['cut_mid_sentence'] = int(bool(pre) and bool(mid_units) and not mid_units[-1]['text'].rstrip().endswith(tuple('.!?')))
    tc = s.get('trigger_counts') or {}
    rec.update({f'progress_{k}': v for k, v in tc.items()})
    pv = s.get('prompt_variant') or {}
    rec['final_prompt'] = pv.get('final')
    rec['easy_prompt'] = pv.get('easy')
    nc = s.get('norm_counts') or {}
    rec['norm_changed_chunks'] = nc.get('changed')
    rec['norm_long_formula'] = nc.get('long_formula')
    rec['bridge_guard_v4'] = s.get('bridge_guard_fired_v4')
    if cond in CATT and rec['route'] == 'hard':
        included = rec['open'] or (rec['closed'] and rec.get('correct_strict') == 1)
        rec['g2_included'] = int(complete and included)
        labs = L.g2.get(key)
        rec['intermediate_error'] = int(any(v in ('WRONG', 'REVERSED') for v in labs)) if labs is not None else None
    tok = defaultdict(int)
    for call in s['calls'].values():
        eng = call.get('engine') or 'thinker'
        u1, u2 = call.get('usage_phase1') or {}, call.get('usage_phase2') or {}
        tok[f'prompt_{eng}'] += (u1.get('prompt_tokens') or 0) + (u2.get('prompt_tokens') or 0)
        tok[f'completion_{eng}'] += (u1.get('completion_tokens') or 0) + (u2.get('completion_tokens') or 0)
        if call.get('thinking'):
            if call.get('budget_forced'):
                tok['reasoning'] += call.get('reasoning_tokens_at_stop') or 0
            elif call.get('reasoning_tokens_upto_first_content') is not None:
                tok['reasoning'] += max(0, call['reasoning_tokens_upto_first_content'] - 1)
            else:
                tok['reasoning'] += u1.get('completion_tokens') or 0
    rec.update({f'tokens_{k}': v for k, v in tok.items()})
    rec['tokens_prompt'] = tok['prompt_thinker'] + tok['prompt_talker']
    rec['tokens_completion'] = tok['completion_thinker'] + tok['completion_talker']
    for e in ('thinker', 'talker'):
        rec[f'calls_{e}'] = (s.get('llm_calls_by_engine') or {}).get(e)
    rec['reasoning_sha256'] = s.get('reasoning_sha256')
    rec['forced'] = int(bool((s.get('reasoning') or {}).get('forced'))) if cond in CATT else int(s['budget_forced_calls'] > 0)
    rec['wall_seconds'] = (s['wall_end'] - s['wall_start']) if s.get('wall_start') and s.get('wall_end') else None
    rec['gpu_energy_j'] = None
    if energy is not None and s.get('wall_start') and s.get('wall_end'):
        rec['gpu_energy_j'] = energy.joules(s['wall_start'], s['wall_end'])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--run-label', default=None)
    ap.add_argument('--fixtures', nargs='+', required=True)
    ap.add_argument('--hard', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--sampler', nargs='*', default=[])
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fixture = {r['id']: r for f in args.fixtures for r in C.read_jsonl(Path(f))}
    hl = json.loads(Path(args.hard).read_text())
    hard = {'route': set(hl['route']), 'nvroute': set(hl['nvroute'])}
    excluded = set()
    if args.exclude:
        excluded = {l.strip() for l in open(args.exclude, encoding='utf-8') if l.strip()}
    L = LunaLabels(Path(args.labels))
    energy = Energy([Path(p) for p in args.sampler]) if args.sampler else None
    records = []
    for run_name in args.runs:
        run = Path(run_name)
        label = args.run_label or run.parent.name
        for row in C.read_jsonl(run / 'attempts.jsonl'):
            r = turn_record(run, row, label, fixture, hard, L, energy)
            if r['question_id'] not in excluded:
                records.append(r)
    records.sort(key=lambda r: (r['question_id'], r['condition']))
    with open(args.out, 'w', newline='', encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)
    print(json.dumps({'turns': len(records)}))


if __name__ == '__main__':
    main()
