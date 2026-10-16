"""EXP-20261009-001 shared helpers for units, development selection and analysis (model-free)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'v3' / 'code'))
import scorer_v4 as S  # noqa: E402  (v3's frozen scorer v4, read-only)

CLOSED = S.MCQ_SOURCES + S.AIME_SOURCES
OPEN = S.OPEN_SOURCES
_END = tuple('.!?。！？')


def read_jsonl(path) -> List[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_run(run: Path) -> Iterator[Tuple[Path, dict]]:
    run = Path(run)
    for row in read_jsonl(run / 'attempts.jsonl'):
        cell = run / f"attempt-{row['attempt']:04d}"
        yield cell, json.loads((cell / 'summary.json').read_text())


def played(s: dict) -> List[dict]:
    return sorted(s.get('played_chunks') or [], key=lambda c: c['playback_start'])


def sentence_units(chunks: List[dict]) -> List[dict]:
    """Played chunks merged into sentences, identically in every condition: a chunk joins the previous unit iff both
    belong to the same dispatch (same seq) and the previous unit's text does not end with sentence-final punctuation
    (. ! ? and the CJK forms). This undoes the progress clause chunks of change 1 and the first-chunk comma split
    that every talker (CATT and baselines) uses. Unit start = playback start of its first chunk."""
    units: List[dict] = []
    for c in chunks:
        prev = units[-1] if units else None
        if prev is not None and prev['seq'] == c['seq'] and not prev['text'].rstrip().endswith(_END):
            prev['text'] += ' ' + c['text']
            prev['spoken'] += ' ' + (c.get('spoken') or c['text'])
            prev['playback_end'] = c['playback_end']
            prev['chunk_ids'].append(c.get('chunk_id'))
            continue
        units.append({'seq': c['seq'], 'mode': c.get('mode'), 'text': c['text'], 'spoken': c.get('spoken') or c['text'],
                      'playback_start': c['playback_start'], 'playback_end': c['playback_end'],
                      'chunk_ids': [c.get('chunk_id')]})
    return units


def gaps(chunks: List[dict]) -> Dict[str, float]:
    comp = {'total': 0.0, 'max_single': 0.0, 'before_final': 0.0, 'bridge_to_progress': 0.0, 'between_progress': 0.0,
            'within_dispatch': 0.0, 'other': 0.0}
    for a, b in zip(chunks, chunks[1:]):
        g = max(0.0, b['playback_start'] - a['playback_end'])
        comp['total'] += g
        comp['max_single'] = max(comp['max_single'], g)
        if a['seq'] == b['seq']:
            kind = 'within_dispatch'
        elif b.get('mode') == 'hard_final':
            kind = 'before_final'
        elif a.get('mode') == 'hard_first' and b.get('mode') == 'hard_mid':
            kind = 'bridge_to_progress'
        elif a.get('mode') == 'hard_mid' and b.get('mode') == 'hard_mid':
            kind = 'between_progress'
        else:
            kind = 'other'
        comp[kind] += g
    return comp


def final_answer_time(q: dict, s: dict) -> Optional[float]:
    """Scorer v4 final-answer time (closed-form items; None if not COMPLETE or no answer)."""
    if s['status'] != 'COMPLETE' or q['source'] not in CLOSED:
        return None
    t = S.answer_times(q['source'], q['answer'], q['question'], played(s))
    return t['final_answer']


def correct(q: dict, s: dict) -> Optional[int]:
    if q['source'] not in CLOSED:
        return None
    if s['status'] != 'COMPLETE':
        return 0
    x, ok = S.score_chunks(q['source'], q['answer'], played(s), q['question'])
    return int(ok is True)
