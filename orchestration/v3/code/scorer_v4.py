"""EXP-20261007-001 corrected answer extraction and answer timing (BRIEF 6.3, 6.4; design change 6).

Wraps EXP-20261005-001's ``scorer_v3.py`` (pkg/v2/code, read-only), which wraps EXP-20260930-003's
recovered ``score_results.py``. The same functions score every condition.

Additions to scorer_v3 (general rules, no item patches):
  1. MCQ copula answer: in the first sentence of the terminal text, "<anything> is B." /
     "... would be (B)," / "... is option B" reads as B. The letter must be a standalone capital A-D
     directly after the copula (optionally "option"/"choice", optional parentheses) and be followed by
     punctuation or the end of the text. Checked after the explicit statements and before the
     option-content and fallback rules of the terminal text.
  2. First candidate (explicit only): the earliest played chunk outside the bridge at which an explicit
     candidate answer has been stated within its own dispatch: MCQ explicit statements, the copula rule
     (any sentence), the option-content rule (first sentence of the dispatch); AIME explicit statements
     and the first-sentence copula rule. The standalone-letter fallback is never used, so an article
     "a/A" is never read as an option.
  3. Final-answer time: within the terminal dispatch only (scope of the final extraction); see answer_times.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / 'v2' / 'code'))
import scorer_v3 as S3  # noqa: E402

SR = S3.SR
MCQ_SOURCES, AIME_SOURCES = S3.MCQ_SOURCES, S3.AIME_SOURCES
TERMINAL_MODES = S3.TERMINAL_MODES
OPEN_SOURCES = ('voicebench_alpaca_eval', 'voicebench_alpacaeval_full')
Extraction = S3.Extraction

_COPULA_MCQ = re.compile(r"\b(?:is|was|would\s+be|should\s+be|must\s+be|be)\s+(?:option\s+|choice\s+)?\(?([A-D])\)?(?=\s*(?:[.,;:!?)]|$))")


def mcq_copula(text: str) -> Optional[str]:
    hits = [m for m in _COPULA_MCQ.finditer(text) if not re.search(r"\bnot\s*$|n't\s*$", text[max(0, m.start() - 12):m.start()])]
    return hits[-1].group(1) if hits else None


def extract(source: str, terminal_text: Optional[str], full_text: str, question: str = '') -> Extraction:
    if source not in MCQ_SOURCES:
        return S3.extract(source, terminal_text, full_text, question)
    texts = []
    if terminal_text and terminal_text.strip():
        texts.append(('terminal', terminal_text))
    texts.append(('full', full_text or ''))
    for scope, text in texts:
        if not text:
            continue
        x = S3.mcq_explicit(text)
        if x:
            return Extraction(x, f'v4/mcq_explicit/{scope}')
        x = mcq_copula(S3.first_sentence(text))
        if x:
            return Extraction(x, f'v4/mcq_copula_first_sentence/{scope}')
        x = S3.mcq_option_content(question, text) if question else None
        if x:
            return Extraction(x, f'v4/mcq_option_content/{scope}')
        r = SR.score_mmlu(text, 'A')
        if r.extracted_answer:
            return Extraction(r.extracted_answer, f'v4/fallback_{r.score_method}/{scope}')
    return Extraction('', 'v4/unscorable')


def is_correct(source: str, answer: str, gold: str) -> Optional[bool]:
    return S3.is_correct(source, answer, gold)


def terminal_and_full(chunks: Sequence[dict]) -> Tuple[Optional[str], str]:
    return S3.terminal_and_full(chunks)


def score_chunks(source: str, gold: str, chunks: Sequence[dict], question: str = '') -> Tuple[Extraction, Optional[bool]]:
    terminal, full = terminal_and_full(chunks)
    x = extract(source, terminal, full, question)
    return x, is_correct(source, x.answer, gold)


def explicit_candidate(source: str, dispatch_text: str, question: str = '') -> Optional[str]:
    """Explicit candidate answer stated in one dispatch's text so far (no fallback rules)."""
    if not dispatch_text.strip():
        return None
    if source in MCQ_SOURCES:
        return (S3.mcq_explicit(dispatch_text) or mcq_copula(dispatch_text)
                or (S3.mcq_option_content(question, dispatch_text) if question else None))
    if source in AIME_SOURCES:
        x = S3.aime_explicit(dispatch_text) or S3.aime_first_sentence(dispatch_text)
        return (x.lstrip('0') or '0') if x else None
    return None


def answer_times(source: str, gold: str, question: str, chunks: List[dict]) -> Dict[str, Optional[float]]:
    """chunks: played chunks in playback order with 'seq', 'mode', 'text', 'playback_start'.

    first_candidate: first non-bridge chunk at which explicit_candidate() of its dispatch prefix is non-empty.
    final_answer: if the final extraction comes from the terminal dispatch, the playback start of the
      earliest terminal chunk at which the terminal prefix already yields the final answer; otherwise the
      earliest non-bridge chunk at which the full-text prefix (bridge excluded) yields it.
    """
    out: Dict[str, Optional[float]] = {'first_candidate': None, 'final_answer': None, 'final': None, 'scope': None}
    if not chunks:
        return out
    x, _ = score_chunks(source, gold, chunks, question)
    out['final'] = x.answer or None
    prefix: Dict[int, str] = {}
    for c in chunks:
        if c.get('mode') == 'hard_first' or not c.get('text'):
            continue
        prefix[c['seq']] = (prefix.get(c['seq'], '') + ' ' + c['text']).strip()
        if out['first_candidate'] is None and explicit_candidate(source, prefix[c['seq']], question):
            out['first_candidate'] = c['playback_start']
    if not x.answer:
        return out
    scope = 'terminal' if x.method.endswith('/terminal') else 'full'
    out['scope'] = scope
    acc = ''
    for c in chunks:
        if not c.get('text') or c.get('mode') == 'hard_first':
            continue
        if scope == 'terminal' and c.get('mode') not in TERMINAL_MODES:
            continue
        acc = (acc + ' ' + c['text']).strip()
        y = extract(source, acc if scope == 'terminal' else None, acc, question)
        if y.answer == x.answer:
            out['final_answer'] = c['playback_start']
            break
    return out
