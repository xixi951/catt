"""EXP-20261005-001 corrected answer extraction (BRIEF section 6.3; design change 9).

Wraps EXP-20260930-003's recovered ``score_results.py`` (imported read-only) and changes only the
extraction step. The same function scores every condition.

Why a correction is needed (audit of all EXP-20260930-003 closed-form outputs, see
``scorer_audit/``): the recovered scorer reads the whole spoken text and, within each regex
level, takes the last match. Two failure modes follow.
  (a) Conclusion-first speech states the answer and then explains it; the explanation contains
      other numbers (AIME) or option letters (MCQ), and a later, weaker match wins.
  (b) In CATT turns, progress sentences that precede the final answer can supply the match.

General rules (no item patches):
  1. Score the terminal dispatch (easy, hard_final, or the baseline's single call) when it is
     known; fall back to the whole spoken text only if the terminal text yields nothing.
  2. Explicit answer statements outrank everything else. MCQ: "the (final/correct) answer is X",
     "answer: X", "the answer is option X", "option X is correct", "X is the correct answer".
     AIME: "the (final) answer is N", "answer: N", boxed (full integer, not truncated to three
     digits). Statements whose letter or number is negated ("is not B") never match.
  3. If several explicit statements disagree, the last one wins (self-correction convention of
     the recovered scorer).
  4. AIME, no explicit statement: the answer contract of every condition is conclusion-first,
     so the first sentence of the terminal text is read for a copula statement ("... is N",
     "... = N", "... equals N"; last such statement in that sentence). Excluded: single-letter
     assignments ("m = 35"), and numbers followed by a fraction bar, power, root, decimal part
     or percent sign (not a final integer).
  5. MCQ, no letter anywhere: if exactly one option's text occurs in the first sentence of the
     terminal text, that option's letter is the answer.
  6. Otherwise the recovered scorer's own hierarchy runs on the terminal text, then on the full
     text.
  7. Spelled-out integers 0-999 in an explicit AIME statement are converted to digits.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve()
for _cand in (_HERE.parents[1] / 'exp003' / 'snapshot' / 'scripts',
              Path('/Users/xixi/research/metahuman/project_management/experiments/EXP-20260930-003/snapshot/scripts')):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
import score_results as SR  # noqa: E402

MCQ_SOURCES = ('gpqa_diamond', 'gpqa_main', 'voicebench_mmsu', 'mmlu', 'mmlu_pro')
AIME_SOURCES = ('aime_2026', 'aime_2025', 'aime_2024')
TERMINAL_MODES = ('easy', 'hard_final', 'baseline')

_L = r'([A-D])'
_MCQ_EXPLICIT = [
    re.compile(r"\b(?:final\s+answer|correct\s+answer|right\s+answer|best\s+answer|the\s+answer|answer)"
               r"(?:\s+(?:is|would\s+be|should\s+be|must\s+be)|\s*:|'s)\s*(?:option\s+|choice\s+)?\(?" + _L + r"\)?(?![\w'])",
               re.IGNORECASE),
    re.compile(r"\b(?:option|choice)\s+\(?" + _L + r"\)?\s*(?:,\s*[^,.;]{0,80},\s*)?(?:is|would\s+be)\s+(?:the\s+)?(?:correct|right|best|answer)\b",
               re.IGNORECASE),
    re.compile(r"(?<![\w'])\(?" + _L + r"\)?\s+is\s+(?:the\s+)?(?:correct|right|best)\s+(?:answer|option|choice)\b"),
    re.compile(r"\bgo(?:ing)?\s+with\s+(?:option\s+|choice\s+)?\(?" + _L + r"\)?(?![\w'])", re.IGNORECASE),
    re.compile(r"\b(?:correct|right|best)\s+(?:[a-z-]+\s+){1,3}(?:is|would\s+be)\s+(?:option\s+|choice\s+)?\(?"
               + _L + r"\)?(?![\w'])", re.IGNORECASE),
]

_UNITS = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen '
    'sixteen seventeen eighteen nineteen'.split())}
_TENS = {w: 10 * i for i, w in enumerate('_ _ twenty thirty forty fifty sixty seventy eighty ninety'.split()) if w != '_'}
_NUMWORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|" \
           r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|and|-|\s)+"
_AIME_EXPLICIT = re.compile(
    r"\b(?:final\s+answer|the\s+answer|answer)(?:\s+(?:is|would\s+be|comes\s+out\s+to|equals)|\s*:|'s)\s*"
    r"(?:\\\(|\$)?\s*(?:\*\*)?\s*(?:\\boxed\{)?(\d+|" + _NUMWORD + r")", re.IGNORECASE)
# Copula statement in the first sentence: "... is N", "... = N", "... equals N", optional $ / ** / boxed.
_COPULA = re.compile(r"(?P<lhs>[^\s$=]*\s*\$?\s*)(?:\bis|\bequals|=)\s*\$?\s*(?:\*\*)?\s*(?:\\boxed\{)?\s*"
                     r"(?P<num>\d+)(?P<after>[^\s]{0,3}\s*\S{0,6})")
# A lone variable on the left of '=' ("m = 35", "so $ N = 6776"); "m + n = 277" is not one.
_SINGLE_LETTER_LHS = re.compile(r"(?:^|[^\w+\-*/^\s$]|\b(?:so|and|thus|then|with|where|giving|gives|has)\b)[\s$]*([A-Za-z])\s*\$?\s*$")


def first_sentence(text: str) -> str:
    """Text up to the first sentence end ('.', '!' or '?' followed by whitespace and an uppercase
    letter, '$', or end of text). A period directly between digits ('113. 06') does not end it."""
    for m in re.finditer(r"[.!?](?=\s+[A-Z$*\\]|\s*$)", text):
        before = text[max(0, m.start() - 1):m.start()]
        after = text[m.end():m.end() + 3]
        if before.isdigit() and re.match(r"\s?\d", after):
            continue
        return text[:m.end()]
    return text


def aime_first_sentence(text: str) -> Optional[str]:
    sent = first_sentence(text)
    found: List[str] = []
    for m in _COPULA.finditer(sent):
        after = m.group('after')
        if re.match(r"\s*(?:/|\^|\\sqrt|\\frac|\.\d|%|\\%|\\cdot|\\times|\\pi)", after):
            continue
        lhs_ctx = sent[max(0, m.start('num') - 40):m.start('num')]
        is_eq = re.search(r"=\s*\$?\s*(?:\*\*)?\s*(?:\\boxed\{)?\s*$", lhs_ctx) is not None
        if is_eq:
            left = re.sub(r"=\s*\$?\s*(?:\*\*)?\s*(?:\\boxed\{)?\s*$", "", lhs_ctx)
            if _SINGLE_LETTER_LHS.search(left):
                continue
        found.append(m.group('num'))
    return found[-1] if found else None


_OPTION_LINE = re.compile(r"^\s*([A-D])[.)]\s+(.+?)\s*$", re.MULTILINE)


def mcq_option_content(question: str, text: str) -> Optional[str]:
    sent = re.sub(r"\s+", " ", first_sentence(text).lower())
    options = {m.group(1): re.sub(r"\s+", " ", m.group(2).lower()).strip(' .') for m in _OPTION_LINE.finditer(question)}
    hits = [letter for letter, opt in options.items() if len(opt) >= 3 and opt in sent]
    return hits[0] if len(hits) == 1 else None


def words_to_int(text: str) -> Optional[int]:
    tokens = [t for t in re.split(r'[\s-]+', text.lower().strip()) if t and t != 'and']
    if not tokens:
        return None
    total, current = 0, 0
    for t in tokens:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == 'hundred':
            current = max(current, 1) * 100
        else:
            return None
    total += current
    return total if 0 <= total <= 999 else None


def _negated(text: str, start: int) -> bool:
    return bool(re.search(r"\bnot\s*$|\bisn't\s*$", text[max(0, start - 12):start], re.IGNORECASE))


def mcq_explicit(text: str) -> Optional[str]:
    hits: List[Tuple[int, str]] = []
    for pat in _MCQ_EXPLICIT:
        for m in pat.finditer(text):
            seg = text[m.start():m.end()]
            if re.search(r"\bnot\b|n't\b", seg, re.IGNORECASE):
                continue
            hits.append((m.start(1), m.group(1).upper()))
    if not hits:
        return None
    hits.sort()
    return hits[-1][1]


def aime_explicit(text: str) -> Optional[str]:
    boxed = SR._extract_last_boxed(text)
    candidates: List[Tuple[int, str]] = []
    if boxed:
        digits = re.search(r"\d+", boxed)
        if digits:
            candidates.append((text.rfind('\\boxed'), digits.group(0)))
    for m in _AIME_EXPLICIT.finditer(text):
        if _negated(text, m.start(1)):
            continue
        tok = m.group(1).strip()
        if tok.isdigit():
            candidates.append((m.start(1), tok))
        else:
            value = words_to_int(tok)
            if value is not None:
                candidates.append((m.start(1), str(value)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


@dataclass(frozen=True)
class Extraction:
    answer: str
    method: str


def extract(source: str, terminal_text: Optional[str], full_text: str, question: str = '') -> Extraction:
    """Corrected extraction. ``terminal_text`` is the text of the terminal dispatch (None if unknown)."""
    texts = []
    if terminal_text and terminal_text.strip():
        texts.append(('terminal', terminal_text))
    texts.append(('full', full_text or ''))
    if source in MCQ_SOURCES:
        # Terminal text is exhausted (explicit, option content, recovered hierarchy) before the
        # full text is read, so a letter that only a progress sentence mentions cannot win.
        for scope, text in texts:
            if not text:
                continue
            x = mcq_explicit(text)
            if x:
                return Extraction(x, f'v3/mcq_explicit/{scope}')
            x = mcq_option_content(question, text) if question else None
            if x:
                return Extraction(x, f'v3/mcq_option_content/{scope}')
            r = SR.score_mmlu(text, 'A')
            if r.extracted_answer:
                return Extraction(r.extracted_answer, f'v3/fallback_{r.score_method}/{scope}')
        return Extraction('', 'v3/unscorable')
    if source in AIME_SOURCES:
        for scope, text in texts:
            x = aime_explicit(text)
            if x:
                return Extraction(x.lstrip('0') or '0', f'v3/aime_explicit/{scope}')
        for scope, text in texts:
            x = aime_first_sentence(text) if text else None
            if x:
                return Extraction(x.lstrip('0') or '0', f'v3/aime_first_sentence/{scope}')
        for scope, text in texts:
            x = SR._extract_aime_integer(text) if text else None
            if x:
                return Extraction(x.lstrip('0') or '0', f'v3/fallback_aime/{scope}')
        return Extraction('', 'v3/unscorable')
    return Extraction('', f'v3/not_scored/{source}')


def is_correct(source: str, answer: str, gold: str) -> Optional[bool]:
    if not answer:
        return None
    if source in MCQ_SOURCES:
        return answer.strip().upper() == gold.strip().upper()
    if source in AIME_SOURCES:
        return (answer.strip().lstrip('0') or '0') == (gold.strip().lstrip('0') or '0')
    return None


def terminal_and_full(chunks: Sequence[dict]) -> Tuple[Optional[str], str]:
    """chunks: played chunks in playback order with 'mode' and 'text'. Returns (terminal, full)."""
    full = ' '.join(c['text'] for c in chunks if c.get('text'))
    term = [c['text'] for c in chunks if c.get('text') and c.get('mode') in TERMINAL_MODES]
    return (' '.join(term) if term else None), full


def score_chunks(source: str, gold: str, chunks: Sequence[dict], question: str = '') -> Tuple[Extraction, Optional[bool]]:
    terminal, full = terminal_and_full(chunks)
    x = extract(source, terminal, full, question)
    return x, is_correct(source, x.answer, gold)
