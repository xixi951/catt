"""EXP-20261009-001 (CATT v4, change 5): text fixes shared by every condition.

* ``spoken(text)``: one deterministic normalizer applied to every chunk right before TTS, in all
  conditions (CATT configurations and both baselines). It turns LaTeX / markdown / math symbols into
  words and replaces any formula whose spoken form would exceed ``LONG_FORMULA_WORDS`` words with a
  short phrase, so no condition reads a long formula aloud. The displayed (scored) text is unchanged.
  The post-study fix adds two pre-passes, everything else is the frozen
  EXP-20261009-001 file: markdown emphasis asterisks are removed (they were read as "times") and a
  spaced dash between words is kept as a dash (it was read as "minus"); see ``_strip_emphasis`` and
  ``_protect_prose_dashes``. Math reads as before.
* ``split_sentences(text, accept_comma)``: the sentence splitter used by every talker. Unlike the
  recovered splitter it never ends a sentence at a decimal point, a thousands comma, an initial or a
  common abbreviation, and it waits for the next character before deciding on a period.
* ``split_clauses(sentence)``: clause chunks of a progress sentence (final-answer cut-in, change 1).
* ``question_kind(text)``: 'closed' (multiple choice or boxed answer) or 'open', from the question
  text alone (change 4).
Nothing here raises: on an internal error the input is returned unchanged.
"""
from __future__ import annotations

import re
from typing import List, Tuple

LONG_FORMULA_WORDS = 20
LONG_FORMULA_PHRASE = 'a longer expression'

# ---------------------------------------------------------------------------
# Question kind (change 4)
# ---------------------------------------------------------------------------

_OPTION_LINE = re.compile(r'(?m)^\s*\(?([A-J])[.):]\s+\S')


def question_kind(text: str) -> str:
    """'closed' iff the question asks for a boxed answer or lists at least two lettered options on their
    own lines (A) ..., B. ..., (C) ...); otherwise 'open'."""
    if '\\boxed' in text:
        return 'closed'
    letters = {m.group(1) for m in _OPTION_LINE.finditer(text)}
    return 'closed' if len(letters) >= 2 else 'open'


# ---------------------------------------------------------------------------
# Sentence splitter (change 5)
# ---------------------------------------------------------------------------

_ABBREV = {'e.g', 'i.e', 'etc', 'vs', 'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'no', 'fig', 'eq',
           'approx', 'cf', 'al', 'ca', 'inc', 'ltd', 'co', 'mt', 'u.s', 'u.k', 'e.u', 'a.m', 'p.m', 'vol', 'ch',
           'sec', 'min', 'max', 'resp', 'dept', 'est', 'jan', 'feb', 'mar', 'apr', 'aug', 'sept', 'oct', 'nov',
           'dec', 'ph.d', 'ref', 'refs', 'eqs', 'figs', 'nos'}
_ABBREV_SENTENCE_END_OK = {'etc'}   # "etc." usually ends a sentence when followed by a capital
_HARD_TERMINATORS = '。！？；!?;\n'
_COMMA_TERMINATORS = '，,、'


def _period_ends_sentence(text: str, i: int) -> bool:
    """Decide whether the '.' at ``text[i]`` ends a sentence; ``text[i + 1]`` must exist."""
    nxt = text[i + 1]
    if not (nxt.isspace() or nxt in '"\'”’)]'):
        return False                          # decimal point, "U.S", "e.g.,", file.ext, "3.14"
    m = re.search(r'([A-Za-z][A-Za-z.]*)$', text[:i])
    word = m.group(1).lower() if m else ''
    if word:
        if len(word) == 1 and text[i - 1].isupper():
            return False                      # initial: "J. Smith", option letter "A." in a list is rare in speech
        if word in _ABBREV:
            if word in _ABBREV_SENTENCE_END_OK:
                rest = text[i + 1:].lstrip()
                return bool(rest) and rest[0].isupper()
            return False
    if i > 0 and text[i - 1].isdigit():
        # "in 2020." ends a sentence; an enumerator such as "1." at the very start does not
        before = text[:i].rstrip('0123456789')
        if before.strip() == '':
            return False
    return True


def split_sentences(text: str, *, accept_comma: bool = False) -> Tuple[List[str], str]:
    """Split a streaming buffer into complete sentences plus the unfinished tail. A '.' (or a comma when
    ``accept_comma``) is only decided once the following character has arrived."""
    sentences: List[str] = []
    start = 0
    n = len(text)
    for i, ch in enumerate(text):
        end = False
        if ch in _HARD_TERMINATORS:
            end = True
        elif ch == '.':
            if i + 1 >= n:
                break                             # wait for the next character
            end = _period_ends_sentence(text, i)
        elif accept_comma and ch in _COMMA_TERMINATORS:
            if ch == ',':
                if i + 1 >= n:
                    break
                end = text[i + 1].isspace() and not (i > 0 and text[i - 1].isdigit() and text[i + 1:i + 2].isdigit())
            else:
                end = True
        if end:
            # keep closing quotes/brackets with the sentence
            j = i + 1
            while j < n and text[j] in '"\'”’)]' and ch != '\n':
                j += 1
            s = text[start:j].strip()
            if s:
                sentences.append(s)
            start = j
    return sentences, text[start:]


_CLAUSE_SPLIT = re.compile(r'(?<=[,;:])\s+|\s+(?=—)|(?<=—)\s+')
CLAUSE_MIN_WORDS = 4


def split_clauses(sentence: str, min_words: int = CLAUSE_MIN_WORDS) -> List[str]:
    """Clause chunks of one sentence for playback: split after ',', ';', ':' (not inside numbers) and around
    an em dash, then merge pieces shorter than ``min_words`` words into their neighbour so no chunk is a
    fragment. Joining the chunks with spaces gives back the sentence."""
    try:
        parts = [p for p in _CLAUSE_SPLIT.split(sentence.strip()) if p]
        if len(parts) <= 1:
            return [sentence.strip()] if sentence.strip() else []
        merged: List[str] = []
        for p in parts:
            if merged and len(merged[-1].split()) < min_words:
                merged[-1] = merged[-1] + ' ' + p
            else:
                merged.append(p)
        if len(merged) > 1 and len(merged[-1].split()) < min_words:
            last = merged.pop()
            merged[-1] = merged[-1] + ' ' + last
        return merged
    except Exception:
        return [sentence.strip()]


# ---------------------------------------------------------------------------
# Spoken-form normalizer (change 5)
# ---------------------------------------------------------------------------

_GREEK = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota',
          'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi',
          'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Phi', 'Psi', 'Omega']
_CMD_WORDS = {
    'times': ' times ', 'cdot': ' times ', 'div': ' divided by ', 'pm': ' plus or minus ', 'mp': ' minus or plus ',
    'le': ' less than or equal to ', 'leq': ' less than or equal to ', 'ge': ' greater than or equal to ',
    'geq': ' greater than or equal to ', 'neq': ' not equal to ', 'ne': ' not equal to ', 'approx': ' approximately ',
    'sim': ' approximately ', 'equiv': ' is equivalent to ', 'infty': ' infinity ', 'to': ' to ', 'rightarrow': ' gives ',
    'Rightarrow': ' implies ', 'implies': ' implies ', 'leftarrow': ' from ', 'sum': ' the sum of ',
    'prod': ' the product of ', 'int': ' the integral of ', 'partial': ' partial ', 'nabla': ' del ',
    'log': ' log ', 'ln': ' natural log of ', 'sin': ' sine ', 'cos': ' cosine ', 'tan': ' tangent ', 'exp': ' exp ',
    'lim': ' the limit ', 'min': ' min ', 'max': ' max ', 'cap': ' intersect ', 'cup': ' union ', 'in': ' in ',
    'subset': ' subset of ', 'forall': ' for all ', 'exists': ' there exists ', 'angle': ' angle ',
    'perp': ' perpendicular to ', 'parallel': ' parallel to ', 'circ': ' degrees ', 'degree': ' degrees ',
    'ldots': ' ... ', 'cdots': ' ... ', 'dots': ' ... ', 'hbar': ' h bar ', 'ell': ' l ', 'prime': ' prime ',
    'cong': ' is congruent to ', 'mid': ' divides ', 'pmod': ' mod ', 'bmod': ' mod ', 'mod': ' mod ',
    'binom': ' choose ', 'choose': ' choose ', 'neg': ' not ', 'land': ' and ', 'lor': ' or ',
}
_DROP_CMDS = {'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'displaystyle', 'quad', 'qquad', 'limits', 'nolimits',
              'textstyle', 'scriptstyle', ',', ';', '!', ':', ' '}
_UNICODE = {'×': ' times ', '·': ' times ', '÷': ' divided by ', '±': ' plus or minus ', '≤': ' less than or equal to ',
            '≥': ' greater than or equal to ', '≠': ' not equal to ', '≈': ' approximately ', '∞': ' infinity ',
            '→': ' to ', '⇒': ' implies ', '√': ' the square root of ', 'π': ' pi ', '²': ' squared', '³': ' cubed',
            '°': ' degrees ', '∑': ' the sum of ', '∫': ' the integral of ', '∆': ' delta ', 'Δ': ' delta ',
            'θ': ' theta ', 'λ': ' lambda ', 'μ': ' mu ', 'σ': ' sigma ', 'α': ' alpha ', 'β': ' beta ',
            'γ': ' gamma ', 'ω': ' omega ', 'Ω': ' ohm ', '−': '-', '–': '-', '∈': ' in ', '≡': ' is equivalent to '}
_MATH_CHARS = set('=+^_\\{}×÷±≤≥≠≈√∑∫<>|')


def _read_group(s: str, i: int) -> Tuple[str, int]:
    """Read a {...} group (or a single token) starting at s[i]; return (content, next index)."""
    while i < len(s) and s[i] == ' ':
        i += 1
    if i >= len(s):
        return '', i
    if s[i] == '{':
        depth, j = 0, i
        while j < len(s):
            if s[j] == '{':
                depth += 1
            elif s[j] == '}':
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    if s[i] == '\\':
        m = re.match(r'\\[A-Za-z]+', s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _power_words(exp: str) -> str:
    e = exp.strip()
    if e == '2':
        return ' squared '
    if e == '3':
        return ' cubed '
    if e in ('\\circ', '°', 'o'):
        return ' degrees '
    if e in ('-1',):
        return ' to the minus one '
    return f' to the power of {_latex_to_words(e)} '


def _latex_to_words(s: str) -> str:
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '\\':
            m = re.match(r'\\([A-Za-z]+|.)', s[i:])
            name = m.group(1) if m else ''
            i += 1 + len(name)
            if name in ('frac', 'dfrac', 'tfrac', 'cfrac'):
                a, i = _read_group(s, i)
                b, i = _read_group(s, i)
                out.append(f' {_latex_to_words(a)} over {_latex_to_words(b)} ')
            elif name == 'sqrt':
                idx = ''
                if i < n and s[i] == '[':
                    j = s.find(']', i)
                    idx, i = (s[i + 1:j], j + 1) if j > 0 else ('', i)
                a, i = _read_group(s, i)
                out.append(f' the {_latex_to_words(idx)}-th root of {_latex_to_words(a)} ' if idx
                           else f' the square root of {_latex_to_words(a)} ')
            elif name in ('boxed', 'text', 'mathrm', 'mathbf', 'mathit', 'textbf', 'textit', 'operatorname', 'mathsf',
                          'mathcal', 'mathbb', 'emph', 'vec', 'hat', 'bar', 'overline', 'underline', 'tilde', 'dot',
                          'ce', 'mbox', 'textrm', 'pmb', 'bm'):
                a, i = _read_group(s, i)
                out.append(f' {_latex_to_words(a)} ')
            elif name == 'binom':
                a, i = _read_group(s, i)
                b, i = _read_group(s, i)
                out.append(f' {_latex_to_words(a)} choose {_latex_to_words(b)} ')
            elif name in _GREEK:
                out.append(f' {name.replace("var", "").lower()} ')
            elif name in _CMD_WORDS:
                out.append(_CMD_WORDS[name])
            elif name in _DROP_CMDS:
                out.append(' ')
            elif name in ('%', '$', '&', '#', '_', '{', '}'):
                out.append({'%': ' percent', '$': ' dollars', '&': ' and ', '#': ' number ', '_': ' ',
                            '{': ' ', '}': ' '}[name])
            elif name in ('(', ')', '[', ']'):
                out.append(' ')
            elif name.isalpha():
                out.append(f' {name} ')
            else:
                out.append(' ')
        elif c == '^':
            a, i = _read_group(s, i + 1)
            out.append(_power_words(a))
        elif c == '_':
            a, i = _read_group(s, i + 1)
            out.append(f' {_latex_to_words(a)} ')
        elif c in '{}$':
            out.append(' ')
            i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


_MATH_SPAN = re.compile(r'\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]', re.S)


def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", s))


def _operators_to_words(s: str) -> str:
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)                      # 1,000 -> 1000
    s = re.sub(r'\s*==?\s*', ' equals ', s)
    s = re.sub(r'\s*\+\s*', ' plus ', s)
    s = re.sub(r'(?<=[\w)])\s*\*\s*(?=[\w(])', ' times ', s)
    s = re.sub(r'(?<=\d)\s*/\s*(?=\d)', ' over ', s)
    s = re.sub(r'(?<=[\w)])\s+-\s+(?=[\w(])', ' minus ', s)       # spaced binary minus only; hyphens kept
    s = re.sub(r'(?<![\w.])-(?=\d)', 'minus ', s)                 # leading minus of a number
    s = re.sub(r'\s*<=\s*', ' less than or equal to ', s)
    s = re.sub(r'\s*>=\s*', ' greater than or equal to ', s)
    s = re.sub(r'\s+<\s+', ' less than ', s)
    s = re.sub(r'\s+>\s+', ' greater than ', s)
    s = re.sub(r'(?<=\w)\^(\{[^}]*\}|-?\w+)', lambda m: _power_words(m.group(1).strip('{}')), s)
    return s


def _is_math_token(tok: str) -> bool:
    if any(ch in _MATH_CHARS for ch in tok):
        return True
    t = tok.strip('.,;:!?')
    return bool(re.fullmatch(r'[\d.()\-*/]+|[A-Za-z]', t)) and t not in ('a', 'A', 'I')


def _guard_long_runs(text: str, info: dict) -> str:
    """Undelimited formulas: a run of >= 8 consecutive math-looking tokens with >= 2 operator characters whose
    spoken form exceeds LONG_FORMULA_WORDS words is replaced by LONG_FORMULA_PHRASE."""
    toks = text.split(' ')
    out: List[str] = []
    i = 0
    while i < len(toks):
        j = i
        while j < len(toks) and toks[j] and _is_math_token(toks[j]):
            j += 1
        if j - i >= 8:
            span = ' '.join(toks[i:j])
            ops = sum(ch in _MATH_CHARS for ch in span)
            spoken = _operators_to_words(_latex_to_words(span))
            if ops >= 2 and _words(spoken) > LONG_FORMULA_WORDS:
                out.append(LONG_FORMULA_PHRASE)
                info['long_formula'] = info.get('long_formula', 0) + 1
                i = j
                continue
        if j > i:
            out.extend(toks[i:j])
            i = j
        else:
            out.append(toks[i])
            i += 1
    return ' '.join(out)


# ---------------------------------------------------------------------------
# Post-study fix pre-passes: markdown emphasis and prose dashes
# ---------------------------------------------------------------------------

_EMPH_OPEN_LEFT = set('([{"“‘\'')                 # besides start of text and whitespace
_EMPH_OPEN_RIGHT = set('"“‘\'(-')                 # besides letters and digits
_EMPH_CLOSE_LEFT = set('.,;:!?"”’\')-')           # besides letters and digits
_EMPH_CLOSE_RIGHT = set('.,;:!?)]}—–-"”’\'')      # besides end of text and whitespace
_QUOTES = '"“”‘’\''


def _strip_emphasis(s: str) -> str:
    """Remove markdown emphasis asterisks (``*text*``), which the operator step read as "times".

    After v4's removal of ``**``, an *opening* asterisk has start of text, whitespace or an opening
    bracket / quote on its left and a letter, digit, quote, ``(`` or ``-`` on its right; a *closing*
    asterisk has a letter, digit, closing punctuation, quote, ``)`` or ``-`` on its left and end of text,
    whitespace or punctuation on its right. Opening / closing pairs are removed. An unpaired one (emphasis
    split across sentence chunks) is removed only in prose context: an opening one with text after it and
    nothing, a word or prose punctuation before it; a closing one after a word of two or more letters or a
    quote. A math asterisk has no space on either side (``2*3``, ``k*b``) or a space on both (``A² * B``),
    so it is never in either form; an asymmetric one next to a math operand (``(1/2) *kq²``) and star
    notation after a single letter (``a*``, ``A*``) are unpaired and keep their reading."""
    if '*' not in s:
        return s
    n = len(s)
    kinds = {}
    for i, ch in enumerate(s):
        if ch != '*':
            continue
        left = s[i - 1] if i > 0 else ''
        right = s[i + 1] if i + 1 < n else ''
        if (not left or left.isspace() or left in _EMPH_OPEN_LEFT) and right and (right.isalnum() or right in _EMPH_OPEN_RIGHT):
            kinds[i] = 'open'
        elif left and (left.isalnum() or left in _EMPH_CLOSE_LEFT) and (not right or right.isspace() or right in _EMPH_CLOSE_RIGHT):
            kinds[i] = 'close'
    def stray_open(i: int) -> bool:
        # unpaired opening asterisk: text follows, and prose (or nothing) precedes; a math operand before it
        # ("(1/2) *kq²") keeps the asterisk
        if not (s[i + 1].isalpha() or s[i + 1] in _QUOTES):
            return False
        before = s[:i].rstrip()
        if not before:
            return True
        return bool(re.search(r'(?:[A-Za-z]{2,}|[,;:!?—–' + _QUOTES + r'(\[{])$', before))

    def stray_close(i: int) -> bool:
        # unpaired closing asterisk after a word of two or more letters (or a quote); a single letter before it
        # is math star notation ("a*", "A*") and keeps the asterisk
        return bool(re.search(r'(?:[A-Za-z]{2,}|[' + _QUOTES + r'])$', s[:i]))

    drop = set()
    pending = None
    for i in sorted(kinds):
        if kinds[i] == 'open':
            if pending is not None and stray_open(pending):
                drop.add(pending)              # an earlier unpaired opening asterisk
            pending = i
        elif pending is not None:
            drop.update((pending, i))
            pending = None
        elif stray_close(i):
            drop.add(i)                        # unpaired closing asterisk
    if pending is not None and stray_open(pending):
        drop.add(pending)
    if not drop:
        return s
    return ''.join(ch for i, ch in enumerate(s) if i not in drop)


_DASH_KEEP = '\ue001'          # placeholder for a prose dash; restored as '-' (v4 maps an en dash to '-')
_SPACED_DASH = re.compile(r'(?<=\S)(\s+)([-–])(\s+)(?=\S)')
_MATH_WORDS = set(g.lower() for g in _GREEK) | {
    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'log', 'ln',
    'exp', 'sqrt', 'max', 'min', 'det', 'lim', 'mod', 'gcd', 'lcm', 'dx', 'dy', 'dz', 'dt', 'du', 'dv', 'dr', 'ds'}
_MATH_OPERAND = re.compile(r'^[(\[]*(?:\d[\d.,]*|[A-Za-z])[)\]]*[,.;:]?$')


def _protect_prose_dashes(s: str) -> str:
    """Hide a spaced dash between words from the operator step, which read it as "minus".

    A spaced en dash (``Cinema – Cruel``, ``Shine – 2024``) is prose unless both neighbours are numbers or
    single letters (``5 – 3``, ``x – y`` keep "minus"). A spaced hyphen-minus is prose only between two words
    of at least two letters that are not math words (``Radio - Study``); single-letter operands, digits and
    brackets (``n - R``, ``3y - 12``, ``S(tau) - S(rho)``) keep "minus". The dash is restored as ``-``, the
    character v4 already left in place for a dash it did not convert."""
    if '-' not in s and '–' not in s:
        return s

    def neighbours(m: re.Match):
        left = s[:m.start()].rsplit(None, 1)[-1] if s[:m.start()].strip() else ''
        right = s[m.end():].split(None, 1)[0] if s[m.end():].strip() else ''
        return left, right

    def sub(m: re.Match) -> str:
        left, right = neighbours(m)
        lt, rt = left.strip(_QUOTES + '*'), right.strip(_QUOTES + '*')
        if m.group(2) == '–':
            prose = not (_MATH_OPERAND.match(lt) and _MATH_OPERAND.match(rt))
        else:
            lw = re.search(r'(?:^|[^A-Za-z0-9_^])([A-Za-z]{2,})[' + _QUOTES + r'*]*$', left)
            rw = re.match(r'^[' + _QUOTES + r'*]*([A-Za-z]{2,})(?![A-Za-z0-9_(^])', right)
            prose = bool(lw and rw and lw.group(1).lower() not in _MATH_WORDS and rw.group(1).lower() not in _MATH_WORDS)
        return m.group(1) + _DASH_KEEP + m.group(3) if prose else m.group(0)

    return _SPACED_DASH.sub(sub, s)


def spoken(text: str, info: dict = None) -> str:
    """Spoken form of one chunk (see module docstring). ``info`` (optional) receives counters."""
    info = {} if info is None else info
    try:
        s = text
        # markdown
        s = re.sub(r'\*\*|__|`+', '', s)
        s = re.sub(r'(?m)^\s*#{1,6}\s*', '', s)
        s = re.sub(r'(?m)^\s*[-*•]\s+', '', s)
        s = s.replace('|', ' ')
        s = _strip_emphasis(s)                 # post-study fix
        s = _protect_prose_dashes(s)           # post-study fix

        def span_sub(m: re.Match) -> str:
            body = next(g for g in m.groups() if g is not None)
            words = _operators_to_words(_latex_to_words(body))
            info['math_spans'] = info.get('math_spans', 0) + 1
            if _words(words) > LONG_FORMULA_WORDS:
                info['long_formula'] = info.get('long_formula', 0) + 1
                return f' {LONG_FORMULA_PHRASE} '
            return f' {words} '

        s = _MATH_SPAN.sub(span_sub, s)
        s = _guard_long_runs(s, info)
        if '\\' in s or '^' in s or '_' in s:
            s = _latex_to_words(s)
        for k, v in _UNICODE.items():
            if k in s:
                s = s.replace(k, v)
        s = _operators_to_words(s)
        s = re.sub(r'\s+([,.;:!?])', r'\1', s)
        s = re.sub(r'[ \t]+', ' ', s).strip()
        s = s.replace(_DASH_KEEP, '-')         # post-study fix
        if s != text:
            info['changed'] = True
        return s or text
    except Exception as exc:  # never break playback
        info['error'] = f'{type(exc).__name__}: {exc}'
        return text
