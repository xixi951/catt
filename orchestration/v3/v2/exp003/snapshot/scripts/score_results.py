"""§8.6 post-hoc scoring: extract model answers + compare against ground truth.

Reads a per-turn CSV (from ``sweep_router_threshold.py`` or
``baseline_direct_llm.py``) plus the workspace JSONL logs and the
fixture, extracts each row's *model answer* (the model's actual
choice / numerical result), compares it to the fixture's ground truth,
and writes a scored CSV with three new columns:

* ``correct``: 1 if model's answer matches ground truth, 0 if it does
  not, "" if the row could not be scored (no answer extractable, log
  missing, or fixture entry missing).
* ``extracted_answer``: the canonical answer string the scorer pulled
  from the model's text — letter for MMLU, ``\\boxed{...}`` payload for
  MATH-Hard.
* ``score_method``: which extraction path produced the answer (e.g.,
  ``mmlu/regex_letter``, ``math_hard/sympy_eq``, ``unscorable``).

This is a **post-hoc** tool — it does not call any LLM. Run it after a
sweep finishes to add an ``accuracy`` dimension to the latency CSVs.
The driver scripts intentionally do NOT score inline so the experiment
runs are not coupled to a specific evaluation strategy; we can iterate
the scoring rules (regex tweaks, switch to LLM-judge, etc.) without
re-running DashScope.

Usage::

    PYTHONPATH=src python scripts/score_results.py \\
        --source-type router  \\
        --per-turn-csv router_full_per_turn.csv \\
        --logs-dir logs/router-full \\
        --fixture scripts/data/router_sweep_questions.jsonl \\
        --output-csv router_full_per_turn_scored.csv

    PYTHONPATH=src python scripts/score_results.py \\
        --source-type baseline \\
        --per-turn-csv baseline_direct_per_turn.csv \\
        --logs-dir logs/baseline-direct \\
        --fixture scripts/data/router_sweep_questions.jsonl \\
        --output-csv baseline_direct_per_turn_scored.csv

The ``--source-type`` flag picks the JSONL parser (router-sweep
workspace logs use ``talker_segment.text`` to assemble the model
answer; baseline logs use ``baseline_turn_metrics.model_output``
directly).

Scoring methods:

* **MMLU** (multiple-choice): regex hierarchy over the model output,
  documented in ``score_mmlu``. The hierarchy is
  ``answer_is`` → ``choose`` → ``paren`` → ``standalone``; for each
  pattern, the LAST match wins so self-correction phrasings like
  "I first thought B but the answer is C" extract C, and
  parenthesized reasoning like "(A) wrong, (B) wrong, so (C)" picks
  C. The first pattern that produces any match wins; if no pattern
  fires, the row is marked ``unscorable`` rather than guessing.

* **Spoken-MQA** (free-form numeric answer): hierarchy is
  ``gsm8k_trigger`` (``#### N``) → ``boxed`` → ``answer_is`` →
  ``standalone``, documented in ``score_spoken_mqa``. The first
  three are "trigger" patterns mirroring the official Spoken-MQA
  evaluator (AudioBench's ``answer_clean``), which splits on
  ``"The answer is:"`` and ``"####"`` markers; ``standalone`` is
  the regex fallback for outputs that do not include any trigger.
  Each pattern extracts a numeric token, then compares against the
  fixture's ground-truth number with ``math.isclose`` (rel_tol=1e-6,
  abs_tol=1e-6) so trailing-zero or float-rounding differences
  (``"731"`` vs ``"731.0"`` vs ``"731.00"``) all match. Last-match
  wins inside each pattern so the scorer reads the model's *final*
  answer rather than intermediate work.

* **MATH-Hard** (free-form math): extract the last ``\\boxed{...}``
  payload from both the model output and the fixture's ground-truth
  solution. Try cheap normalized-string equality first (catches the
  common case where both sides write the answer the same way without
  paying SymPy's import cost). On mismatch, try SymPy equivalence
  (``sympy.simplify(a - b) == 0``) so ``1/2`` and ``0.5`` and
  ``\\frac{1}{2}`` all match. ImportError or parse error falls back
  to a marker method (``string_eq_fallback_no_sympy`` or
  ``string_eq_fallback_parse_error``) so a no-SymPy environment
  still produces a usable score. Rows whose model output has no
  ``\\boxed`` expression are ``unscorable``.

  MATH-Hard scoring is retained as a code path even though the
  current §8.6 fixture does not use it — the scorer must remain
  back-compatible with prior runs whose CSV/log captures still
  carry ``source: math_hard`` rows.

The scoring is best-effort — it will miss cases where the model gives
the right answer in an unusual form. The summary at the end of the
run prints how many rows scored vs. unscorable so the noise floor is
visible.

A separate ``--summary-csv`` (default: same path as ``--per-turn-csv``
with ``_scored_summary`` suffix) aggregates per-cell accuracy
alongside the per-cell latency aggregates that already live in the
sweep's summary CSV. It also reports ``n_scorable`` and ``n_correct``
so the denominator is explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("score_results")


# ----- fixture loading -----


@dataclass(frozen=True)
class FixtureEntry:
    """One row from the JSONL fixture, indexed for fast lookup by id."""
    id: str
    source: str  # "mmlu" | "spoken_mqa" | "math_hard" (legacy)
    subject: str
    question: str
    answer: str
    difficulty_hint: str


def load_fixture_index(path: Path) -> Dict[str, FixtureEntry]:
    """Read the JSONL fixture into a ``{question_id: FixtureEntry}``
    dict. Raises ValueError on malformed JSON or missing required
    fields.
    """
    required = ("id", "source", "subject", "question", "answer", "difficulty_hint")
    entries: Dict[str, FixtureEntry] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"fixture row {path}:{line_number} must be an object")
            values: Dict[str, str] = {}
            for key in required:
                value = payload.get(key)
                if not isinstance(value, str):
                    raise ValueError(
                        f"fixture row {path}:{line_number} field {key!r} must be str"
                    )
                values[key] = value
            if values["id"] in entries:
                raise ValueError(f"duplicate fixture id at {path}:{line_number}: {values['id']}")
            entries[values["id"]] = FixtureEntry(**values)
    return entries


# ----- model-output recovery -----


def model_output_from_router_log(log_path: Path, turn_id: str) -> Optional[str]:
    """Scan a router-sweep workspace JSONL log and concatenate every
    ``talker_segment.text`` event for ``turn_id`` in seq order. Returns
    None if the turn has no talker_segment events at all (malformed or
    crashed turn).

    The router sweep emits one ``talker_segment`` event per dispatch
    (hard_first / hard_mid / hard_final on hard path; one segment for
    easy path). Concatenated text is the full spoken answer.
    """
    segments: List[Tuple[int, str]] = []
    with Path(log_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSONL at {log_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"malformed JSONL at {log_path}:{line_number}: event is not an object")
            kind = event.get("kind")
            data = event.get("data")
            if kind != "talker_segment" or not isinstance(data, dict):
                continue
            if data.get("turn_id") != turn_id:
                continue
            seq = data.get("seq")
            text = data.get("text")
            if isinstance(seq, int) and isinstance(text, str):
                segments.append((seq, text))
    if not segments:
        return None
    return "".join(text for _seq, text in sorted(segments, key=lambda item: item[0]))


def model_output_from_baseline_log(log_path: Path, turn_id: str) -> Optional[str]:
    """Scan a baseline JSONL log for the ``baseline_turn_metrics`` event
    matching ``turn_id`` and return its ``model_output`` field.
    Returns None if the turn errored (only a ``baseline_turn_error``
    event was written) or the event is missing.
    """
    with Path(log_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed JSONL at {log_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"malformed JSONL at {log_path}:{line_number}: event is not an object")
            kind = event.get("kind")
            data = event.get("data")
            if not isinstance(data, dict) or data.get("turn_id") != turn_id:
                continue
            if kind == "baseline_turn_error":
                return None
            if kind == "baseline_turn_metrics":
                model_output = data.get("model_output")
                return model_output if isinstance(model_output, str) else None
    return None


# ----- scoring per source -----


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one row."""
    correct: Optional[bool]   # True / False / None (unscorable)
    extracted_answer: str
    score_method: str


# Regex hierarchy for MMLU letter extraction. Each pattern in the
# ordered list is tried; for every pattern the LAST match wins (round-2
# review M2: a model output that revises itself, e.g. "I first thought
# B but actually the answer is C", must extract C, not B; same for
# parenthesized reasoning "(A) is wrong, (B) is wrong, so (C) is
# correct"). Patterns are case-insensitive; the first capture group
# holds the letter, normalized to uppercase by the scorer.
_MMLU_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "answer_is",
        re.compile(
            r"\b(?:final\s+answer|correct\s+answer|the\s+answer|answer)"
            r"(?:\s+is|:)\s*\(?([A-J])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "choose",
        re.compile(
            r"\b(?:choose|select|pick|going\s+with|option)\s*\(?([A-J])\b",
            re.IGNORECASE,
        ),
    ),
    ("paren", re.compile(r"\(([A-J])\)", re.IGNORECASE)),
    ("standalone", re.compile(r"\b([A-J])\b", re.IGNORECASE)),
)


def score_mmlu(model_output: str, ground_truth_letter: str) -> ScoreResult:
    """Score a single MCQ answer (MMLU vanilla A-D **or** MMLU-Pro A-J).

    For each pattern in ``_MMLU_PATTERNS``, run ``re.findall`` and
    take the LAST match. First pattern that produces a match wins.
    If no pattern fires, returns ``unscorable``.

    R20: extended the regex's letter range from ``[A-D]`` to
    ``[A-J]`` so MMLU-Pro's 10-option questions score the same way.
    Vanilla MMLU rows (4 options) are unaffected — the model never
    emits E-J on a 4-option prompt.

    Last-match-wins handles two common phrasings the regex
    documentation calls out (round-2 review M2):

    * Self-correction: "I first thought B, but the answer is C"
      → ``answer_is`` last-match returns C (would return B if we
      took first match).
    * Parenthesized reasoning: "(A) is wrong, (B) is wrong, so (C)"
      → ``paren`` last-match returns C.
    """
    for name, pattern in _MMLU_PATTERNS:
        hits = pattern.findall(model_output)
        if hits:
            extracted = hits[-1].upper()
            correct = extracted == ground_truth_letter.strip().upper()
            return ScoreResult(
                correct=correct,
                extracted_answer=extracted,
                score_method=f"mmlu/{name}",
            )
    return ScoreResult(
        correct=None,
        extracted_answer="",
        score_method="unscorable/no_letter",
    )


# Spoken-MQA scoring: the model output is free-form prose ending in (or
# containing) a numerical answer. Try patterns in order; for each
# pattern, the LAST match wins so the model's *final* number, not an
# intermediate working value, becomes the extracted answer.
_NUMBER_TOKEN = r"-?\d+(?:\.\d+)?"
_SPOKEN_MQA_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        # GSM8K convention: the model copies the dataset's
        # "#### <final answer>" marker at the end of its CoT trace.
        # Highest-priority trigger because it is unambiguous — when a
        # model emits "####", whatever follows is definitionally the
        # final numerical answer. Mirrors AudioBench's
        # ``get_seperation_trigger("gsm8k") == "####"``.
        "gsm8k_trigger",
        re.compile(rf"####\s*\$?({_NUMBER_TOKEN})"),
    ),
    (
        # Strong cue phrases. Examples that should match (last
        # capture group is the number):
        #   "the answer is 731"
        #   "Answer: -4.8"
        #   "= 0.6"
        #   "result: 80.9421"
        #   "equals 79"
        "answer_is",
        re.compile(
            r"(?:final\s+answer|the\s+answer|answer|result|equals?|=)"
            rf"\s*(?:is\s*|to\s*|:\s*)?\$?({_NUMBER_TOKEN})",
            re.IGNORECASE,
        ),
    ),
    (
        # Bare last-number fallback. Picks the final number anywhere
        # in the output. Common for answers like "Joan grew 79
        # pumpkins in all." or just "79".
        "standalone",
        re.compile(rf"({_NUMBER_TOKEN})"),
    ),
)


def _parse_number(token: str) -> Optional[float]:
    """Parse ``token`` to a float; returns None if non-numeric or
    non-finite. Strips a leading ``$`` and any wrapping whitespace
    so currency-style output ("$731") still matches.
    """
    cleaned = token.strip().lstrip("$").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_number_for_record(value: float) -> str:
    """Format a parsed number for ``extracted_answer``. Drop the
    trailing ``.0`` on whole values so ``731.0`` shows up as ``731``,
    matching the fixture's ground-truth string convention.
    """
    if value.is_integer():
        return str(int(value))
    return repr(value)


def score_spoken_mqa(model_output: str, ground_truth: str) -> ScoreResult:
    """Score a single Spoken-MQA answer.

    The fixture's ``answer`` is always a numeric string (e.g. ``"731"``,
    ``"-4.8"``, ``"80.9421"``). The model is free to embed it anywhere
    in prose. We try, in order, mirroring the official Spoken-MQA
    evaluator (AudioBench's ``answer_clean``):

    1. **\\boxed{...}**: thinking models sometimes wrap final answers
       in ``\\boxed`` for emphasis. If present and the contents parse
       as a number, that's the extracted answer.
    2. **gsm8k_trigger** (``#### N``): GSM8K's official final-answer
       marker. Highest priority among text patterns because it is
       unambiguous — anything after ``####`` is definitionally the
       final answer.
    3. **answer_is** pattern: matches "answer is X", "answer: X",
       "result: X", "= X", "equals X", with last-match wins.
    4. **standalone** pattern: last bare number in the output.

    Numeric comparison uses ``math.isclose(rel_tol=1e-6, abs_tol=1e-6)``
    so ``"731"``, ``"731.0"``, ``"731.00"``, and float-rounded
    intermediates all compare equal. A ground-truth string that
    fails to parse as a finite float is treated as a fixture
    corruption: returns ``unscorable/bad_ground_truth`` (the row is
    excluded from accuracy denominators rather than silently
    misjudged).
    """
    gt_value = _parse_number(ground_truth)
    if gt_value is None:
        return ScoreResult(
            correct=None,
            extracted_answer="",
            score_method="unscorable/bad_ground_truth",
        )

    boxed = _extract_last_boxed(model_output)
    if boxed is not None:
        boxed_value = _parse_number(boxed)
        if boxed_value is not None:
            return ScoreResult(
                correct=math.isclose(boxed_value, gt_value, rel_tol=1e-6, abs_tol=1e-6),
                extracted_answer=_format_number_for_record(boxed_value),
                score_method="spoken_mqa/boxed",
            )

    for name, pattern in _SPOKEN_MQA_PATTERNS:
        hits = pattern.findall(model_output)
        for token in reversed(hits):
            extracted = _parse_number(token)
            if extracted is None:
                continue
            return ScoreResult(
                correct=math.isclose(extracted, gt_value, rel_tol=1e-6, abs_tol=1e-6),
                extracted_answer=_format_number_for_record(extracted),
                score_method=f"spoken_mqa/{name}",
            )

    return ScoreResult(
        correct=None,
        extracted_answer="",
        score_method="unscorable/no_number",
    )


_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _extract_last_boxed(text: str) -> Optional[str]:
    """Return the contents of the LAST ``\\boxed{...}`` in ``text``,
    or None if absent. Tolerates one level of nested braces (e.g.,
    ``\\boxed{\\frac{1}{2}}``) but not deeper nesting; the regex is
    documented as a known approximation in the docstring.
    """
    matches = _BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_math_answer(text: str) -> str:
    """Cheap canonical form for MATH-Hard / Omni-MATH string comparison.

    Strips, in this order:
    * Leading / trailing ``$`` (LaTeX inline math markers — Omni-MATH
      ground truths often arrive as ``$\\text{No}$`` or
      ``$2^{n+1}-2$``).
    * Up to a few iterations of ``\\boxed{...}`` and ``\\text{...}``
      wrappers (R20: Omni-MATH GT can nest both, e.g.
      ``$\\boxed{\\text{No}}$``). Iterating handles 1-3 levels which
      covers all observed Omni-MATH formats.
    * All whitespace, all backslashes; then lowercases.

    This is NOT a substitute for SymPy — purely a fast first-pass
    for the common case where both sides write the answer the same
    way modulo formatting noise.
    """
    stripped = text.strip()
    # Strip ``$ ... $`` inline math markers (paired only).
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        stripped = stripped[1:-1].strip()
    for _ in range(3):
        progress = False
        for prefix in ("\\boxed{", "\\text{"):
            if stripped.startswith(prefix) and stripped.endswith("}"):
                stripped = stripped[len(prefix):-1].strip()
                progress = True
                break
        if not progress:
            break
    stripped = re.sub(r"\s+", "", stripped)
    stripped = stripped.replace("\\", "")
    return stripped.lower()


def score_math_hard(model_output: str, ground_truth_solution: str) -> ScoreResult:
    """Score a MATH-Hard answer.

    Steps (round-2 review M3 reordered + ImportError fallback):

    1. Extract last ``\\boxed{...}`` payload from both
       ``model_output`` and ``ground_truth_solution`` (the fixture's
       ``answer`` field is the full solution text including the
       boxed answer).
    2. If either is missing, return ``unscorable``.
    3. **Cheap first**: compare normalized strings (
       ``_normalize_math_answer``). Identical → ``correct=True`` with
       method ``math_hard/string_eq``. This is the common case (both
       sides emit ``6`` or ``\\frac{1}{2}``) and avoids SymPy's
       ``antlr4`` dependency.
    4. **SymPy second**: on string mismatch, try
       ``sympy.simplify(parse(model) - parse(gt)) == 0`` to catch
       semantic equivalences (``1/2`` vs ``0.5`` vs ``\\frac{1}{2}``,
       ``\\sqrt{2}`` vs ``2^{1/2}``, etc.). Method
       ``math_hard/sympy_eq``.
    5. **Fallback**: on ``ImportError`` (sympy or its parser missing),
       return the round-3 string-comparison result with method
       ``math_hard/string_eq_fallback_no_sympy``. On any parse /
       simplification error, use
       ``math_hard/string_eq_fallback_parse_error`` instead. A
       no-SymPy environment still produces a usable score, just less
       generous on equivalent-but-differently-formatted answers.

    Caveat: comma-separated boxed answers like ``\\boxed{27,93}``
    appear in MATH-Hard ("enter all possible values"). Order can vary
    between solutions; current scoring treats them as ordered. A
    follow-up could split on comma + sort before comparing — out of
    scope for this round.
    """
    model_box = _extract_last_boxed(model_output)
    gt_box = _extract_last_boxed(ground_truth_solution)
    if model_box is None or gt_box is None:
        return ScoreResult(
            correct=None,
            extracted_answer=model_box or "",
            score_method="math_hard/unscorable_no_boxed",
        )
    if _normalize_math_answer(model_box) == _normalize_math_answer(gt_box):
        return ScoreResult(
            correct=True,
            extracted_answer=model_box,
            score_method="math_hard/string_eq",
        )
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        diff = sympy.simplify(parse_latex(model_box) - parse_latex(gt_box))
        eq = diff == 0
        return ScoreResult(
            correct=bool(eq),
            extracted_answer=model_box,
            score_method="math_hard/sympy_eq",
        )
    except ImportError:
        return ScoreResult(
            correct=False,
            extracted_answer=model_box,
            score_method="math_hard/string_eq_fallback_no_sympy",
        )
    except Exception:
        return ScoreResult(
            correct=False,
            extracted_answer=model_box,
            score_method="math_hard/string_eq_fallback_parse_error",
        )


_MASSIVE_INTENT_PATTERN = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b")


# AIME prose-answer fallback patterns. The system pipeline's TTS-bound
# talker is told (talker_hard_final.md) to "Do not say the literal
# token \\boxed; just state the integer", so system-pipeline AIME
# outputs reach the scorer in conversational form like "The final
# answer is 277." or "答案是 65。" The baselines preserve the
# user-prompt instruction "Put your answer inside \\boxed{}." and
# DO emit a boxed integer; the scorer must accept both shapes so
# baseline and system are compared on equal footing.
#
# Order is significant: most-specific (explicit "answer is" / "答案
# 是") first, then m+n / final number cues, then the very loose
# "last 0-999 integer in the last paragraph" backup. Each pattern
# captures the integer in group 1.
_AIME_BOLD_PATTERN = re.compile(r"\*\*\s*(\d{1,3})\s*\*\*")
_AIME_PROSE_PATTERNS = [
    re.compile(
        r"(?:final\s+answer\s+is|answer\s+is|the\s+answer\s+is|"
        r"is\s+\**|equal[s]?\s+|=\s*\**)"
        r"\s*(\d{1,3})\b",
        re.IGNORECASE,
    ),
    re.compile(r"答案\s*[:是为]\s*(\d{1,3})\b"),
    re.compile(r"最终答案\s*[:是为]\s*(\d{1,3})\b"),
]


def _extract_aime_integer(text: str) -> Optional[str]:
    """Try a sequence of AIME-specific extractors. Returns the integer
    string (e.g. ``"277"``) on first match, else None. Used when the
    boxed extractor fails — system-pipeline AIME outputs typically
    have no \\boxed{} since talker_hard_final.md forbids the literal
    token in the TTS-bound speech.

    Extraction priority (most specific first):

    1. **Last** ``\\boxed{N}`` payload (baselines + any system rerun
       that re-includes boxed).
    2. **Last** ``**N**`` markdown bold integer (the talker often
       emphasizes the final answer this way: "is **62**.").
    3. **Last sentence** scan: ``\\d{1,3}`` integer that appears
       inside an "is N" / "answer is N" / "= N" pattern. Limit to
       the last 1-2 sentences so we do NOT pick up intermediate
       values like "added 40 permutations" buried mid-paragraph.
    4. **Last 1-3 digit integer in the last sentence overall**
       (covers Chinese "答案是 65" and similar bare statements).
    5. **Last 1-3 digit integer anywhere** as a final fallback.
    """
    if not text:
        return None
    boxed = _extract_last_boxed(text)
    if boxed:
        digits = re.search(r"\d{1,3}", boxed)
        if digits:
            return digits.group(0)

    # Priority 2: last **N** markdown bold integer.
    bold_matches = list(_AIME_BOLD_PATTERN.finditer(text))
    if bold_matches:
        return bold_matches[-1].group(1)

    # Priority 3: prose patterns (answer is / is / =) restricted to the
    # last 2 sentences so we do not catch intermediate cues mid-text.
    sentences = [s for s in re.split(r"(?<=[.。!?！？])\s+", text) if s.strip()]
    for sent in reversed(sentences[-3:]):
        for pat in _AIME_PROSE_PATTERNS:
            ms = list(pat.finditer(sent))
            if ms:
                return ms[-1].group(1)

    # Priority 4: very last integer in the last sentence (handles
    # "… and the answer 132." plus all-Chinese "答案是 65。").
    if sentences:
        ints_last = re.findall(r"\b(\d{1,3})\b", sentences[-1])
        if ints_last:
            return ints_last[-1]

    # Priority 5: last 1-3 digit integer anywhere.
    ints_all = re.findall(r"\b(\d{1,3})\b", text)
    return ints_all[-1] if ints_all else None


def score_aime_2026(model_output: str, ground_truth: str) -> ScoreResult:
    """Score an AIME 2026 answer with both boxed and prose fallbacks.

    Ground truth is a bare integer in [0, 999]. The model output may
    arrive in two shapes:

    * **Baseline** (``baseline_direct_llm.py``): user prompt asks for
      ``\\boxed{N}``; model emits ``\\boxed{277}`` -> we extract it.
    * **System pipeline** (talker hard_final): TTS-friendly prose
      like ``"The final answer is 277."`` or ``"答案是 65"`` —
      ``_extract_aime_integer`` walks a hierarchy (boxed → "answer
      is N" / "答案是 N" / sum-equals-N → very last integer).

    Returns ``correct=True`` iff the extracted integer matches GT.
    """
    extracted = _extract_aime_integer(model_output)
    if extracted is None:
        return ScoreResult(
            correct=None, extracted_answer="",
            score_method="aime_2026/unscorable_no_integer",
        )
    gt_clean = (ground_truth or "").strip().lstrip("0") or "0"
    ext_clean = extracted.lstrip("0") or "0"
    method = (
        "aime_2026/boxed" if "\\boxed{" in (model_output or "")
        and re.search(r"\\boxed\{[^}]*\d", model_output or "")
        else "aime_2026/prose"
    )
    return ScoreResult(
        correct=ext_clean == gt_clean,
        extracted_answer=extracted,
        score_method=method,
    )


def score_omni_math(model_output: str, ground_truth: str) -> ScoreResult:
    """Score one Omni-MATH answer.

    Omni-MATH ground truths are bare strings (often a number or short
    LaTeX expression — not wrapped in \\boxed{} the way MATH-Hard's
    full-solution column is). The model is prompted to wrap its final
    answer in \\boxed{...}; we extract that payload and compare it to
    the bare ground truth via:

    1. Cheap normalised-string equality (drops backslashes,
       whitespace, surrounding ``\\boxed`` / ``\\text`` wrappers,
       lower-cases).
    2. SymPy ``simplify(parse(model) - parse(gt)) == 0`` on string
       mismatch — catches ``1/2`` vs. ``0.5`` vs. ``\\frac{1}{2}``.
    3. ImportError → ``omni_math/string_eq_fallback_no_sympy`` (the
       SymPy LaTeX parser pulls in ``antlr4``; tests run without).
    4. Any parse / simplify exception → fallback marker.

    Rows whose model output has no ``\\boxed{...}`` are unscorable
    rather than guessed at.
    """
    model_box = _extract_last_boxed(model_output)
    if model_box is None:
        return ScoreResult(
            correct=None,
            extracted_answer="",
            score_method="omni_math/unscorable_no_boxed",
        )
    gt = ground_truth.strip()
    if _normalize_math_answer(model_box) == _normalize_math_answer(gt):
        return ScoreResult(
            correct=True,
            extracted_answer=model_box,
            score_method="omni_math/string_eq",
        )
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        diff = sympy.simplify(parse_latex(model_box) - parse_latex(gt))
        return ScoreResult(
            correct=bool(diff == 0),
            extracted_answer=model_box,
            score_method="omni_math/sympy_eq",
        )
    except ImportError:
        return ScoreResult(
            correct=False,
            extracted_answer=model_box,
            score_method="omni_math/string_eq_fallback_no_sympy",
        )
    except Exception:
        return ScoreResult(
            correct=False,
            extracted_answer=model_box,
            score_method="omni_math/string_eq_fallback_parse_error",
        )


def score_massive(model_output: str, ground_truth_intent: str) -> ScoreResult:
    """Score a MASSIVE intent-classification answer.

    The official MASSIVE intent-classification metric is exact-match
    on the intent label string (e.g., ``"alarm_set"``). Our zero-shot
    LLM prompt asks the model to emit the intent and nothing else,
    but well-mannered LLMs tend to wrap the answer in
    extra prose ("The intent is: alarm_set."), so the scorer:

    1. Strips whitespace and trailing punctuation from the raw
       output. If the cleaned string equals the ground truth, that
       is a clean exact match (``massive/exact``).
    2. Otherwise scans for ``snake_case`` tokens (intents are all
       snake_case alphanumeric) and takes the LAST one — handles
       ``"... so the intent is alarm_set"`` and rejects ``"The"``
       prose tokens (which start with a capital, not lowercase).
       If the chosen token equals the ground truth, that is a
       lenient match (``massive/last_token``).
    3. Otherwise, returns unscorable with method
       ``unscorable/no_intent_token``.

    Comparison is case-sensitive because the dataset's intent names
    are lowercase by construction. A model emitting ``"Alarm_Set"``
    is treated as wrong; this is consistent with the official
    intent-classification eval and avoids accepting hallucinated
    intent variants.
    """
    cleaned = model_output.strip().rstrip(".!?,;:\"'`")
    target = ground_truth_intent.strip()
    if cleaned == target:
        return ScoreResult(correct=True, extracted_answer=cleaned,
                           score_method="massive/exact")
    hits = _MASSIVE_INTENT_PATTERN.findall(model_output)
    if hits:
        extracted = hits[-1]
        return ScoreResult(
            correct=extracted == target,
            extracted_answer=extracted,
            score_method="massive/last_token",
        )
    return ScoreResult(correct=None, extracted_answer="",
                       score_method="unscorable/no_intent_token")


def _score_with_optional_judge(
    fixture_entry: "FixtureEntry",
    model_output: str,
    *,
    use_omni_judge: bool,
    use_spoken_judge: bool,
    judge_model: str,
) -> "ScoreResult":
    """Dispatch to either the deterministic scorers (default) or one of
    the two LLM-as-judge wrappers in ``scripts/omni_judge.py``:

    * ``--omni-judge`` routes ``omni_math`` rows through ``judge_omni``
      (Omni-MATH GPT-eval template).
    * ``--spoken-judge`` routes ``spoken_mqa`` rows through
      ``judge_spoken_mqa`` (arithmetic-word-problem template tailored
      to conversational tutor responses).

    Both flags can be set independently. Other sources keep their
    deterministic scorers regardless. ``score_method`` reports
    ``<source>/judge`` so downstream analysis can tell judge rows
    apart from regex rows.
    """
    # R23 fixture introduces ``gsm_hard`` and ``math`` as separate
    # source labels for the GSM-Hard and Hendrycks MATH (levels 1-3)
    # pulls. They share the underlying spoken/omni judges with
    # spoken_mqa / omni_math respectively because the ground-truth
    # shape (numeric vs LaTeX) maps cleanly:
    #   spoken_mqa  + gsm_hard  → spoken-judge (numeric / short closed-form)
    #   omni_math   + math      → omni-judge   (LaTeX / boxed expressions)
    if fixture_entry.source in ("omni_math", "math") and use_omni_judge:
        return _judge_dispatch(
            fixture_entry, model_output, judge_model, source=fixture_entry.source
        )
    if fixture_entry.source in ("spoken_mqa", "gsm_hard") and use_spoken_judge:
        return _judge_dispatch(
            fixture_entry, model_output, judge_model, source=fixture_entry.source
        )
    return score_row(fixture_entry, model_output)


def _judge_dispatch(
    fixture_entry: "FixtureEntry",
    model_output: str,
    judge_model: str,
    *,
    source: str,
) -> "ScoreResult":
    """Shared judge invocation path. Imports the right judge, calls it,
    maps the JudgeResult onto a ScoreResult with a source-tagged
    ``score_method``. Lazy import so score_results stays importable
    on machines without the dashscope SDK.
    """
    try:
        from omni_judge import (  # type: ignore[import-not-found]
            judge_omni,
            judge_spoken_mqa,
        )
    except ImportError:
        from scripts.omni_judge import (  # type: ignore[import-not-found]
            judge_omni,
            judge_spoken_mqa,
        )

    # math + omni_math share judge_omni (LaTeX-aware); gsm_hard +
    # spoken_mqa share judge_spoken_mqa (numeric / closed-form).
    judge_fn = judge_omni if source in ("omni_math", "math") else judge_spoken_mqa
    result = judge_fn(
        problem=fixture_entry.question,
        reference=fixture_entry.answer,
        student=model_output,
        model=judge_model,
    )
    if result.error:
        return ScoreResult(
            correct=None,
            extracted_answer=result.extracted_answer,
            score_method=f"{source}/judge_error/{result.error}",
        )
    return ScoreResult(
        correct=result.correct,
        extracted_answer=result.extracted_answer,
        score_method=f"{source}/judge",
    )


def score_row(fixture_entry: FixtureEntry, model_output: str) -> ScoreResult:
    """Dispatch to the right scorer based on ``fixture_entry.source``.
    Empty / None ``model_output`` returns unscorable.
    """
    if not model_output:
        return ScoreResult(correct=None, extracted_answer="",
                           score_method="unscorable/empty_output")
    if fixture_entry.source in ("mmlu", "mmlu_pro"):
        # Same MCQ scorer; the regex covers A-J so MMLU-Pro's
        # 10 options are handled identically to MMLU's 4.
        return score_mmlu(model_output, fixture_entry.answer)
    if fixture_entry.source == "spoken_mqa":
        return score_spoken_mqa(model_output, fixture_entry.answer)
    if fixture_entry.source == "massive":
        return score_massive(model_output, fixture_entry.answer)
    if fixture_entry.source == "omni_math":
        return score_omni_math(model_output, fixture_entry.answer)
    if fixture_entry.source == "gsm_hard":
        # Same scoring shape as Spoken-MQA: numeric ground truth, GSM-style
        # output. The four-level extractor (boxed → #### → answer_is →
        # standalone) still applies.
        return score_spoken_mqa(model_output, fixture_entry.answer)
    if fixture_entry.source == "math":
        # Hendrycks MATH ground truth comes wrapped in ``\\boxed{...}``
        # already; the math_hard scorer handles boxed-string equivalence
        # plus a SymPy-equivalence fallback for things like
        # ``1/2`` vs ``\\frac{1}{2}``.
        return score_math_hard(model_output, fixture_entry.answer)
    if fixture_entry.source == "math_hard":
        return score_math_hard(model_output, fixture_entry.answer)
    if fixture_entry.source == "aime_2026":
        # AIME 2026 ground truths are bare integers in [0, 999]. The
        # ``score_aime_2026`` extractor accepts both shapes:
        # baselines emit ``\\boxed{277}`` (talker prompt asks for
        # boxed); system pipeline talker_hard_final.md forbids the
        # literal token so its outputs are TTS prose like "the final
        # answer is 277" / "答案是 65" / "m + n = 277".
        return score_aime_2026(model_output, fixture_entry.answer)
    if fixture_entry.source == "gpqa_diamond":
        # GPQA Diamond is 4-option MCQ (A-D); reuse the MMLU scorer
        # which already handles A-J via the four-level regex
        # hierarchy (answer_is → choose → paren → standalone).
        return score_mmlu(model_output, fixture_entry.answer)
    if fixture_entry.source == "voicebench_mmsu":
        # VoiceBench MMSU subset is 4-option MCQ (A-D); same scorer
        # as MMLU.
        return score_mmlu(model_output, fixture_entry.answer)
    if fixture_entry.source == "voicebench_alpaca_eval":
        # Open-ended chat. VoiceBench official protocol uses GPT-4
        # as a 1-5 judge; we use DashScope qwen-max via a separate
        # offline pass (scripts/score_alpaca_judge.py) and pre-fill
        # ``correct`` / ``extracted_answer`` columns. The deterministic
        # path here returns ``unscorable/needs_alpaca_judge`` so the
        # downstream aggregator knows to either require the judge or
        # exclude this source from accuracy_mean.
        return ScoreResult(
            correct=None,
            extracted_answer="",
            score_method="unscorable/needs_alpaca_judge",
        )
    return ScoreResult(correct=None, extracted_answer="",
                       score_method=f"unscorable/unknown_source/{fixture_entry.source}")


# ----- per-cell aggregation -----


@dataclass(frozen=True)
class CellAccuracy:
    """One row of the scored summary CSV (per cell = per threshold or
    per mode, depending on source)."""
    cell_key: str  # threshold value (router) or mode name (baseline)
    n_questions: int
    n_scorable: int
    n_correct: int
    accuracy: float  # n_correct / n_scorable; NaN if n_scorable == 0


def aggregate_by_cell(
    scored_rows: List[dict],
    cell_key_field: str,
) -> Dict[str, CellAccuracy]:
    """Aggregate scored per-turn rows by cell. ``cell_key_field`` is
    ``"threshold"`` for router-sweep CSVs or ``"mode"`` for baseline.
    """
    grouped: Dict[str, List[dict]] = {}
    for row in scored_rows:
        grouped.setdefault(str(row.get(cell_key_field, "")), []).append(row)

    result: Dict[str, CellAccuracy] = {}
    for cell_key, rows in grouped.items():
        n_scorable = 0
        n_correct = 0
        for row in rows:
            value = row.get("correct")
            if value in (True, "1", 1):
                n_scorable += 1
                n_correct += 1
            elif value in (False, "0", 0):
                n_scorable += 1
        accuracy = n_correct / n_scorable if n_scorable else math.nan
        result[cell_key] = CellAccuracy(
            cell_key=cell_key,
            n_questions=len(rows),
            n_scorable=n_scorable,
            n_correct=n_correct,
            accuracy=accuracy,
        )
    return result


def _question_set_fingerprint(entries: Iterable[FixtureEntry]) -> str:
    import hashlib

    blob = "\n".join(f"{entry.id}\t{entry.question}" for entry in entries).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def _reconstruct_log_path_legacy(
    source_type: str,
    row: dict,
    logs_dir: Path,
    question_set_id: str,
    question_index_by_id: Dict[str, int],
) -> Tuple[Path, str]:
    question_id = row.get("question_id", "")
    turn_index = question_index_by_id.get(question_id)
    if turn_index is None:
        turn_index = 0

    if source_type == "baseline":
        mode = row.get("mode", "")
        dialog_id = f"baseline-direct-{mode}-q{question_set_id}"
    else:
        threshold = float(row.get("threshold", "nan"))
        dialog_id = f"sweep-router-t{threshold:.3f}-q{question_set_id}"
    return Path(logs_dir) / f"{dialog_id}.ws.jsonl", f"{dialog_id}/turn-{turn_index}"


def _format_correct(value: Optional[bool]) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return ""


def _default_summary_path(output_csv: Path) -> Path:
    return output_csv.with_name(f"{output_csv.stem}_scored_summary{output_csv.suffix}")


def _atomic_write_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


SUMMARY_FIELDS: Tuple[str, ...] = (
    "cell_key",
    "n_questions",
    "n_scorable",
    "n_correct",
    "accuracy",
)


# ----- top-level -----


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--source-type", choices=["router", "baseline"], required=True,
        help="picks the JSONL parser; router uses talker_segment.text, "
             "baseline uses baseline_turn_metrics.model_output",
    )
    parser.add_argument(
        "--per-turn-csv", type=Path, required=True,
        help="input per-turn CSV from sweep_router_threshold or baseline_direct_llm",
    )
    parser.add_argument(
        "--logs-dir", type=Path, required=True,
        help="JSONL logs directory matching the per-turn CSV's source",
    )
    parser.add_argument(
        "--fixture", type=Path, required=True,
        help="JSONL fixture used at sweep time (for ground-truth answers)",
    )
    parser.add_argument(
        "--output-csv", type=Path, required=True,
        help="output scored CSV; original columns + correct + extracted_answer + score_method",
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="optional per-cell accuracy summary; default: same as --output-csv "
             "with '_scored_summary' suffix",
    )
    parser.add_argument(
        "--omni-judge",
        action="store_true",
        default=False,
        help=(
            "Replace the strict `\\boxed{}` extractor for omni_math rows "
            "with an LLM-as-judge (DashScope qwen-plus by default). The "
            "judge uses the official Omni-MATH GPT-eval template — see "
            "scripts/omni_judge.py — and accepts semantic equivalence "
            "even when the spoken talker output strips LaTeX. Requires "
            "DASHSCOPE_API_KEY in the environment. Other sources still "
            "use their deterministic scorers."
        ),
    )
    parser.add_argument(
        "--omni-judge-model",
        type=str,
        default="qwen-plus-2025-09-11",
        help=(
            "Judge model id passed to DashScope (default: "
            "qwen-plus-2025-09-11). Used by both --omni-judge and "
            "--spoken-judge."
        ),
    )
    parser.add_argument(
        "--spoken-judge",
        action="store_true",
        default=False,
        help=(
            "Replace the deterministic regex extractor for spoken_mqa "
            "rows with an LLM-as-judge tailored to arithmetic word "
            "problems. The system's talker rewrites GSM-style answers "
            "into conversational prose (often re-stating intermediate "
            "values after the final answer), which trips the "
            "last-number-wins regex; the judge accepts semantic "
            "equivalence regardless of where the answer appears in "
            "the response. Other sources still use their deterministic "
            "scorers."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    fixture_index = load_fixture_index(args.fixture)
    fixture_entries = list(fixture_index.values())
    question_set_id = _question_set_fingerprint(fixture_entries)
    question_index_by_id = {
        entry.id: index for index, entry in enumerate(fixture_entries)
    }

    with args.per_turn_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        input_rows = list(reader)

    score_fields = ["correct", "extracted_answer", "score_method"]
    scored_fields = original_fields + [field for field in score_fields if field not in original_fields]
    scored_rows: List[dict] = []

    for row in input_rows:
        question_id = row.get("question_id", "")
        fixture_entry = fixture_index.get(question_id)
        if fixture_entry is None:
            scored_rows.append(
                {
                    **row,
                    "correct": "",
                    "extracted_answer": "",
                    "score_method": "unscorable/missing_fixture",
                }
            )
            continue

        dialog_id = row.get("dialog_id", "")
        turn_id = row.get("turn_id", "")
        if dialog_id and turn_id:
            log_path = args.logs_dir / f"{dialog_id}.ws.jsonl"
        else:
            log_path, turn_id = _reconstruct_log_path_legacy(
                args.source_type,
                row,
                args.logs_dir,
                question_set_id,
                question_index_by_id,
            )

        if args.source_type == "baseline" and row.get("model_output"):
            model_output = row["model_output"]
        elif not log_path.exists():
            scored_rows.append(
                {
                    **row,
                    "correct": "",
                    "extracted_answer": "",
                    "score_method": "unscorable/missing_log",
                }
            )
            continue
        elif args.source_type == "router":
            model_output = model_output_from_router_log(log_path, turn_id)
        else:
            model_output = model_output_from_baseline_log(log_path, turn_id)

        score = _score_with_optional_judge(
            fixture_entry,
            model_output or "",
            use_omni_judge=args.omni_judge,
            use_spoken_judge=args.spoken_judge,
            judge_model=args.omni_judge_model,
        )
        scored_rows.append(
            {
                **row,
                "correct": _format_correct(score.correct),
                "extracted_answer": score.extracted_answer,
                "score_method": score.score_method,
            }
        )

    _atomic_write_rows(args.output_csv, scored_fields, scored_rows)

    summary_path = args.summary_csv if args.summary_csv is not None else _default_summary_path(args.output_csv)
    cell_key_field = "threshold" if args.source_type == "router" else "mode"
    summary = aggregate_by_cell(scored_rows, cell_key_field)
    summary_rows = [
        {
            "cell_key": accuracy.cell_key,
            "n_questions": str(accuracy.n_questions),
            "n_scorable": str(accuracy.n_scorable),
            "n_correct": str(accuracy.n_correct),
            "accuracy": str(accuracy.accuracy),
        }
        for accuracy in summary.values()
    ]
    _atomic_write_rows(summary_path, list(SUMMARY_FIELDS), summary_rows)

    n_rows = len(scored_rows)
    n_scorable = sum(1 for row in scored_rows if row.get("correct") in ("1", "0"))
    n_correct = sum(1 for row in scored_rows if row.get("correct") == "1")
    n_unscorable = n_rows - n_scorable
    overall_accuracy = n_correct / n_scorable if n_scorable else math.nan
    print(
        "score-results complete: "
        f"rows={n_rows}; scored={n_scorable}; unscorable={n_unscorable}; "
        f"accuracy={overall_accuracy}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
