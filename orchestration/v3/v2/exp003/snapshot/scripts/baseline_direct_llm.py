"""§8.6 baseline: direct DashScope LLM → sherpa-onnx, no orchestration layer.

This driver answers "what would TTFA / total response time look like if a
developer wrote the most natural streaming-LLM-to-TTS pipeline WITHOUT
the metahuman Router / Controller / Thinker / Talker / SpeakerBridge
stack?" R20 unified the two modes onto a single base model
(``qwen-plus-2025-09-11``); the only difference is whether DashScope's
deep-thinking budget is enabled:

* ``--modes non_thinking`` → streams ``qwen-plus-2025-09-11`` with
  ``enable_thinking=False``; sentence chunks fed straight into
  sherpa-onnx Piper TTS.
* ``--modes thinking`` → streams the SAME model with
  ``enable_thinking=True`` + ``thinking_budget=8000``. The model
  reasons internally (``reasoning_content`` field, discarded by the
  streamer) before emitting the final-answer ``content`` tokens; no
  multi-call thinker loop, the model does its own reasoning inside
  one call.
* ``--modes non_thinking,thinking`` → run both modes in sequence so a
  single CSV captures both baselines.

The base model is pinned via ``BASELINE_MODEL`` to match what
``talker.py`` uses for its ``easy_model`` / ``thinking_model`` defaults;
a regression test enforces that pin (``test_baseline_model_matches_
talker_default``). Earlier rounds (R14-R19) used different model
versions for the two modes (qwen-plus vs qwen-plus-2025-09-11),
which conflated "thinking value" with "model-version uplift" — see
the BASELINE_MODES docstring for the migration history.

What this baseline does NOT do (by design):
- No router classification (no NVIDIA call).
- No thinker multi-call loop (the thinking-model baseline asks once and
  lets the model self-reason).
- No talker mode switching (no easy / hard_first / hard_mid / hard_final).
- No SpeakerBridge — just sentence-chunk LLM stream → edge-tts.
- No audio gap measurement (one continuous stream → trivial 0).

Metrics captured:
- ``ttfa``: user request → first audio chunk landed at the consumer.
- ``total_time``: user request → last audio chunk played.
- ``llm_ttft``: user request → LLM first text token.
- ``llm_total_time``: user request → LLM final text token.
- ``audio_gap_total``: always 0 in this baseline (one stream).

Output schema (CSV, two files):
- ``baseline_direct_summary.csv``: one row per mode, aggregated over the
  fixture (means + p95 of the four latency metrics).
- ``baseline_direct_per_turn.csv``: one row per (mode, question), with
  source / subject / difficulty_hint preserved from the fixture.

Resumability (``--resume`` default on): per-mode logs are stored under
``--log-dir`` keyed by ``baseline-direct-{mode}-q{fixture_fingerprint}``.
The fingerprint covers the exact ordered question list so editing the
fixture invalidates old logs automatically. A mode is considered
complete only when every expected turn id has a finite ``ttfa`` AND
finite ``total_time`` AND finite ``llm_ttft`` AND finite
``llm_total_time``.

Fault containment matches the router-threshold sweep:
- Single-question failure → log + skip + count as ``invalid_turns``.
- Mode-level setup/teardown failure → write an ``error`` row, continue.
- CSVs are atomically rewritten after every mode completes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from metahuman.tts import TTSInterface, TTSResult
from metahuman.types import AudioMeta

DEFAULT_FIXTURE_PATH = Path(__file__).parent / "data" / "router_sweep_questions.jsonl"
DEFAULT_SUMMARY_CSV = Path("baseline_direct_summary.csv")
DEFAULT_PER_TURN_CSV = Path("baseline_direct_per_turn.csv")
DEFAULT_LOG_DIR = Path("logs/baseline-direct")

# Pinned to match src/metahuman/talker.py defaults so the baseline is a
# fair comparison (same backbone model, no orchestration). If those
# defaults change, update BOTH locations in lockstep.
#
# Both modes use the SAME underlying model (qwen-plus-2025-09-11). The
# only difference is the ``enable_thinking`` flag: the non_thinking
# mode disables DashScope's deep-thinking budget; the thinking mode
# enables it. This lets the baseline isolate the value of internal
# reasoning from any inter-model accuracy/speed differences. Earlier
# revisions used different models for each mode (qwen-plus vs.
# qwen-plus-2025-09-11), which conflated "thinking value" with
# "model-version uplift" in the per-mode metrics — observed in §8.6
# R19 baseline run where the thinking-mode total LLM time was
# **shorter** than non-thinking despite supposedly more reasoning.
BASELINE_MODEL: str = os.environ.get("METAHUMAN_LLM_MODEL", "qwen-plus-2025-09-11")
BASELINE_MODES: Dict[str, bool] = {
    "non_thinking": False,
    "thinking": True,
}
# DashScope's deep-thinking budget for the thinking mode (caps internal
# reasoning tokens; when exceeded, the model emits a final answer
# immediately). 8000 is a moderate cap: enough room for multi-step
# reasoning on Spoken-MQA / MMLU difficulty, far below the 32K wall
# clock observed on competition-grade math (see §10 thinking-budget
# scaling note).
THINKING_BUDGET_TOKENS: int = int(os.environ.get("METAHUMAN_THINKING_BUDGET", "8000"))

# Total-output caps. ``thinking_token_budget`` (vLLM) and
# ``thinking_budget`` (DashScope) cap the reasoning phase only;
# ``max_tokens`` caps reasoning + answer combined. We set both
# below the deployed ``max_model_len`` so requests never get
# truncated by the server. Override per-environment via
# ``METAHUMAN_MAX_TOKENS_THINKING`` / ``METAHUMAN_MAX_TOKENS_NON_THINKING``.
MAX_TOKENS_THINKING: int = int(os.environ.get("METAHUMAN_MAX_TOKENS_THINKING", "30000"))
MAX_TOKENS_NON_THINKING: int = int(os.environ.get("METAHUMAN_MAX_TOKENS_NON_THINKING", "20000"))
TURN_TIMEOUT_SECONDS: float = float(os.environ.get("METAHUMAN_TURN_TIMEOUT_SECONDS", "1200"))

# Prompt-construction policy for the baseline. ``"raw_question"`` means
# the driver sends ``question.question`` verbatim with no system-prompt
# wrapping or answer-style instruction. This is the most "naive
# developer baseline" — no orchestration AND no prompt engineering.
# Recorded as a column in every per-turn JSONL event so summary
# analysis cannot mistakenly attribute an answer-length / tone
# difference (which would shift total_time) to orchestration overhead.
# A future variant ("prompt_v2_short_answer", "prompt_v3_voice_friendly",
# etc.) would change THIS constant in lockstep with the templated
# prompt formatter and surface in the same column for clear A/B.
PROMPT_CONTRACT: str = "raw_question"
PROMPT_CONTRACTS: Tuple[str, ...] = ("raw_question", "spoken_concise")


def format_prompt(question: str, prompt_contract: str) -> str:
    if prompt_contract == "raw_question":
        return question
    if prompt_contract == "spoken_concise":
        return (
            "You are the response generator for a spoken digital-human agent. "
            "Answer the user's question directly in a natural, concise spoken style. "
            "For tasks that require calculation or reasoning, make the final answer clear.\n\n"
            f"User question:\n{question}"
        )
    raise ValueError(f"unknown prompt_contract: {prompt_contract}")

# Sentence terminators matching DashScopeTalker._SENTENCE_TERMINATORS.
# Duplicated here so the baseline stays self-contained (no metahuman
# package imports beyond TTSInterface). A regression test in
# tests/test_baseline_direct_driver.py imports the talker class attribute
# at runtime and asserts equality with this string, so drift between
# the two files surfaces as a test failure rather than a silent metric
# mismatch (e.g., DashScope emits "\n" before punctuation → main system
# would flush to TTS earlier than the baseline if newlines are missing).
_SENTENCE_TERMINATORS: str = "。！？；.!?;\n"

logger = logging.getLogger("baseline_direct_llm")


# ----- result dataclasses -----


@dataclass(frozen=True)
class BaselineParams:
    """One sweep cell: a mode selecting whether to enable DashScope's
    deep-thinking budget on the shared base model."""
    mode: str  # "non_thinking" | "thinking"
    prompt_contract: str = PROMPT_CONTRACT

    @property
    def model(self) -> str:
        return BASELINE_MODEL

    @property
    def enable_thinking(self) -> bool:
        return BASELINE_MODES[self.mode]

    @property
    def dialog_stem(self) -> str:
        suffix = "" if self.prompt_contract == PROMPT_CONTRACT else f"-{self.prompt_contract}"
        return f"baseline-direct-{self.mode}{suffix}"

    def dialog_id(self, question_set_id: str) -> str:
        return f"{self.dialog_stem}-q{question_set_id}"


@dataclass(frozen=True)
class SummaryRow:
    """Per-mode aggregate row."""
    params: BaselineParams
    n_questions: int
    valid_turns: int
    invalid_turns: int
    error: Optional[str]
    ttfa_mean: float
    ttfa_p95: float
    total_time_mean: float
    total_time_p95: float
    llm_ttft_mean: float
    llm_ttft_p95: float
    llm_total_time_mean: float
    llm_total_time_p95: float


@dataclass(frozen=True)
class PerTurnRow:
    """Per-question row (one mode × one question).

    ``model`` is stored explicitly (not derived from ``BASELINE_MODES``
    at write time) so the CSV is interpretable without the source code
    and a future model-name change does not silently relabel old rows.
    ``model_output`` is the concatenated LLM text response — exposed in
    the CSV so a quick inspection can read the answer directly without
    opening the JSONL log; downstream scoring
    (``scripts/score_results.py``) is the authoritative path.

    ``dialog_id`` and ``turn_id`` (round-2 review M4) make scoring
    robust without reconstructing log paths from convention. ``dialog_id``
    is ``params.dialog_id(question_set_id)``; ``turn_id`` is
    ``f"{dialog_id}/turn-{i}"``.
    """
    mode: str
    model: str
    prompt_contract: str
    dialog_id: str
    turn_id: str
    question_id: str
    source: str
    subject: str
    difficulty_hint: str
    ttfa: float
    total_time: float
    llm_ttft: float
    llm_total_time: float
    audio_gap_total: float
    model_output: str
    error: Optional[str]


# ----- adapter wiring -----


@dataclass
class Adapters:
    """Per-mode adapter bundle.

    ``llm_streamer_factory(model_name, enable_thinking) -> LLMStreamer``
    returns a thin wrapper that owns one DashScope client and exposes
    ``stream(prompt) -> AsyncIterator[str]`` yielding text deltas
    (content only — reasoning_content is captured separately on the
    streamer instance). ``tts_factory()`` returns a fresh TTSInterface
    per mode (no state leak between modes, same invariant as the
    router sweep).
    """
    llm_streamer_factory: Callable[[str, bool], Any]
    tts_factory: Callable[[], Any]


def real_adapters() -> Adapters:
    """Production adapters: a thin DashScope streaming client + SherpaOnnxTTS
    (local Piper VITS; replaces EdgeTTS so latency stays deterministic).

    Imports are lazy so a smoke run on a machine without dashscope or
    sherpa-onnx still works.
    """
    from metahuman.adapters_impl.sherpa_onnx_tts import SherpaOnnxTTS
    import dashscope

    class DashScopeStreamer:
        """Thin DashScope streamer that yields ONLY ``content`` deltas.

        For ``enable_thinking=True`` the model emits two interleaved
        streams of fields: ``reasoning_content`` (internal reasoning,
        hidden from the user / never spoken) and ``content`` (the
        final answer that goes to TTS). We discard reasoning deltas
        on purpose — TTFA / first audio must reflect the moment the
        FINAL ANSWER's first token arrives, not when thinking begins.
        Reasoning duration (in seconds) is exposed via
        ``self.last_reasoning_seconds`` for the driver to log as a
        separate metric without polluting TTFA.

        For ``enable_thinking=False`` the model never emits
        reasoning_content; behaviour collapses to "stream content
        deltas only", same as before this change.

        ``incremental_output=True`` (changed from False in R20):
        DashScope guarantees each chunk's ``delta.content`` is a
        non-overlapping increment in this mode, so we no longer need
        the cumulative diff helper. The reasoning_content field uses
        the same incremental contract.
        """

        def __init__(
            self,
            model: str,
            *,
            enable_thinking: bool = False,
            thinking_budget: int = THINKING_BUDGET_TOKENS,
        ) -> None:
            self.model = model
            self.enable_thinking = bool(enable_thinking)
            self.thinking_budget = int(thinking_budget)
            # Populated by ``stream`` per call. Wall-clock seconds
            # spent receiving reasoning_content tokens before the first
            # content token arrived. 0.0 if ``enable_thinking=False``
            # or if no reasoning was emitted.
            self.last_reasoning_seconds: float = 0.0

        async def stream(self, prompt: str):
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "result_format": "message",
                "incremental_output": True,
            }
            if self.enable_thinking:
                kwargs["enable_thinking"] = True
                kwargs["thinking_budget"] = self.thinking_budget
            stream = await asyncio.to_thread(
                dashscope.Generation.call,
                **kwargs,
            )
            iterator = iter(stream)
            sentinel = object()
            reasoning_started_at: Optional[float] = None
            content_started_at: Optional[float] = None
            self.last_reasoning_seconds = 0.0
            while True:
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                reasoning_delta, content_delta = self._extract_deltas(chunk)
                if reasoning_delta and reasoning_started_at is None:
                    reasoning_started_at = time.monotonic()
                if content_delta:
                    if content_started_at is None:
                        content_started_at = time.monotonic()
                        if reasoning_started_at is not None:
                            self.last_reasoning_seconds = (
                                content_started_at - reasoning_started_at
                            )
                    yield content_delta

        def _extract_deltas(self, chunk: Any) -> Tuple[str, str]:
            """Return ``(reasoning_delta, content_delta)`` for one chunk.

            Both strings are empty when the chunk carries no new text
            for that field. Tolerant of dict / object shape differences
            in the dashscope SDK response.
            """
            output = self._get(chunk, "output")
            choices = self._get(output, "choices") if output is not None else None
            if choices is None:
                choices = self._get(chunk, "choices")
            if not choices:
                return "", ""
            choice = choices[0]
            delta = self._get(choice, "delta")
            if delta is None:
                # Some SDK versions wrap the increment in ``message``
                # instead of ``delta`` when streaming. Fall through.
                delta = self._get(choice, "message")
            if delta is None:
                return "", ""
            reasoning = self._get(delta, "reasoning_content")
            content = self._get(delta, "content")
            return (
                reasoning if isinstance(reasoning, str) else "",
                content if isinstance(content, str) else "",
            )

        def _get(self, value: Any, key: str) -> Any:
            if isinstance(value, dict):
                return value.get(key)
            return getattr(value, key, None)

    backend = os.environ.get("METAHUMAN_LLM_BACKEND", "dashscope").lower()
    if backend == "openai":
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "METAHUMAN_LLM_BACKEND=openai requires `pip install openai>=1.40`."
            ) from exc

        class OpenAIStreamer:
            """OpenAI-compatible streamer (vLLM / OpenAI cloud) with same
            interface as DashScopeStreamer above. For Qwen3 served by
            vLLM 0.11+, thinking mode is controlled via
            ``chat_template_kwargs={"enable_thinking": <bool>}`` in
            ``extra_body``; ``--reasoning-parser qwen3`` on the server
            splits the ``<think>`` block out of ``delta.content`` into
            ``delta.reasoning_content`` so we can keep the same
            "reasoning vs content" timing logic."""

            def __init__(
                self,
                model: str,
                *,
                enable_thinking: bool = False,
                thinking_budget: int = THINKING_BUDGET_TOKENS,
            ) -> None:
                self.model = model
                self.enable_thinking = bool(enable_thinking)
                self.thinking_budget = int(thinking_budget)
                self.last_reasoning_seconds: float = 0.0
                self._client = openai.AsyncOpenAI(
                    base_url=os.environ.get("OPENAI_API_BASE"),
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                )

            async def stream(self, prompt: str):
                # Qwen3 ``/no_think`` workaround for vLLM 0.20.1 +
                # ``--reasoning-parser qwen3`` stream-mode bug:
                # ``chat_template_kwargs.enable_thinking=false`` alone
                # is silently ignored in streaming mode (the model's
                # output ends up in the ``reasoning`` field, leaving
                # ``content`` empty). Append the ``/no_think``
                # directive — supported by Qwen3 chat templates — to
                # the user message so the model deterministically
                # skips the thinking phase. The ``chat_template_kwargs``
                # flag is left in place as a defensive double-set.
                user_text = prompt
                # Auto-detect dialect: DashScope's compat URL uses
                # ``enable_thinking`` / ``thinking_budget``; vLLM 0.19+
                # uses ``chat_template_kwargs`` and
                # ``thinking_token_budget``. For DashScope, the flag
                # works in streaming mode. For vLLM, the flag is
                # silently ignored in streaming mode (the model's
                # output goes to ``reasoning`` instead of ``content``);
                # the only reliable switch is the Qwen3 ``/no_think``
                # directive appended to the user message — and the
                # presence of ``chat_template_kwargs.enable_thinking=
                # False`` actually overrides ``/no_think``, so for
                # vLLM we MUST omit it on non-thinking calls.
                base_url = (os.environ.get("OPENAI_API_BASE") or "").lower()
                is_dashscope = "dashscope" in base_url
                if not self.enable_thinking and not is_dashscope:
                    if "/no_think" not in user_text:
                        user_text = user_text.rstrip() + " /no_think"
                extra_body: dict = {}
                if is_dashscope:
                    extra_body["enable_thinking"] = self.enable_thinking
                    if self.enable_thinking:
                        extra_body["thinking_budget"] = self.thinking_budget
                else:
                    if self.enable_thinking:
                        extra_body["chat_template_kwargs"] = {
                            "enable_thinking": True
                        }
                        extra_body["thinking_token_budget"] = self.thinking_budget
                kwargs: dict = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": user_text}],
                    "stream": True,
                    "extra_body": extra_body,
                }
                # Cap total output below max_model_len for vLLM (32768
                # in our deployment). For DashScope the cap is the
                # service-side default; we pass max_tokens to keep
                # parity with prior runs.
                if self.enable_thinking:
                    kwargs["max_tokens"] = MAX_TOKENS_THINKING
                else:
                    kwargs["max_tokens"] = MAX_TOKENS_NON_THINKING
                stream = await self._client.chat.completions.create(**kwargs)
                reasoning_started_at: Optional[float] = None
                content_started_at: Optional[float] = None
                self.last_reasoning_seconds = 0.0
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # vLLM 0.19+ aligned to OpenAI's o-series schema and
                    # renamed ``reasoning_content`` → ``reasoning``;
                    # DashScope and older vLLM still expose
                    # ``reasoning_content``. Read both — only one is
                    # populated per delta.
                    reasoning_delta = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                        or ""
                    )
                    content_delta = getattr(delta, "content", None) or ""
                    if reasoning_delta and reasoning_started_at is None:
                        reasoning_started_at = time.monotonic()
                    if content_delta:
                        if content_started_at is None:
                            content_started_at = time.monotonic()
                            if reasoning_started_at is not None:
                                self.last_reasoning_seconds = (
                                    content_started_at - reasoning_started_at
                                )
                        yield content_delta

        return Adapters(
            llm_streamer_factory=lambda model, enable_thinking: OpenAIStreamer(
                model, enable_thinking=enable_thinking
            ),
            tts_factory=lambda: SherpaOnnxTTS(),
        )

    return Adapters(
        llm_streamer_factory=lambda model, enable_thinking: DashScopeStreamer(
            model, enable_thinking=enable_thinking
        ),
        tts_factory=lambda: SherpaOnnxTTS(),
    )


def smoke_adapters() -> Adapters:
    """Deterministic fakes for the unit test. No DashScope / edge-tts.

    The fake LLM streamer yields a canned three-sentence response with
    fixed inter-token delays so per-question TTFA / total_time fall in a
    predictable window. The fake TTS mirrors sweep_router_threshold's
    smoke FakeTTS (duration_s=0.05 per call, virtual play_until cursor).
    """
    import numpy as np

    class FakeStreamer:
        def __init__(self, model: str) -> None:
            self.model = model
            self.calls: List[str] = []

        async def stream(self, prompt: str):
            self.calls.append(prompt)
            for sentence in ("Hello world.", "How are you?", "Goodbye."):
                for ch in sentence:
                    await asyncio.sleep(0.001)
                    yield ch

    class FakeTTS(TTSInterface):
        """Same shape as sweep_router_threshold smoke FakeTTS."""

        def __init__(self) -> None:
            self.calls: List[str] = []
            self._play_until = 0.0

        def synthesize_text(self, text: str, turn_anchor_monotonic: float) -> TTSResult:
            self.calls.append(text)
            started_at_monotonic = time.monotonic()
            finished_at_monotonic = time.monotonic()
            duration_s = 0.05
            self._play_until = max(self._play_until, finished_at_monotonic) + duration_s
            return TTSResult(
                audio=np.zeros(800, dtype=np.int16),
                sample_rate=16000,
                duration_s=duration_s,
                started_at=started_at_monotonic - turn_anchor_monotonic,
                finished_at=finished_at_monotonic - turn_anchor_monotonic,
            )

        def backlog_seconds(self) -> float:
            return max(0.0, self._play_until - time.monotonic())

    return Adapters(
        llm_streamer_factory=lambda model, enable_thinking: FakeStreamer(model),
        tts_factory=lambda: FakeTTS(),
    )


# ----- fixture loading + fingerprinting (mirrors sweep_router_threshold) -----


@dataclass(frozen=True)
class Question:
    id: str
    source: str
    subject: str
    question: str
    answer: str
    difficulty_hint: str


def load_fixture(path: Path) -> List[Question]:
    """Read a JSONL fixture into a list of Question dataclasses.

    Skips empty lines; raises ValueError on malformed JSON or on a row
    missing any required key. Order is preserved — the baselines iterate
    in file order so the per-question CSV is reproducible.
    """
    rows: List[Question] = []
    required = ("id", "source", "subject", "question", "answer", "difficulty_hint")
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
            rows.append(Question(**values))
    return rows


def question_set_fingerprint(questions: List[Question]) -> str:
    """SHA-256 prefix over the ordered (id, question) pairs."""
    blob = "\n".join(f"{q.id}\t{q.question}" for q in questions).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


# ----- mode parsing -----


def parse_modes(spec: str) -> List[str]:
    """Parse a comma-separated mode list. Each mode must be a known key
    of BASELINE_MODES. Empty or duplicate entries raise
    argparse.ArgumentTypeError."""
    seen: List[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in BASELINE_MODES:
            raise argparse.ArgumentTypeError(
                f"unknown mode {token!r}; valid choices: {sorted(BASELINE_MODES)}"
            )
        if token in seen:
            raise argparse.ArgumentTypeError(f"duplicate mode {token!r}")
        seen.append(token)
    if not seen:
        raise argparse.ArgumentTypeError("--modes must list at least one mode")
    return seen


def parse_prompt_contract(spec: str) -> str:
    token = spec.strip()
    if token not in PROMPT_CONTRACTS:
        raise argparse.ArgumentTypeError(
            f"unknown prompt contract {token!r}; valid choices: {sorted(PROMPT_CONTRACTS)}"
        )
    return token


# ----- per-cell run -----


async def run_cell(
    params: BaselineParams,
    questions: List[Question],
    adapters: Adapters,
    log_dir: Path,
    question_set_id: str,
) -> Path:
    """Stream the LLM for ``params.model`` over every question, sentence-
    chunk into edge-tts, and write a JSONL log to ``log_dir`` so the
    aggregate / resume paths can read it back later.

    The JSONL schema is hand-rolled (no WorkspaceStore — the whole
    point of this baseline is no orchestration). Each turn produces:

    * ``user_utterance``: ``{turn_id, received_at, text, question_id,
      source, subject, difficulty_hint}`` — the trailing four fields
      let downstream analysis correlate without rejoining the fixture.
    * ``baseline_turn_metrics``: ``{turn_id, mode, model, prompt_contract,
      ttfa, total_time, llm_ttft, llm_total_time, audio_gap_total,
      model_output}`` — ``model_output`` is the concatenated LLM text
      response, captured for post-hoc scoring; ``prompt_contract`` is
      ``"raw_question"`` (the baseline sends ``question.question``
      verbatim, no system-prompt wrapping; documented so summary
      analysis does not over-interpret total_time differences caused
      by answer-length policy drift).

    On per-question failure, emit one ``baseline_turn_error``:
    ``{turn_id, mode, model, error}`` so the JSONL is self-describing
    and a remote run can be debugged from the log alone.

    ``play_until_monotonic`` is initialized to ``time.monotonic()`` at
    cell start and CARRIED across the question loop — matches the
    SpeakerBridge per-cell cursor lifecycle in the router sweep.

    Per-question failure isolation: log the exception, emit
    ``baseline_turn_error``, continue with the next question. The
    missing metrics show up as invalid_turns at aggregate time.
    Mode-level setup/teardown failure raises out, gets caught by
    ``main_async``, and writes an error row.
    """
    dialog_id = params.dialog_id(question_set_id)
    log_path = Path(log_dir) / f"{dialog_id}.ws.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    streamer = adapters.llm_streamer_factory(params.model, params.enable_thinking)
    tts = adapters.tts_factory()
    play_until_monotonic = time.monotonic()

    def event(kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"ts": time.time(), "kind": kind, "data": data}

    with log_path.open("w", encoding="utf-8") as handle:
        def emit(kind: str, data: Dict[str, Any]) -> None:
            handle.write(json.dumps(event(kind, data), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()

        for index, question in enumerate(questions):
            turn_id = f"{dialog_id}/turn-{index}"
            anchor_monotonic = time.monotonic()
            prompt = format_prompt(question.question, params.prompt_contract)
            emit(
                "user_utterance",
                {
                    "turn_id": turn_id,
                    "received_at": 0.0,
                    "text": question.question,
                    "question_id": question.id,
                    "source": question.source,
                    "subject": question.subject,
                    "difficulty_hint": question.difficulty_hint,
                },
            )
            try:
                # Per-turn cap. Defends against a hung stream wedging
                # the entire baseline run; controlled by
                # METAHUMAN_TURN_TIMEOUT_SECONDS for long ablations.
                timings = await asyncio.wait_for(
                    _stream_one_turn(
                        streamer,
                        tts,
                        prompt,
                        anchor_monotonic,
                        play_until_monotonic,
                    ),
                    timeout=TURN_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "question timed out after %.0f s; mode=%s turn=%s",
                    TURN_TIMEOUT_SECONDS, params.mode, turn_id,
                )
                emit(
                    "baseline_turn_error",
                    {
                        "turn_id": turn_id,
                        "mode": params.mode,
                        "model": params.model,
                        "prompt_contract": params.prompt_contract,
                        "error": f"TimeoutError: turn exceeded {TURN_TIMEOUT_SECONDS:.0f} s",
                    },
                )
                continue
            except Exception as exc:
                logger.exception("question failed; mode=%s turn=%s", params.mode, turn_id)
                emit(
                    "baseline_turn_error",
                    {
                        "turn_id": turn_id,
                        "mode": params.mode,
                        "model": params.model,
                        "prompt_contract": params.prompt_contract,
                        "error": repr(exc),
                    },
                )
                continue

            play_until_monotonic = timings.play_until_monotonic
            # Drain the playback queue before moving to the next
            # question — mirrors the sweep driver's
            # ``bridge.wait_until_drained()`` step. Each turn must
            # start with an empty virtual playback buffer so the next
            # turn's ttfa is not inflated by queue-wait inherited from
            # this turn's audio. Cap the wait at 60 s so a single
            # misbehaving turn cannot wedge the whole baseline run.
            now_monotonic = time.monotonic()
            backlog = max(0.0, play_until_monotonic - now_monotonic)
            if backlog > 0:
                await asyncio.sleep(min(backlog, 60.0))
            # **Always** rebase the cursor to "now" after the drain
            # attempt — even if the cap kicked in and some audio is
            # still nominally queued. Without this rebase the next
            # turn's ``playback_start = max(now, play_until)`` picks
            # up the leftover queue, and per-turn ttfa drifts up
            # monotonically across the cell (observed: 4.79 s on
            # turn-0 → 4727 s on turn-39 in the §8.6 R19 baseline
            # run). The experimental contract is "each turn measures
            # only its own synth pipeline"; queue accumulation across
            # turns is a deployment-realism concern that does not
            # belong in per-turn ttfa.
            play_until_monotonic = time.monotonic()
            emit(
                "baseline_turn_metrics",
                {
                    "turn_id": turn_id,
                    "mode": params.mode,
                    "model": params.model,
                    "prompt_contract": params.prompt_contract,
                    "ttfa": timings.ttfa,
                    "total_time": timings.total_time,
                    "llm_ttft": timings.llm_ttft,
                    "llm_total_time": timings.llm_total_time,
                    "audio_gap_total": 0.0,
                    "model_output": timings.model_output,
                },
            )
    return log_path


# ----- per-turn streaming + chunking -----


@dataclass(frozen=True)
class TurnTimings:
    """What ``_stream_one_turn`` measures, in turn-relative seconds.

    ``play_until_monotonic`` is the updated raw-monotonic playback cursor
    after the turn finishes. ``run_cell`` carries it across questions
    within the same mode so the per-mode virtual playback queue mirrors
    SpeakerBridge's per-cell-dialog cursor in the router sweep — without
    this, an audio answer that's still playing when the next question
    fires would be invisible to the next turn's timing math.

    ``model_output`` is the full concatenated LLM text for the turn,
    captured for post-hoc scoring (extract MMLU letter / Spoken-MQA
    numerical answer). The driver does NOT score inline; scoring
    lives in scripts/score_results.py.
    """
    llm_ttft: float
    llm_total_time: float
    ttfa: float
    total_time: float
    play_until_monotonic: float
    model_output: str


async def _stream_one_turn(
    streamer: Any,
    tts: Any,
    prompt: str,
    anchor_monotonic: float,
    play_until_monotonic: float,
) -> TurnTimings:
    """Stream the LLM AND synthesize sentence chunks concurrently.

    Pipeline shape (round-1 review High): the LLM streamer must keep
    consuming DashScope chunks while the TTS consumer synthesizes
    earlier sentence chunks — chunk N is in TTS while the LLM produces
    chunk N+1. If TTS synthesis is awaited inside the LLM iteration
    loop, the baseline measures "synth latency stalls LLM" instead of
    the natural direct-streaming pipeline a real developer would build,
    and the experiment's headline numbers become unfair.

    Producer (LLM) task:
      - consume ``streamer.stream(prompt)`` deltas;
      - stamp ``llm_first_at`` on the first non-empty delta;
      - append to a running buffer; whenever a sentence terminator
        (._SENTENCE_TERMINATORS) appears, flush the completed text up
        to and including the terminator to a sentence queue;
      - on stream end, stamp ``llm_last_at`` immediately, push the
        residual buffer (if non-empty) to the queue, then push a None
        sentinel.

    Consumer (TTS) task:
      - pull chunks from the queue in order;
      - await ``tts.async_synthesize_text(chunk, anchor_monotonic)``;
      - playback_start = max(time.monotonic(), play_until_monotonic);
        playback_end = playback_start + result.duration_s;
        update play_until_monotonic;
      - stamp ``tts_first_at`` on the first playback_start.

    ``model_output`` is the concatenation of every text chunk that
    actually went through the synth path (so "\\boxed{...}" payload
    or "Answer is C" gets preserved verbatim for downstream scoring).

    Returns turn-relative timings + the updated cursor + the captured
    text. Empty streams return llm_ttft / ttfa as NaN so resume rejects
    the turn and run_cell counts it invalid.

    Sentence chunking uses ``_SENTENCE_TERMINATORS`` from this module —
    a runtime regression test asserts that string equals
    ``DashScopeTalker._SENTENCE_TERMINATORS``.
    """
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
    llm_first_at: Optional[float] = None
    llm_last_at: Optional[float] = None
    tts_first_at: Optional[float] = None
    model_output_parts: List[str] = []

    def first_terminator_index(text: str) -> Optional[int]:
        indexes = [index for index, char in enumerate(text) if char in _SENTENCE_TERMINATORS]
        if not indexes:
            return None
        return min(indexes)

    async def producer() -> None:
        nonlocal llm_first_at, llm_last_at
        buffer = ""
        try:
            async for delta in streamer.stream(prompt):
                if delta and llm_first_at is None:
                    llm_first_at = time.monotonic()
                buffer += delta
                while True:
                    idx = first_terminator_index(buffer)
                    if idx is None:
                        break
                    chunk_text = buffer[:idx + 1]
                    buffer = buffer[idx + 1:]
                    stripped = chunk_text.strip()
                    if not stripped:
                        continue
                    model_output_parts.append(chunk_text)
                    await queue.put(stripped)
            llm_last_at = time.monotonic()
            residual = buffer.strip()
            if residual:
                model_output_parts.append(buffer)
                await queue.put(residual)
        finally:
            await queue.put(None)

    async def consumer() -> None:
        nonlocal play_until_monotonic, tts_first_at
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            result = await tts.async_synthesize_text(chunk, anchor_monotonic)
            playback_start = max(time.monotonic(), play_until_monotonic)
            playback_end = playback_start + result.duration_s
            play_until_monotonic = playback_end
            if tts_first_at is None:
                tts_first_at = playback_start

    producer_task = asyncio.create_task(producer())
    consumer_task = asyncio.create_task(consumer())
    tasks = (producer_task, consumer_task)
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = [
        result for result in results
        if isinstance(result, BaseException)
        and not isinstance(result, asyncio.CancelledError)
    ]
    if errors:
        for secondary in errors[1:]:
            logger.error(
                "secondary _stream_one_turn task failed",
                exc_info=(type(secondary), secondary, secondary.__traceback__),
            )
        raise errors[0]

    return TurnTimings(
        llm_ttft=(llm_first_at - anchor_monotonic) if llm_first_at is not None else math.nan,
        llm_total_time=(llm_last_at - anchor_monotonic) if llm_last_at is not None else math.nan,
        ttfa=(tts_first_at - anchor_monotonic) if tts_first_at is not None else math.nan,
        total_time=play_until_monotonic - anchor_monotonic,
        play_until_monotonic=play_until_monotonic,
        model_output="".join(model_output_parts),
    )


# ----- aggregation -----


def _percentile(values: List[float], p: float) -> float:
    """Linear-interpolation percentile over already-finite-filtered
    values. NaN for empty input; the only value for length 1; otherwise
    interpolate at ``(len-1) * p``.
    """
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def _parse_log_with_per_turn_metrics(log_path: Path) -> Dict[str, dict]:
    """Single-pass JSONL parse for baseline logs.

    Returns one record per turn id with the FULL per-turn payload
    (round-2 review L2: parser must preserve everything resume +
    aggregate downstream callers need, not just the four latency
    numbers). Specifically the per-turn dict carries:

    * ``ttfa`` / ``total_time`` / ``llm_ttft`` / ``llm_total_time``:
      the four latency floats from ``baseline_turn_metrics``.
      ``audio_gap_total`` is also included (always 0.0 in baseline).
    * ``mode`` / ``model`` / ``prompt_contract`` / ``model_output``:
      preserved so ``cell_already_complete`` can validate the log
      was written under the current ``BASELINE_MODES`` entry and
      so post-hoc scoring can read the model output without a
      second pass.

    Turns whose only event is ``baseline_turn_error`` (per-question
    failure path) appear in the returned dict with NO latency keys —
    use ``.get()`` with NaN defaults at the call site.

    Tolerates blank lines; raises ValueError on malformed JSON or
    missing kind/data envelope. Used by both ``aggregate_cell`` and
    ``cell_already_complete``.
    """
    result: Dict[str, dict] = {}
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
            if not isinstance(kind, str) or not isinstance(data, dict):
                raise ValueError(f"malformed JSONL at {log_path}:{line_number}: missing kind/data")
            turn_id = data.get("turn_id")
            if not isinstance(turn_id, str):
                continue
            if kind == "baseline_turn_metrics":
                record = result.setdefault(turn_id, {})
                for key in (
                    "ttfa",
                    "total_time",
                    "llm_ttft",
                    "llm_total_time",
                    "audio_gap_total",
                    "mode",
                    "model",
                    "prompt_contract",
                    "model_output",
                ):
                    record[key] = data.get(key)
            elif kind == "baseline_turn_error":
                record = result.setdefault(turn_id, {})
                record["error"] = data.get("error")
                record["mode"] = data.get("mode")
                record["model"] = data.get("model")
                record["prompt_contract"] = data.get("prompt_contract")
    return result


def aggregate_cell(
    params: BaselineParams,
    questions: List[Question],
    log_path: Optional[Path],
    error: Optional[str],
) -> Tuple[SummaryRow, List[PerTurnRow]]:
    """Read the JSONL log and produce one SummaryRow + one PerTurnRow
    per question.

    Validity rule: a turn is valid iff ``ttfa``, ``total_time``,
    ``llm_ttft``, AND ``llm_total_time`` are all finite. Aggregates
    compute over the valid subset; ``valid_turns`` is the count.
    Per-question CSV rows are emitted for every expected question (one
    PerTurnRow per fixture entry, even if its log entry is missing or
    invalid).

    ``error`` is plumbed through from a cell-level exception in
    ``run_cell``; when set, the row carries the error message and
    aggregates default to NaN / 0 without reading the log.
    """
    n_questions = len(questions)
    if error is not None or log_path is None:
        row_error = error if error is not None else "missing log path"
        return (
            SummaryRow(
                params=params,
                n_questions=n_questions,
                valid_turns=0,
                invalid_turns=n_questions,
                error=row_error,
                ttfa_mean=math.nan,
                ttfa_p95=math.nan,
                total_time_mean=math.nan,
                total_time_p95=math.nan,
                llm_ttft_mean=math.nan,
                llm_ttft_p95=math.nan,
                llm_total_time_mean=math.nan,
                llm_total_time_p95=math.nan,
            ),
            [
                PerTurnRow(
                    mode=params.mode,
                    model=params.model,
                    prompt_contract=params.prompt_contract,
                    dialog_id="",
                    turn_id="",
                    question_id=question.id,
                    source=question.source,
                    subject=question.subject,
                    difficulty_hint=question.difficulty_hint,
                    ttfa=math.nan,
                    total_time=math.nan,
                    llm_ttft=math.nan,
                    llm_total_time=math.nan,
                    audio_gap_total=math.nan,
                    model_output="",
                    error=row_error,
                )
                for question in questions
            ],
        )

    try:
        parsed = _parse_log_with_per_turn_metrics(log_path)
    except ValueError as exc:
        return aggregate_cell(params, questions, None, repr(exc))

    dialog_id = log_path.name[:-len(".ws.jsonl")] if log_path.name.endswith(".ws.jsonl") else log_path.stem
    per_turn_rows: List[PerTurnRow] = []
    finite_rows: List[PerTurnRow] = []
    for index, question in enumerate(questions):
        turn_id = f"{dialog_id}/turn-{index}"
        record = parsed.get(turn_id, {})
        ttfa = float(record.get("ttfa", math.nan)) if isinstance(record.get("ttfa", math.nan), (int, float)) else math.nan
        total_time_value = record.get("total_time", math.nan)
        llm_ttft_value = record.get("llm_ttft", math.nan)
        llm_total_time_value = record.get("llm_total_time", math.nan)
        audio_gap_value = record.get("audio_gap_total", math.nan)
        total_time = float(total_time_value) if isinstance(total_time_value, (int, float)) else math.nan
        llm_ttft = float(llm_ttft_value) if isinstance(llm_ttft_value, (int, float)) else math.nan
        llm_total_time = float(llm_total_time_value) if isinstance(llm_total_time_value, (int, float)) else math.nan
        audio_gap_total = float(audio_gap_value) if isinstance(audio_gap_value, (int, float)) else math.nan
        is_valid = all(
            math.isfinite(value)
            for value in (ttfa, total_time, llm_ttft, llm_total_time)
        )
        row = PerTurnRow(
            mode=params.mode,
            model=params.model,
            prompt_contract=str(record.get("prompt_contract") or params.prompt_contract),
            dialog_id=dialog_id,
            turn_id=turn_id,
            question_id=question.id,
            source=question.source,
            subject=question.subject,
            difficulty_hint=question.difficulty_hint,
            ttfa=ttfa,
            total_time=total_time,
            llm_ttft=llm_ttft,
            llm_total_time=llm_total_time,
            audio_gap_total=audio_gap_total,
            model_output=str(record.get("model_output") or ""),
            error=None if is_valid else str(record.get("error") or "missing or non-finite metrics"),
        )
        per_turn_rows.append(row)
        if is_valid:
            finite_rows.append(row)

    ttfas = [row.ttfa for row in finite_rows]
    total_times = [row.total_time for row in finite_rows]
    llm_ttfts = [row.llm_ttft for row in finite_rows]
    llm_total_times = [row.llm_total_time for row in finite_rows]
    return (
        SummaryRow(
            params=params,
            n_questions=n_questions,
            valid_turns=len(finite_rows),
            invalid_turns=max(0, n_questions - len(finite_rows)),
            error=None,
            ttfa_mean=_mean(ttfas),
            ttfa_p95=_percentile(ttfas, 0.95),
            total_time_mean=_mean(total_times),
            total_time_p95=_percentile(total_times, 0.95),
            llm_ttft_mean=_mean(llm_ttfts),
            llm_ttft_p95=_percentile(llm_ttfts, 0.95),
            llm_total_time_mean=_mean(llm_total_times),
            llm_total_time_p95=_percentile(llm_total_times, 0.95),
        ),
        per_turn_rows,
    )


# ----- resumability -----


def cell_already_complete(
    params: BaselineParams,
    log_dir: Path,
    expected_questions: List[Question],
    question_set_id: str,
) -> Tuple[bool, Path]:
    """Resume predicate. A mode is reusable iff its log exists AND every
    expected turn id has:

    * a ``baseline_turn_metrics`` event (route-equivalent: it ran end-to-end);
    * finite ``ttfa``, ``total_time``, ``llm_ttft``, AND ``llm_total_time``;
    * ``mode == params.mode`` AND ``model == params.model`` on the event
      payload (round-1 review M2: stale logs from an older
      ``BASELINE_MODES`` entry would otherwise be reused under the new
      label, producing a mislabeled summary row).

    Malformed JSONL → catch ValueError, log a warning, return
    ``(False, log_path)``; the caller unlinks before rerun.
    """
    dialog_id = params.dialog_id(question_set_id)
    log_path = Path(log_dir) / f"{dialog_id}.ws.jsonl"
    if not log_path.exists():
        return False, log_path

    try:
        parsed = _parse_log_with_per_turn_metrics(log_path)
    except ValueError as exc:
        logger.warning("existing baseline log is malformed and will be rerun: %s", exc)
        return False, log_path

    for index in range(len(expected_questions)):
        turn_id = f"{dialog_id}/turn-{index}"
        record = parsed.get(turn_id)
        if record is None:
            return False, log_path
        if record.get("mode") != params.mode:
            return False, log_path
        if record.get("model") != params.model:
            return False, log_path
        if record.get("prompt_contract") != params.prompt_contract:
            return False, log_path
        for key in ("ttfa", "total_time", "llm_ttft", "llm_total_time"):
            value = record.get(key, math.nan)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, log_path
            if not math.isfinite(float(value)):
                return False, log_path
    return True, log_path


# ----- CSV output -----


SUMMARY_FIELDS: Tuple[str, ...] = (
    "mode",
    "model",
    "prompt_contract",
    "n_questions",
    "valid_turns",
    "invalid_turns",
    "error",
    "ttfa_mean",
    "ttfa_p95",
    "total_time_mean",
    "total_time_p95",
    "llm_ttft_mean",
    "llm_ttft_p95",
    "llm_total_time_mean",
    "llm_total_time_p95",
)

PER_TURN_FIELDS: Tuple[str, ...] = (
    "mode",
    "model",
    "prompt_contract",
    "dialog_id",
    "turn_id",
    "question_id",
    "source",
    "subject",
    "difficulty_hint",
    "ttfa",
    "total_time",
    "llm_ttft",
    "llm_total_time",
    "audio_gap_total",
    "model_output",
    "error",
)


def write_summary_csv(rows: List[SummaryRow], path: Path) -> None:
    """Atomic full-rewrite of the per-mode summary CSV (parent.mkdir +
    .tmp + Path.replace, NaN → "nan", error None → "")."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.params.mode,
                    "model": row.params.model,
                    "prompt_contract": row.params.prompt_contract,
                    "n_questions": str(row.n_questions),
                    "valid_turns": str(row.valid_turns),
                    "invalid_turns": str(row.invalid_turns),
                    "error": row.error or "",
                    "ttfa_mean": str(row.ttfa_mean),
                    "ttfa_p95": str(row.ttfa_p95),
                    "total_time_mean": str(row.total_time_mean),
                    "total_time_p95": str(row.total_time_p95),
                    "llm_ttft_mean": str(row.llm_ttft_mean),
                    "llm_ttft_p95": str(row.llm_ttft_p95),
                    "llm_total_time_mean": str(row.llm_total_time_mean),
                    "llm_total_time_p95": str(row.llm_total_time_p95),
                }
            )
    tmp.replace(path)


def write_per_turn_csv(rows: List[PerTurnRow], path: Path) -> None:
    """Atomic full-rewrite of the per-question CSV. Same contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_TURN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row.mode,
                    "model": row.model,
                    "prompt_contract": row.prompt_contract,
                    "dialog_id": row.dialog_id,
                    "turn_id": row.turn_id,
                    "question_id": row.question_id,
                    "source": row.source,
                    "subject": row.subject,
                    "difficulty_hint": row.difficulty_hint,
                    "ttfa": str(row.ttfa),
                    "total_time": str(row.total_time),
                    "llm_ttft": str(row.llm_ttft),
                    "llm_total_time": str(row.llm_total_time),
                    "audio_gap_total": str(row.audio_gap_total),
                    "model_output": row.model_output,
                    "error": row.error or "",
                }
            )
    tmp.replace(path)


# ----- top-level -----


async def main_async(args: argparse.Namespace) -> int:
    """Entry point. The unit test calls this directly with a synthetic
    Namespace under --smoke."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    prompt_contract = getattr(args, "prompt_contract", PROMPT_CONTRACT)
    logger.info("baseline-direct sweep starting; mode=%s; modes=%s; prompt_contract=%s",
                "smoke" if args.smoke else "real", args.modes, prompt_contract)

    adapters = smoke_adapters() if args.smoke else real_adapters()
    questions = load_fixture(args.questions_file)
    question_set_id = question_set_fingerprint(questions)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[SummaryRow] = []
    per_turn_rows: List[PerTurnRow] = []

    for mode in args.modes:
        params = BaselineParams(mode=mode, prompt_contract=prompt_contract)
        dialog_log_path = log_dir / f"{params.dialog_id(question_set_id)}.ws.jsonl"
        if args.resume:
            complete, log_path = cell_already_complete(
                params,
                log_dir,
                expected_questions=questions,
                question_set_id=question_set_id,
            )
            if complete:
                logger.info("baseline mode complete; reusing log: %s", params.dialog_stem)
                summary, per_turn = aggregate_cell(params, questions, log_path, error=None)
                summary_rows.append(summary)
                per_turn_rows.extend(per_turn)
                write_summary_csv(summary_rows, args.summary_csv)
                write_per_turn_csv(per_turn_rows, args.per_turn_csv)
                continue
            if log_path.exists():
                log_path.unlink()
        elif dialog_log_path.exists():
            dialog_log_path.unlink()

        try:
            log_path = await run_cell(
                params,
                questions,
                adapters,
                log_dir,
                question_set_id,
            )
        except Exception as exc:
            logger.exception("baseline mode %s crashed at top level", params.dialog_stem)
            summary, per_turn = aggregate_cell(params, questions, None, error=repr(exc))
        else:
            summary, per_turn = aggregate_cell(params, questions, log_path, error=None)

        summary_rows.append(summary)
        per_turn_rows.extend(per_turn)
        write_summary_csv(summary_rows, args.summary_csv)
        write_per_turn_csv(per_turn_rows, args.per_turn_csv)

    successes = [
        row for row in summary_rows
        if not row.error and row.valid_turns == row.n_questions
    ]
    message = f"baseline-direct sweep complete: {len(successes)}/{len(summary_rows)} modes succeeded"
    finite_rows = [row for row in successes if math.isfinite(row.ttfa_mean)]
    if finite_rows:
        message += "; " + ", ".join(
            f"ttfa[{row.params.mode}]={row.ttfa_mean:.4f}" for row in finite_rows
        )
    print(message)
    return 0


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--questions-file", type=Path, default=DEFAULT_FIXTURE_PATH,
        help=f"JSONL fixture path (default: {DEFAULT_FIXTURE_PATH})",
    )
    parser.add_argument(
        "--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV,
        help=f"per-mode summary CSV (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--per-turn-csv", type=Path, default=DEFAULT_PER_TURN_CSV,
        help=f"per-question detail CSV (default: {DEFAULT_PER_TURN_CSV})",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR,
        help=f"per-mode JSONL output dir (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--modes", type=parse_modes,
        default=list(BASELINE_MODES.keys()),  # ["non_thinking", "thinking"]
        help='comma-separated mode list; valid choices: ' +
             ', '.join(sorted(BASELINE_MODES)),
    )
    parser.add_argument(
        "--prompt-contract",
        type=parse_prompt_contract,
        default=PROMPT_CONTRACT,
        help="prompt contract for the baseline request; valid choices: " +
             ", ".join(sorted(PROMPT_CONTRACTS)),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="real adapters (default)")
    mode.add_argument("--smoke", action="store_true",
                      help="deterministic fakes; no DashScope / edge-tts")
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="(default) skip modes whose JSONL is already complete",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="re-run every mode from scratch",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
