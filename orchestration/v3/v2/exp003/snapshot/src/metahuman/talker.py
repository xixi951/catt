"""Talker: 4-mode TTS-text generator (§6.5)."""
from __future__ import annotations

import abc
import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING, Tuple

from .types import (
    ControlSignal,
    EndOfThinking,
    TalkerMode,
    TalkerSegment,
    ThinkerEvent,
    ThinkerSegment,
    TurnId,
    UserUtterance,
    Workspace,
)


if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .workspace import WorkspaceStore


class TalkerInterface(abc.ABC):
    """Mode is supplied by the controller at dispatch time; emits text chunks for TTS."""

    @abc.abstractmethod
    def dispatch(self, signal: ControlSignal, mode: TalkerMode) -> AsyncIterator[str]:
        """Run one dispatch in the given ``mode``. Yields text chunks (already
        split by punctuation per §6.6) for the orchestrator to push to TTS.

        ``mode`` is captured by the controller in ``_send_talker`` BEFORE any
        state advancement (e.g., HARD_FIRST → HARD_MID auto-advance), so the
        talker must NOT re-derive mode from ``ws.state`` here. Doing so was the
        cause of an earlier race where HARD_FIRST dispatches read HARD_MID
        from the workspace and silently switched to the deep-thinking path,
        inflating user-perceived TTFA on every hard turn.

        Idempotency is scoped per turn: if signal.snapshot_seq <=
        last_completed_snapshot_seq(signal.turn_id), the dispatch is dropped.
        A new turn must not inherit completion state from earlier turns."""
        ...

    @abc.abstractmethod
    def last_completed_snapshot_seq(self, turn_id: TurnId) -> int:
        """Highest snapshot_seq of a completed dispatch for `turn_id`.
        Returns -1 if no dispatch has completed for this turn."""
        ...


class DashScopeTalker(TalkerInterface):
    """DashScope-backed Talker implementation with four prompt modes."""

    _PROMPT_FILES: Dict[TalkerMode, str] = {
        TalkerMode.EASY: "talker_easy.md",
        TalkerMode.HARD_FIRST: "talker_hard_first.md",
        TalkerMode.HARD_MID: "talker_hard_mid.md",
        TalkerMode.HARD_FINAL: "talker_hard_final.md",
    }
    _SENTENCE_TERMINATORS = "。！？；.!?;\n"
    # Comma-class characters used for "faster first response" (mirrors
    # Open-LLM-VTuber's sentence_divider.py): for the FIRST chunk of a
    # dispatch only, the talker also accepts a comma as a terminator so
    # the first audio reaches the user faster. Multilingual coverage:
    # Latin / CJK / Arabic / etc. Subsequent chunks fall back to
    # ``_SENTENCE_TERMINATORS`` only — splitting every clause would
    # produce choppy playback after the first.
    _FIRST_CHUNK_COMMA_TERMINATORS = ",，、"

    def __init__(
        self,
        ws: "WorkspaceStore",
        prompts_dir: Path = Path("prompts"),
        easy_model: str = "qwen-plus-2025-09-11",
        thinking_model: str = "qwen-plus-2025-09-11",
        max_prior_turns: int = 5,
        faster_first_response: bool = True,
        thinking_budget: int = 8000,
        hard_final_thinking_budget: int = 2000,
        enable_thinking: bool = True,
        llm_client: "Optional[LLMClient]" = None,
    ) -> None:
        """``faster_first_response`` (default True) lets the FIRST chunk
        of every dispatch terminate at a comma in addition to the usual
        sentence-end punctuation. Mirrors Open-LLM-VTuber's
        ``faster_first_response`` flag: shaves 300-1500 ms off TTFA in
        the typical case where the LLM's first sentence runs long
        before its closing period. Subsequent chunks of the same
        dispatch use the strict sentence-only terminator set so
        playback does not become choppy.

        ``easy_model`` and ``thinking_model`` default to the SAME
        base model name (``qwen-plus-2025-09-11``). The two
        configurations differ only in whether DashScope's deep-
        thinking budget is enabled — controlled per dispatch via
        ``_should_enable_thinking(mode)``: HARD_MID / HARD_FINAL
        activate the budget, EASY / HARD_FIRST do not. Earlier
        revisions used different model versions for the two roles
        (qwen-plus vs. qwen-plus-2025-09-11), which conflated
        "thinking value" with "model-version uplift" in §8.6
        baseline observations. ``thinking_budget`` (default 8000)
        caps the per-call internal reasoning tokens.

        ``enable_thinking`` (default True) is a global override: when
        False, ``_should_enable_thinking`` always returns False so even
        HARD_MID and HARD_FINAL run without DashScope's deep-thinking
        budget. This is the lever the §8.5 R22 sweep uses for the
        "disable" cell — letting us measure the speed/quality
        trade-off of running the whole pipeline without internal
        reasoning. When True (the default), per-mode behaviour is
        unchanged (HARD_MID / HARD_FINAL on, EASY / HARD_FIRST off).

        ``llm_client`` (default None) selects the streaming chat
        backend. None falls back to ``DashScopeClient()`` so existing
        code paths keep their behavior. Pass an ``OpenAIClient`` to
        target a local vLLM server or any OpenAI-compatible endpoint.
        """
        from .llm_client import DashScopeClient

        if max_prior_turns < 0:
            raise ValueError("max_prior_turns must be >= 0")

        self._ws = ws
        self._prompts_dir = Path(prompts_dir)
        self._easy_model = easy_model
        self._thinking_model = thinking_model
        self._max_prior_turns = max_prior_turns
        self._faster_first_response = bool(faster_first_response)
        self._thinking_budget = int(thinking_budget)
        # HARD_FINAL gets a *small* budget (default 2000) because by the
        # time it dispatches the thinker has already produced a full
        # final_summary; the talker only needs minimal organising
        # thought to weave that summary together with prior talker
        # output. Larger budgets here added 30-90 s of dead time to the
        # last segment, dominating the §8.6 audio-gap mean.
        # HARD_MID is forced to *non-thinking* in `_should_enable_thinking`
        # since it just relays the latest streamed thinker chunks; any
        # internal reasoning would just delay TTS.
        self._hard_final_thinking_budget = int(hard_final_thinking_budget)
        self._enable_thinking = bool(enable_thinking)
        self._client = llm_client if llm_client is not None else DashScopeClient()
        self._prompts = self._load_prompts(self._prompts_dir)
        self._last_completed: Dict[TurnId, int] = {}

    async def dispatch(self, signal: ControlSignal, mode: TalkerMode) -> AsyncIterator[str]:
        """Run one DashScope dispatch in the given ``mode`` and yield
        punctuation-split chunks. ``mode`` is supplied by the controller's
        ``_send_talker`` (captured pre-state-mutation) — see TalkerInterface
        docstring on why we must not re-read ``ws.state`` here.
        """
        if signal.snapshot_seq <= self._last_completed.get(signal.turn_id, -1):
            return

        snapshot = self._ws.snapshot()
        model = self._model_for_mode(mode)
        prompt = self._build_prompt(signal.turn_id, mode, snapshot)

        buffer = ""
        # Stamps the first delta of this dispatch so per-stage latency
        # analysis can split "talker dispatch start → talker LLM first
        # token" from "first token → first sentence ready for TTS".
        # Per-dispatch (not per-turn) since hard path can have several
        # dispatches per turn and we want each one's first-token timing.
        first_token_recorded = False
        # Tracks whether this dispatch has already yielded a chunk to
        # the bridge. If False AND ``faster_first_response`` is enabled,
        # _split_complete_sentences also accepts a comma as a
        # terminator so the first audio reaches the listener faster.
        first_chunk_yielded = False
        # HARD_FINAL gets a smaller, dedicated thinking budget; every
        # other thinking-eligible mode (currently none, since HARD_MID
        # is forced off) would use the full ``self._thinking_budget``.
        eff_budget = (
            self._hard_final_thinking_budget
            if mode == TalkerMode.HARD_FINAL
            else self._thinking_budget
        )
        async for text in self._stream_generation(
            model=model,
            prompt=prompt,
            enable_thinking=self._should_enable_thinking(mode),
            thinking_budget=eff_budget,
        ):
            if not first_token_recorded:
                first_token_recorded = True
                # Defensive: tests that exercise dispatch without going
                # through LLMBundle.handle_user_text never register the
                # turn anchor, so ws.now() raises KeyError. The event is
                # purely diagnostic — skip it silently if no anchor.
                try:
                    ts = self._ws.now(signal.turn_id)
                except KeyError:
                    ts = None
                if ts is not None:
                    self._ws.record_event(
                        "talker_first_token",
                        {
                            "turn_id": signal.turn_id,
                            "snapshot_seq": signal.snapshot_seq,
                            "mode": mode.value,
                            "ts": ts,
                        },
                    )
            buffer += text
            sentences, buffer = self._split_complete_sentences(
                buffer,
                accept_comma=self._faster_first_response and not first_chunk_yielded,
            )
            for sentence in sentences:
                yield sentence
                first_chunk_yielded = True

        final_text = buffer.strip()
        if final_text:
            yield final_text

        self._last_completed[signal.turn_id] = max(
            self._last_completed.get(signal.turn_id, -1),
            signal.snapshot_seq,
        )

    def last_completed_snapshot_seq(self, turn_id: TurnId) -> int:
        """Highest snapshot_seq of a completed dispatch for `turn_id`."""
        return self._last_completed.get(turn_id, -1)

    def _load_prompts(self, prompts_dir: Path) -> Dict[TalkerMode, str]:
        prompts: Dict[TalkerMode, str] = {}
        for mode, filename in self._PROMPT_FILES.items():
            path = prompts_dir / filename
            prompts[mode] = path.read_text(encoding="utf-8")
        return prompts

    def _model_for_mode(self, mode: TalkerMode) -> str:
        if mode in (TalkerMode.EASY, TalkerMode.HARD_FIRST):
            return self._easy_model
        return self._thinking_model

    def _should_enable_thinking(self, mode: TalkerMode) -> bool:
        """Whether DashScope's deep-thinking budget should be enabled
        for this dispatch.

        EASY and HARD_FIRST are designed for the FAST first response
        — the user must hear something within ~1 s, so we cannot
        afford the model to think internally for several seconds.
        HARD_MID and HARD_FINAL are designed for substantive
        responses; they enable the budget so the model can reason
        before producing audio-bound text. Mirrors the previous
        per-mode model split which used a separate "thinking" model
        version for hard_mid/hard_final but is now controlled by a
        single boolean flag (R20: unify base model).

        ``self._enable_thinking`` is a global gate: when False, the
        per-mode True is overridden back to False. This is the lever
        the §8.5 R22 sweep's "disable" cell uses to measure
        end-to-end behaviour without any internal reasoning anywhere
        in the pipeline.
        """
        if not self._enable_thinking:
            return False
        # All four modes are now forced non-thinking. The original
        # design let HARD_MID and HARD_FINAL run with internal
        # reasoning to "polish" their output, but in practice this
        # added 5-22 s per dispatch with no measurable accuracy gain
        # once the streaming Thinker emits content tokens as
        # ThinkerSegments (so HARD_FINAL has the model's full polished
        # answer in thinker_history and only needs to confirm it).
        # Forcing non-thinking here cuts the lingering hard_mid ->
        # hard_final transition gap to ~1 s, making the system's
        # audio truly seamless.
        return False

    def _build_prompt(self, turn_id: TurnId, mode: TalkerMode, snapshot: Workspace) -> str:
        user_text = self._ws.current_user_text(turn_id) or ""
        thinker_history = ""
        if mode != TalkerMode.HARD_FIRST:
            thinker_history = self._format_thinker_history(
                self._current_turn_thinker(snapshot, turn_id),
                prefer_final_summary=mode == TalkerMode.HARD_FINAL,
            )

        return self._prompts[mode].format(
            user_text=user_text,
            thinker_history=thinker_history,
            talker_history=self._format_talker_history(
                self._current_turn_talker(snapshot, turn_id)
            ),
            prior_dialog=self._format_prior_dialog(snapshot, turn_id),
        )

    def _current_turn_thinker(
        self,
        snapshot: Workspace,
        turn_id: TurnId,
    ) -> List[ThinkerEvent]:
        return [event for event in snapshot.thinker_history if event.turn_id == turn_id]

    def _current_turn_talker(
        self,
        snapshot: Workspace,
        turn_id: TurnId,
    ) -> List[TalkerSegment]:
        return [segment for segment in snapshot.talker_history if segment.turn_id == turn_id]

    def _format_thinker_history(
        self,
        events: List[ThinkerEvent],
        prefer_final_summary: bool,
    ) -> str:
        if not events:
            return ""

        segment_lines: List[str] = []
        final_summary: Optional[str] = None
        for event in events:
            if isinstance(event, ThinkerSegment):
                content = event.content.strip()
                if content:
                    segment_lines.append(f"思考片段 {event.seq}: {content}")
            elif isinstance(event, EndOfThinking):
                summary = event.final_summary.strip()
                if summary:
                    final_summary = summary

        if prefer_final_summary and final_summary is not None:
            lines = [f"最终思考摘要: {final_summary}"]
            if segment_lines:
                lines.append("本轮思考片段:")
                lines.extend(segment_lines)
            return "\n".join(lines)

        lines = list(segment_lines)
        if final_summary is not None:
            lines.append(f"最终思考摘要: {final_summary}")
        return "\n".join(lines)

    def _format_talker_history(self, segments: List[TalkerSegment]) -> str:
        lines = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                lines.append(f"已说过 {segment.seq} ({segment.mode.value}): {text}")
        return "\n".join(lines)

    def _format_prior_dialog(self, snapshot: Workspace, turn_id: TurnId) -> str:
        turn_order: List[TurnId] = []
        for utterance in snapshot.user_history:
            if utterance.turn_id != turn_id and utterance.turn_id not in turn_order:
                turn_order.append(utterance.turn_id)
        for segment in snapshot.talker_history:
            if segment.turn_id != turn_id and segment.turn_id not in turn_order:
                turn_order.append(segment.turn_id)

        # Keep the most recent max_prior_turns turns; older context is dropped
        # to bound prompt length (see research-plan §6.5 long-dialog note).
        if self._max_prior_turns > 0:
            turn_order = turn_order[-self._max_prior_turns:]
        else:
            turn_order = []

        lines: List[str] = []
        for prior_turn_id in turn_order:
            for utterance in self._user_history_for_turn(snapshot, prior_turn_id):
                lines.append(f"用户({prior_turn_id}): {utterance.text.strip()}")
            for segment in self._talker_history_for_turn(snapshot, prior_turn_id):
                text = segment.text.strip()
                if text:
                    lines.append(f"数字人({prior_turn_id}): {text}")
        return "\n".join(lines)

    def _user_history_for_turn(
        self,
        snapshot: Workspace,
        turn_id: TurnId,
    ) -> List[UserUtterance]:
        return [utterance for utterance in snapshot.user_history if utterance.turn_id == turn_id]

    def _talker_history_for_turn(
        self,
        snapshot: Workspace,
        turn_id: TurnId,
    ) -> List[TalkerSegment]:
        return [segment for segment in snapshot.talker_history if segment.turn_id == turn_id]

    async def _stream_generation(
        self,
        model: str,
        prompt: str,
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Yield ``content`` deltas only via the configured ``LLMClient``.

        Drops ``reasoning_content`` at the client boundary so the TTS
        pipeline only ever sees final-answer text. Per-mode
        ``enable_thinking`` is decided by ``_should_enable_thinking``;
        ``thinking_budget`` defaults to ``self._thinking_budget`` and
        can be overridden per dispatch (``dispatch`` passes
        ``hard_final_thinking_budget`` for HARD_FINAL).
        """
        budget = (
            thinking_budget if thinking_budget is not None
            else self._thinking_budget
        )
        async for chunk in self._client.stream_chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            enable_thinking=enable_thinking,
            thinking_budget=budget,
        ):
            if chunk:
                yield chunk

    def _split_complete_sentences(
        self,
        text: str,
        *,
        accept_comma: bool = False,
    ) -> Tuple[List[str], str]:
        """Split ``text`` into complete sentences plus the trailing
        partial buffer. Standard terminator set is
        ``_SENTENCE_TERMINATORS`` (sentence-end punctuation).

        When ``accept_comma`` is True, the comma class
        (``_FIRST_CHUNK_COMMA_TERMINATORS``) is also accepted as a
        terminator. The dispatch loop sets this only while no chunk
        has been yielded yet — once the first chunk is out, splitting
        returns to strict sentence-end so subsequent playback is not
        choppy.
        """
        terminators = self._SENTENCE_TERMINATORS
        if accept_comma:
            terminators = terminators + self._FIRST_CHUNK_COMMA_TERMINATORS
        sentences: List[str] = []
        start = 0
        for index, char in enumerate(text):
            if char in terminators:
                sentence = text[start:index + 1].strip()
                if sentence:
                    sentences.append(sentence)
                start = index + 1
        return sentences, text[start:]
