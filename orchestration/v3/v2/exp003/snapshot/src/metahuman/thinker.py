"""Thinker: single-call streaming reasoner (§6.4).

Phase 1 redesign. Instead of a multi-call SEGMENT/<END_OF_THINKING>
tag-protocol loop, we make ONE streaming chat call with
``enable_thinking=True``, consume the model's hidden CoT as it streams
in via ``reasoning_content``, and dispatch chunks at natural prose
boundaries. The model's visible ``content`` (expected to be just
``\\boxed{<integer>}``) is collected in parallel and delivered as the
``EndOfThinking.final_summary``.

Consumer-side chunking policy (``_split_chunk``):
1. Paragraph break (``\\n\\n``): flush up to and including the first
   double newline.
2. Sentence count: when the buffer accumulates >= 8 sentences (split
   on terminal ``.!?`` followed by whitespace + uppercase), flush the
   first 8 joined together.
3. Length cap: if the buffer exceeds 1500 chars, flush the first
   <= 1200 chars at the closest sentence boundary, otherwise flush
   the first 1200 chars verbatim.

Cancellation is checked at four points: entry to ``start()``, before
processing each delta, after the stream loop exits but before the tail
flush, and after the tail flush before the EoT emission. Any cancel
hit triggers an immediate ``return`` (no further events, no tail flush,
no EoT) so a cancelled turn cannot leak segments or terminators into
the workspace.
"""
from __future__ import annotations

import abc
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, TYPE_CHECKING, Tuple

from .types import EndOfThinking, ThinkerEvent, ThinkerSegment, TurnId

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .workspace import WorkspaceStore


class ThinkerInterface(abc.ABC):
    """Self-driven; only start/cancel from outside (§6.4)."""

    @abc.abstractmethod
    def start(self, turn_id: TurnId, prompt: str) -> AsyncIterator[ThinkerEvent]:
        """Drive a single streaming reasoning call. The model's hidden
        chain-of-thought arrives via ``reasoning_content`` and is split
        into ``ThinkerSegment`` events at paragraph / 8-sentence /
        length-cap boundaries. The model's visible ``content`` (final
        boxed answer) accumulates in parallel and is delivered as the
        ``EndOfThinking.final_summary`` once the stream completes
        naturally. ``thinking_budget`` (passed to the LLM client) caps
        the hidden reasoning token count; if hit, the model wraps up
        and visible content still streams normally. The only way no
        ``EndOfThinking`` is emitted is via cancellation.
        """
        ...

    @abc.abstractmethod
    def cancel(self, turn_id: TurnId) -> None:
        """Cancel the turn if running. Idempotent; no-op if turn already finished."""
        ...


class DashScopeThinker(ThinkerInterface):
    """Single-call streaming Thinker.

    Despite the historical name, this works against any
    ``LLMClient`` implementation (DashScope native or OpenAI-compatible
    via vLLM / DashScope's compat endpoint). The name is preserved to
    keep the public API stable across the Phase 1 refactor.
    """

    _PARAGRAPH_BREAK = "\n\n"
    _SENTENCE_FLUSH_THRESHOLD = 8
    _LENGTH_CAP_HARD = 1500
    _LENGTH_CAP_TARGET = 1200
    # Sentence boundary: ``.!?`` followed by whitespace + an uppercase ASCII
    # letter. Known limits (acceptable for AIME-style English math):
    #   - splits after abbreviations like "Dr. Smith" (rare in math prose).
    #   - decimals like "0.5" lack the trailing whitespace + upper, so the
    #     regex correctly skips them.
    #   - CJK punctuation is not handled; current fixtures are English-only.
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        ws: "WorkspaceStore",
        prompts_dir: Path = Path("prompts"),
        thinking_model: str = "qwen-plus-2025-09-11",
        enable_thinking: bool = True,
        thinking_budget: int = 16000,
        llm_client: "Optional[LLMClient]" = None,
    ) -> None:
        """``enable_thinking`` activates the model's deep-thinking
        path. The model reasons internally and the hidden trace is
        streamed back to us via ``reasoning_content``; we split that
        trace into voiceable chunks. Visible ``content`` is collected
        in parallel and reported in ``EndOfThinking.final_summary``.

        ``thinking_budget`` caps the hidden reasoning token count for
        the call (default 16000, matching the AIME-2026 single-call
        baseline). This is the only upstream cap on stream length; the
        Thinker itself does not impose a chunk-count limit.

        ``llm_client`` (default None) selects the streaming chat
        backend. None falls back to ``DashScopeClient()`` for backward
        compat. Pass an ``OpenAIClient`` to target a local vLLM server
        or any OpenAI-compatible endpoint.
        """
        from .llm_client import DashScopeClient

        self._ws = ws
        self._prompts_dir = Path(prompts_dir)
        self._thinking_model = thinking_model
        self._enable_thinking = bool(enable_thinking)
        self._thinking_budget = int(thinking_budget)
        self._prompt = (self._prompts_dir / "thinker.md").read_text(encoding="utf-8")
        self._client = llm_client if llm_client is not None else DashScopeClient()
        self._cancelled: Set[TurnId] = set()
        self._running: Set[TurnId] = set()

    async def start(self, turn_id: TurnId, prompt: str) -> AsyncIterator[ThinkerEvent]:
        """Single-call streaming reasoning loop (see ThinkerInterface.start)."""
        if turn_id in self._cancelled:
            return

        self._running.add(turn_id)
        seq = 0
        reasoning_buffer = ""
        # ``content_buffer`` accumulates the FULL post-</think> content
        # for ``EndOfThinking.final_summary`` (the talker hard_final
        # reads this as the canonical answer string). We additionally
        # stream chunks out of a sliding window so hard_mid talker
        # dispatches can speak the model's polished answer prose as
        # it arrives, instead of waiting silently for the full
        # ~5000-token "Step 1 / Step 2 / boxed answer" rewrite the
        # model sometimes produces post-thinking. ``content_emit_offset``
        # tracks how much of ``content_buffer`` has already been
        # flushed via ``_split_chunk``.
        content_buffer = ""
        content_emit_offset = 0
        first_token_recorded = False

        try:
            messages = self._build_messages(prompt)
            async for delta in self._client.stream_chat_with_reasoning(
                model=self._thinking_model,
                messages=messages,
                enable_thinking=self._enable_thinking,
                thinking_budget=self._thinking_budget,
            ):
                if turn_id in self._cancelled:
                    return

                if not first_token_recorded:
                    first_token_recorded = True
                    # Defensive: tests can drive the client without going
                    # through LLMBundle.register_turn, in which case
                    # ws.now() raises. The event is purely diagnostic --
                    # skip silently if no anchor.
                    try:
                        ts_now = self._ws.now(turn_id)
                    except KeyError:
                        ts_now = None
                    if ts_now is not None:
                        self._ws.record_event(
                            "thinker_first_token",
                            {"turn_id": turn_id, "ts": ts_now},
                        )

                if delta.reasoning_content:
                    reasoning_buffer += delta.reasoning_content
                    while True:
                        chunk_text, reasoning_buffer = self._split_chunk(
                            reasoning_buffer
                        )
                        if chunk_text is None:
                            break
                        yield ThinkerSegment(
                            turn_id=turn_id,
                            seq=seq,
                            content=chunk_text,
                            created_at=self._ws.now(turn_id),
                        )
                        seq += 1

                if delta.content:
                    content_buffer += delta.content
                    # Stream chunks out of the un-emitted tail of
                    # content_buffer using the same paragraph /
                    # sentence-count / length-cap policies that
                    # govern reasoning_content above. final_summary
                    # still receives the FULL content at EoT.
                    pending = content_buffer[content_emit_offset:]
                    while True:
                        chunk_text, pending = self._split_chunk(pending)
                        if chunk_text is None:
                            break
                        yield ThinkerSegment(
                            turn_id=turn_id,
                            seq=seq,
                            content=chunk_text,
                            created_at=self._ws.now(turn_id),
                        )
                        seq += 1
                    content_emit_offset = len(content_buffer) - len(pending)

            # Stream ended naturally.
            if turn_id in self._cancelled:
                return

            tail = reasoning_buffer.strip()
            if tail:
                yield ThinkerSegment(
                    turn_id=turn_id,
                    seq=seq,
                    content=tail,
                    created_at=self._ws.now(turn_id),
                )
                seq += 1
                if turn_id in self._cancelled:
                    return

            # Flush the trailing content tail too -- the model's last
            # paragraph (often the ``\boxed{N}`` line) wouldn't have
            # met any of the split policies (no terminating ``\n\n``
            # and < 1500 chars), so without this it gets buried in
            # final_summary and the talker has nothing fresh to say
            # between the last hard_mid and hard_final.
            content_tail = content_buffer[content_emit_offset:].strip()
            if content_tail:
                yield ThinkerSegment(
                    turn_id=turn_id,
                    seq=seq,
                    content=content_tail,
                    created_at=self._ws.now(turn_id),
                )
                seq += 1
                if turn_id in self._cancelled:
                    return

            yield EndOfThinking(
                turn_id=turn_id,
                final_summary=content_buffer.strip(),
                total_segments=seq,
                created_at=self._ws.now(turn_id),
                reason="natural",
            )
        finally:
            self._running.discard(turn_id)

    def cancel(self, turn_id: TurnId) -> None:
        """Cancel the turn if running. Idempotent; no-op if turn already finished."""
        self._cancelled.add(turn_id)

    def _build_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self._prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def _split_chunk(self, buffer: str) -> Tuple[Optional[str], str]:
        """Try to flush one chunk from the head of ``buffer``.

        Returns ``(chunk_text, remaining)`` if a chunk is ready, else
        ``(None, buffer)``. Order of policies:

        1. Paragraph break: flush up to and including the first
           ``\\n\\n``.
        2. Sentence count: if the buffer holds >= 8 sentences, flush
           the first 8 (joined back together).
        3. Length cap: if ``len(buffer) > 1500``, flush the first
           <= 1200 chars at the closest sentence boundary; if no
           sentence boundary exists in the first 1200 chars, hard-cut
           at 1200.
        4. Otherwise: not yet flushable.
        """
        if not buffer:
            return None, buffer

        # Policy 1: paragraph break.
        para_idx = buffer.find(self._PARAGRAPH_BREAK)
        if para_idx >= 0:
            cut = para_idx + len(self._PARAGRAPH_BREAK)
            chunk = buffer[:cut]
            stripped = chunk.strip()
            if stripped:
                return stripped, buffer[cut:]
            # Empty paragraph head (just whitespace); drop and recurse on rest.
            return self._split_chunk(buffer[cut:])

        # Policy 2: sentence count.
        sentences = self._SENTENCE_SPLIT_RE.split(buffer)
        if len(sentences) >= self._SENTENCE_FLUSH_THRESHOLD + 1:
            # The split keeps the last (possibly incomplete) tail as
            # the final element; we keep that in the buffer and flush
            # the first N complete sentences.
            head = sentences[: self._SENTENCE_FLUSH_THRESHOLD]
            tail_sentences = sentences[self._SENTENCE_FLUSH_THRESHOLD :]
            chunk = self._rejoin_sentences(buffer, head)
            if chunk is not None:
                rest_start = len(chunk)
                # Skip whitespace separating the flushed chunk from the
                # remainder so the next buffer doesn't lead with " ".
                while rest_start < len(buffer) and buffer[rest_start].isspace():
                    rest_start += 1
                stripped = chunk.strip()
                if stripped:
                    return stripped, buffer[rest_start:]
                # Defensive fallback: treat the original tail-sentences as
                # the new buffer if we somehow produced an empty chunk.
                _ = tail_sentences  # noqa: F841

        # Policy 3: length cap.
        if len(buffer) > self._LENGTH_CAP_HARD:
            head_window = buffer[: self._LENGTH_CAP_TARGET]
            cut = self._last_sentence_boundary(head_window)
            if cut <= 0:
                cut = self._LENGTH_CAP_TARGET
            chunk = buffer[:cut]
            stripped = chunk.strip()
            rest_start = cut
            while rest_start < len(buffer) and buffer[rest_start].isspace():
                rest_start += 1
            if stripped:
                return stripped, buffer[rest_start:]

        return None, buffer

    def _rejoin_sentences(self, buffer: str, head: List[str]) -> Optional[str]:
        """Reconstruct the prefix of ``buffer`` that contains exactly
        the first ``len(head)`` sentences (i.e. the first ``len(head)``
        terminator-followed-by-whitespace+upper boundaries).

        Returns the prefix slice of ``buffer`` (preserving original
        whitespace) up to and including the last sentence in ``head``,
        or ``None`` if the boundary cannot be located (defensive).
        """
        match_iter = self._SENTENCE_SPLIT_RE.finditer(buffer)
        end = 0
        for i, match in enumerate(match_iter, start=1):
            if i == len(head):
                end = match.start()
                break
        if end <= 0:
            return None
        return buffer[:end]

    def _last_sentence_boundary(self, text: str) -> int:
        """Return the index just past the last ``.!?`` followed by
        whitespace inside ``text``, or ``-1`` if none found. Used by
        the length-cap policy to prefer cutting at sentence ends."""
        boundary = -1
        for match in re.finditer(r"[.!?]\s+", text):
            boundary = match.end()
        return boundary
