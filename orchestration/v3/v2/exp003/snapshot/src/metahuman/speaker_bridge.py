"""SpeakerBridge: orchestrator-layer connector between LLM and TTS (§6.6, §8.5).

Position in the architecture::

    user audio ── STT ── LLMInterface ── (TalkerChunk stream) ──┐
                              ▲                                  │
                              │ notify_tts_*                      │
                              │                                   ▼
                              └────── SpeakerBridge ◀────── TTSInterface
                                          │   ▲              (synthesize_text,
                                          │   │ backlog       backlog_seconds)
                                          ▼   │
                                     CLI / consumer
                                     (sees same TalkerChunks
                                      flow through)

Why a bridge: LLM and TTS interfaces stay strictly decoupled (OLV-style); the
bridge is the single place that knows both and wires their streams together.
LLM yields TalkerChunks carrying ``is_last_in_segment`` so the bridge can
detect dispatch boundaries from the stream alone — no observer pattern, no
shared types between LLM and TTS modules.

What it does:

* **Pass-through chunks** — ``run`` is an async generator: every TalkerChunk
  read from the LLM stream is forwarded unchanged after the bridge has done
  its side effects, so any downstream consumer (CLI printer, file writer,
  test harness) sees the same chunks.
* **Drive synthesis** — for each chunk's text, the bridge calls
  ``tts.async_synthesize_text(text, turn_anchor_monotonic)`` and tracks the
  resulting audio duration on a virtual playback queue (``§6.6`` model).
* **Per-dispatch playback bookkeeping** — when a chunk's ``talker_seg_seq``
  first appears, the bridge records that dispatch's playback-start moment;
  when ``is_last_in_segment`` is True, it records the playback-finish moment.
  Both moments are computed from the bridge's virtual playback queue and
  fed back to the LLM via ``notify_tts_first_audio`` / ``notify_tts_finished_at``.
* **Backlog poller** — a sibling task periodically samples
  ``tts.backlog_seconds()`` and pushes the reading via
  ``llm.notify_tts_backlog(...)``; this is what gates hard_mid dispatch.

Why simulated playback (continuous-playback assumption from §6.6) rather
than driving a real audio device:

* §8.5 measures controller-induced gaps (threshold tuning), not OS
  scheduling jitter; the simulated queue isolates the signal.
* Reproducible: re-running a JSONL log gives identical metrics.
* Doesn't depend on speakers being available on srv73 / CI.

The bridge is part of the orchestrator layer, not part of LLMBundle. The
default driver (smoke scripts, future CLI) constructs an LLMInterface, a
TTSInterface, and a SpeakerBridge over both. LLMBundle never imports this
module.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, NamedTuple, Optional, Set, Tuple

from .llm import LLMInterface
from .tts import TTSInterface
from .types import TalkerChunk, Timestamp, TurnId

logger = logging.getLogger(__name__)


class PlaybackWindow(NamedTuple):
    """Pair of turn-relative timestamps describing one synthesized chunk's
    place on the virtual playback queue.

    Named so callers cannot accidentally swap ``started_at`` and
    ``finished_at`` (they share the same type and are easy to confuse).
    """
    started_at: Timestamp
    finished_at: Timestamp


@dataclass
class _DispatchPlayback:
    """Internal: bookkeeping for one in-flight dispatch's audio timeline.

    Fields are the bridge's view; not persisted. The bridge converts these
    to LLM-side notifications when boundaries are seen.
    """
    talker_seg_seq: int
    audio_started_at: Optional[Timestamp]   # turn-relative; first chunk's playback start
    audio_finished_at: Optional[Timestamp]  # turn-relative; last chunk's playback end
    first_chunk_seen: bool
    # If real playback timing differs from synth time (it does: synth is faster
    # than playback), the bridge tracks ``play_until_at_synth_finish`` so the
    # last chunk's playback end can be computed without waiting wall-clock.


class SpeakerBridge:
    """Decouples LLM and TTS at the orchestrator layer (§6.6, §8.5).

    Lifecycle::

        bridge = SpeakerBridge(llm, tts, turn_anchor_monotonic, ...)
        await bridge.start()
        async for chunk in bridge.run(llm.handle_user_text(...)):
            ...  # CLI / file consumer
        await bridge.stop()

    ``start``/``stop`` manage the backlog-poller task. ``run`` is the
    pass-through async generator that drives TTS. Multiple turns within a
    dialog reuse one bridge; the per-turn anchor is supplied per ``run``
    call (TTS uses it to rebase TTSResult timestamps against the turn's
    monotonic zero).
    """

    def __init__(
        self,
        llm: LLMInterface,
        tts: TTSInterface,
        *,
        backlog_poll_interval: float = 0.05,
        max_in_flight_synth: int = 4,
    ) -> None:
        """Build a bridge over the given LLM and TTS instances.

        ``backlog_poll_interval`` is the period (seconds) between samples of
        ``tts.backlog_seconds()``; default 50 ms balances responsiveness
        against polling overhead. The §8.5 ablation may sweep this.

        ``max_in_flight_synth`` (default 4) caps the number of TTS
        synthesis tasks that ``run`` keeps in flight at once. ``run``
        kicks off synth as soon as a chunk arrives from the LLM and
        delivers results to the consumer in FIFO order; this lets
        sentence N's synth overlap with sentence N+1's LLM streaming.
        Set to 1 to fall back to fully serial synth (the pre-pipeline
        behaviour). Set higher (e.g. 8) when the TTS backend is
        thread-safe and CPU/GPU has headroom.
        """
        if max_in_flight_synth < 1:
            raise ValueError("max_in_flight_synth must be >= 1")
        self._llm = llm
        self._tts = tts
        self._backlog_poll_interval = backlog_poll_interval
        self._max_in_flight_synth = int(max_in_flight_synth)
        self._poller_task: Optional[asyncio.Task] = None
        self._running: bool = False
        # Per-(turn_id, talker_seg_seq) playback bookkeeping. Cleared per turn
        # by ``run`` when the chunk stream ends. Keying by the pair lets the
        # bridge tolerate re-runs and concurrent turns (defensive; current
        # design has only one turn at a time).
        self._dispatches: Dict[Tuple[TurnId, int], _DispatchPlayback] = {}
        self._play_until_monotonic: Optional[float] = None

    async def start(self) -> None:
        """Start the backlog-poller background task.

        Idempotent: calling ``start`` twice is a no-op. Must be called before
        ``run`` so the controller actually receives backlog readings during
        the turn.
        """
        if self._running and self._poller_task is not None and not self._poller_task.done():
            return
        self._running = True
        self._poller_task = asyncio.create_task(self._poll_backlog())

    async def stop(self) -> None:
        """Cancel and await the backlog-poller task; clear in-flight state.

        Idempotent. Safe to call from finally blocks; tolerates ``stop``
        before ``start``.
        """
        self._running = False
        task = self._poller_task
        self._poller_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._dispatches.clear()
        self._play_until_monotonic = None

    async def run(
        self,
        chunks: AsyncIterator[TalkerChunk],
        *,
        turn_anchor_monotonic: float,
    ) -> AsyncIterator[TalkerChunk]:
        """Drive TTS for one turn and yield each chunk back to the caller.

        For each ``chunk`` from the upstream LLM iterator:

        1. Synthesize ``chunk.text`` via ``tts.async_synthesize_text``
           (skipped for synthetic empty-text boundary chunks).
        2. Compute playback start/finish on the virtual queue
           (``max(now, last_play_until)`` is the start; start + duration is
           the finish; ``last_play_until`` advances).
        3. If this chunk's ``talker_seg_seq`` is new for this turn → record
           dispatch start and call
           ``self._llm.notify_tts_first_audio(chunk.turn_id,
           chunk.talker_seg_seq, playback_started_at)``.
        4. If ``chunk.is_last_in_segment`` → record dispatch finish and
           call ``self._llm.notify_tts_finished_at(chunk.turn_id,
           chunk.talker_seg_seq, playback_finished_at)``.
        5. Yield the chunk unchanged (pass-through; consumer continues
           regardless of what the bridge did).

        Lifecycle (R1 review item M3): bookkeeping for per-dispatch keys is
        scoped to one ``run()`` call. A local ``seen_keys`` set tracks the
        ``(turn_id, talker_seg_seq)`` pairs the bridge has written into
        ``self._dispatches`` during this iteration. On exit (normal or via
        exception / cancellation) the ``finally`` block removes only those
        keys from ``self._dispatches`` — multi-turn reuse of the bridge is
        preserved while leaks from in-flight failures are not.

        Synthesis is **pipelined**: as soon as a chunk arrives from the
        LLM, ``run`` kicks off its TTS synth as a background task and
        keeps reading. Up to ``max_in_flight_synth`` (constructor
        parameter) tasks may be in flight at once. Results are awaited
        and processed in FIFO order so the playback queue stays in
        chunk order. The downstream consumer's "first chunk visible"
        moment therefore reflects synth latency for the first chunk
        only — subsequent chunks usually arrive already-synthesised
        because their synth was overlapped with earlier chunks'
        playback. For accurate text-TTFT, smoke / metrics must still
        read ``chunk.produced_at`` (stamped by the controller before
        the chunk entered the queue) rather than wall-clock at the
        receive side.

        ``turn_anchor_monotonic`` is forwarded to TTS so each TTSResult's
        ``started_at`` / ``finished_at`` come back turn-relative.
        """
        seen_keys: Set[Tuple[TurnId, int]] = set()
        # FIFO queue of (chunk, synth_future_or_None) pairs. None
        # marks a synthetic empty-text chunk that needs no synth.
        # max_in_flight_synth + 1 capacity lets the producer push the
        # next item while the consumer is processing the current one,
        # without exceeding the configured in-flight cap.
        pipeline: asyncio.Queue = asyncio.Queue(
            maxsize=self._max_in_flight_synth + 1
        )
        sentinel: Tuple[Optional[TalkerChunk], Optional[asyncio.Task]] = (None, None)
        producer_error: List[BaseException] = []

        async def producer() -> None:
            cancelled = False
            try:
                async for chunk in chunks:
                    if chunk.text:
                        synth_task = asyncio.create_task(
                            self._tts.async_synthesize_text(
                                chunk.text,
                                turn_anchor_monotonic,
                            )
                        )
                    else:
                        synth_task = None
                    await pipeline.put((chunk, synth_task))
            except asyncio.CancelledError:
                # Cancellation must propagate so an outer
                # asyncio.wait_for / task.cancel() can actually bound
                # this work. CancelledError is NOT a producer error;
                # do not append it to ``producer_error`` or the
                # consumer would re-raise it as if real failure.
                cancelled = True
                raise
            except Exception as exc:
                producer_error.append(exc)
            finally:
                if cancelled:
                    # On cancellation the consumer is unwinding too;
                    # do best-effort sentinel insertion that cannot
                    # itself block (queue may be full and the consumer
                    # has stopped pulling).
                    try:
                        pipeline.put_nowait(sentinel)
                    except asyncio.QueueFull:
                        pass
                else:
                    # Normal completion or non-cancellation producer
                    # error: the consumer is still draining and needs
                    # the sentinel to break out and re-raise from
                    # ``producer_error``.
                    await pipeline.put(sentinel)

        producer_task = asyncio.create_task(producer())
        in_flight: List[asyncio.Task] = []
        try:
            while True:
                item = await pipeline.get()
                if item is sentinel:
                    break
                chunk, synth_task = item
                if synth_task is not None:
                    in_flight.append(synth_task)

                key = (chunk.turn_id, chunk.talker_seg_seq)
                dispatch = self._dispatches.get(key)
                if dispatch is None:
                    dispatch = _DispatchPlayback(
                        talker_seg_seq=chunk.talker_seg_seq,
                        audio_started_at=None,
                        audio_finished_at=None,
                        first_chunk_seen=False,
                    )
                    self._dispatches[key] = dispatch
                    seen_keys.add(key)

                if synth_task is not None:
                    result = await synth_task
                    if synth_task in in_flight:
                        in_flight.remove(synth_task)
                    # Only notify first_byte for the FIRST chunk of a
                    # dispatch — subsequent chunks have their own synth
                    # but are not "the dispatch's first byte". Mirrors
                    # the first_chunk_seen guard used for first_audio
                    # below.
                    if result.first_byte_at is not None and not dispatch.first_chunk_seen:
                        try:
                            self._llm.notify_tts_first_byte(
                                chunk.turn_id,
                                chunk.talker_seg_seq,
                                result.first_byte_at,
                            )
                        except Exception:
                            logger.exception(
                                "notify_tts_first_byte failed: turn=%s seq=%s",
                                chunk.turn_id,
                                chunk.talker_seg_seq,
                            )
                    window = self._compute_playback_window(
                        synth_finished_at_monotonic=turn_anchor_monotonic + result.finished_at,
                        synth_duration_s=result.duration_s,
                        turn_anchor_monotonic=turn_anchor_monotonic,
                    )
                else:
                    play_until = self._play_until_monotonic
                    if play_until is None:
                        play_until = turn_anchor_monotonic
                    tail = play_until - turn_anchor_monotonic
                    window = PlaybackWindow(started_at=tail, finished_at=tail)

                if not dispatch.first_chunk_seen:
                    dispatch.first_chunk_seen = True
                    dispatch.audio_started_at = window.started_at
                    try:
                        self._llm.notify_tts_first_audio(
                            chunk.turn_id,
                            chunk.talker_seg_seq,
                            window.started_at,
                        )
                    except Exception:
                        logger.exception(
                            "notify_tts_first_audio failed: turn=%s seq=%s",
                            chunk.turn_id,
                            chunk.talker_seg_seq,
                        )

                if chunk.is_last_in_segment:
                    dispatch.audio_finished_at = window.finished_at
                    try:
                        self._llm.notify_tts_finished_at(
                            chunk.turn_id,
                            chunk.talker_seg_seq,
                            window.finished_at,
                        )
                    except Exception:
                        logger.exception(
                            "notify_tts_finished_at failed: turn=%s seq=%s",
                            chunk.turn_id,
                            chunk.talker_seg_seq,
                        )

                yield chunk
            # Surface a producer-side exception (e.g. upstream LLM
            # error) once the consumer has processed everything that
            # was queued before the failure. The sentinel above
            # already broke the loop normally; this re-raise turns a
            # silent producer failure into a visible one.
            if producer_error:
                raise producer_error[0]
        finally:
            # Cancel any synth tasks that never reached the consumer
            # (producer was interrupted, consumer raised, etc.). Drain
            # cancelled tasks so they do not warn-on-destruction.
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                # ``gather(..., return_exceptions=True)`` collects each
                # child's outcome (including CancelledError raised
                # because we just cancelled it) as a return value
                # rather than re-raising, while still propagating
                # cancellation if THIS coroutine is cancelled while
                # gather is awaiting. The previous per-task
                # ``except (CancelledError, Exception): pass`` blanket-
                # swallowed cancellation regardless of source, which
                # made an outer ``asyncio.wait_for`` timeout silently
                # ineffective.
                await asyncio.gather(*in_flight, return_exceptions=True)
            in_flight.clear()
            if not producer_task.done():
                producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
            for key in seen_keys:
                self._dispatches.pop(key, None)

    async def _poll_backlog(self) -> None:
        """Loop: every ``backlog_poll_interval`` seconds, sample
        ``tts.backlog_seconds()`` and push to ``llm.notify_tts_backlog``.

        Exits when ``self._running`` flips False or the task is cancelled.
        Errors from the LLM hook are logged but do not stop the poller —
        a single notify call failing should not silence the whole backlog
        feedback loop.
        """
        while self._running:
            try:
                backlog = self._tts.backlog_seconds()
                self._llm.notify_tts_backlog(backlog)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("notify_tts_backlog failed")
            await asyncio.sleep(self._backlog_poll_interval)

    def reset_playback_cursor(self) -> None:
        """Drop the virtual playback cursor so the next synth's
        ``_compute_playback_window`` treats the queue as empty.

        Used by experimental drivers AFTER ``wait_until_drained`` to
        guarantee that the next turn's ``tts_first_audio_at`` reflects
        only the new turn's synth pipeline, not leftover audio from
        the previous turn. The drain wait can return early on its
        timeout cap; without an explicit cursor reset, leftover queue
        carries over and the next turn's ttfa is inflated. This is a
        deliberate experimental simplification — production deployment
        DOES need to honour the queue (the user actually waits for
        playback), but the §8.6 / §8.7 ablations measure per-turn
        synth latency in isolation.

        Also calls ``self._tts.reset()`` to drop any accumulated
        playback cursor on the TTS adapter side. Without this the
        adapter's ``backlog_seconds()`` keeps reporting future-playback
        time inherited from prior turns, which hangs the controller's
        ``_close_turn_if_done`` (requires exact ``backlog == 0``) for
        as long as it takes wall clock to catch up — observed in §8.5
        cell #4 turn-4 as a 2.5-hour stuck turn.
        """
        self._play_until_monotonic = None
        self._tts.reset()

    async def wait_until_drained(
        self,
        *,
        poll_interval: float = 0.1,
        timeout: Optional[float] = None,
    ) -> None:
        """Block until the TTS playback queue is empty.

        Used by experimental drivers to enforce a "next turn does not
        start until the previous turn's audio has fully played out"
        protocol — without this, consecutive turns share the
        ``_play_until_monotonic`` cursor and the second turn's
        ``tts_first_audio_at`` inherits a queue-wait component from the
        first turn's audio. That queue-wait is not part of the
        turn-under-test's TTFA in the experimental sense, so realistic
        per-turn TTFA is measured only when each turn starts with an
        empty queue.

        Polls ``self._tts.backlog_seconds()`` and sleeps for the
        reported backlog (capped at ``poll_interval``) until backlog
        reaches zero. ``timeout`` (seconds) caps the total wait so a
        misbehaving TTS adapter cannot hang the experiment forever;
        on timeout the method returns without raising — the caller
        proceeds to the next turn knowing the queue may not be fully
        drained.
        """
        started_at = time.monotonic()
        while True:
            backlog = self._tts.backlog_seconds()
            if backlog <= 0:
                return
            if timeout is not None and time.monotonic() - started_at >= timeout:
                logger.warning(
                    "wait_until_drained timed out with backlog=%.3fs after %.3fs",
                    backlog, timeout,
                )
                return
            await asyncio.sleep(min(backlog, poll_interval))

    def _compute_playback_window(
        self,
        synth_finished_at_monotonic: float,
        synth_duration_s: float,
        turn_anchor_monotonic: float,
    ) -> PlaybackWindow:
        """Pure helper: convert one synth result into a PlaybackWindow with
        turn-relative ``started_at`` and ``finished_at``.

        Uses an internal ``_play_until_monotonic`` cursor (raw monotonic).
        ``playback_started = max(synth_finished, _play_until)``.
        ``playback_finished = playback_started + duration``.
        Advances the cursor to playback_finished.

        Kept separate from ``run`` so the timing math is unit-testable
        without mocking the full async generator. The cursor lives on
        ``self`` so successive calls accumulate.
        """
        previous_tail = self._play_until_monotonic
        if previous_tail is None:
            previous_tail = synth_finished_at_monotonic
        playback_started = max(synth_finished_at_monotonic, previous_tail)
        playback_finished = playback_started + synth_duration_s
        self._play_until_monotonic = playback_finished
        return PlaybackWindow(
            started_at=playback_started - turn_anchor_monotonic,
            finished_at=playback_finished - turn_anchor_monotonic,
        )
