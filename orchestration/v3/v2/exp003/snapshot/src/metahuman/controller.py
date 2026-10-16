"""Controller: event loop, state machine, talker dispatch (§6.7, §7)."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple, Union

from .router import RouterInterface
from .talker import TalkerInterface
from .thinker import ThinkerInterface
from .types import (
    ControlCause,
    ControlSignal,
    EndOfThinking,
    RouteDecision,
    SessionState,
    TalkerChunk,
    TalkerMode,
    TalkerSegment,
    ThinkerSegment,
    Timestamp,
    TurnId,
    UserUtterance,
)
from .workspace import WorkspaceStore

# Mid-dispatch tunables (§6.7). These are the project-wide defaults; each
# Controller instance can override them at construction time so the §8.5
# ablation can scan a parameter grid without monkeypatching globals.
#
# Defaults selected from the §8.5 R22 sweep (9-cell × 40-question full
# fixture, commit 4f8d10b + this commit): cell (b=5.0, m=1, c=0.0,
# budget=500) achieved gap_total_mean=0.60s (p95 2.05s), TTFA mean
# 2.07s, accuracy 0.742 — the only cell with all three of {gap≤1s,
# TTFA<2.5s, accuracy>0.7}. See sweep_results/r22_full/
# sweep_mid_results.csv for the full 9-row table.
#
# Why each value:
# * b=5.0: backlog ≥ ~one-segment-of-audio is enough for hard_mid to
#   pre-fill the queue; b=1.0 is too tight (gap mean > 6s in all
#   budget cells), b=10 has diminishing returns (in_flight gate
#   already saturates the queue).
# * m=1: R12 + R22 data both show min_backlog has weak signal; fixed
#   to 1 so any thinker segment can trigger hard_mid.
# * c=0.0: cooldown is a redundant second-level throttle (the in_flight
#   gate already serializes hard_mid). R22 4-cell pivot data showed
#   cooldown 0→2.5 made gap monotonically worse.
#
# Earlier defaults: R12 picked (1.0, 2, 0.0) on EdgeTTS + prompt-thinking;
# pre-R12 heuristic was (1.5, 1, 0.5).
MID_TRIGGER_TTS_BACKLOG_SECONDS: float = 5.0
MID_TRIGGER_MIN_BACKLOG: int = 1
MID_TRIGGER_COOLDOWN: float = 0.0

logger = logging.getLogger(__name__)

_TURN_END_SENTINEL = object()


@dataclass(frozen=True)
class TalkerStarted:
    turn_id: TurnId
    started_at: Timestamp


@dataclass(frozen=True)
class TalkerFinished:
    turn_id: TurnId
    mode: TalkerMode  # which mode this finished dispatch was; gates end-of-turn (§7.2)
    finished_at: Timestamp
    snapshot_seq: int


@dataclass(frozen=True)
class TalkerFailed:
    turn_id: TurnId
    error: str


@dataclass(frozen=True)
class TTSLowWater:
    turn_id: TurnId  # source turn at observation time; stale events from old turns must be dropped
    backlog_seconds: float
    observed_at: Timestamp


ControllerEvent = Union[
    UserUtterance,
    RouteDecision,
    ThinkerSegment,
    EndOfThinking,
    TalkerStarted,
    TalkerFinished,
    TalkerFailed,
    TTSLowWater,
]

_STATE_TO_MODE: Dict[SessionState, TalkerMode] = {
    SessionState.EASY: TalkerMode.EASY,
    SessionState.HARD_FIRST: TalkerMode.HARD_FIRST,
    SessionState.HARD_MID: TalkerMode.HARD_MID,
    SessionState.HARD_FINAL: TalkerMode.HARD_FINAL,
}


class Controller:
    """Orchestrates router/thinker/talker via an async event queue (§6.7)."""

    def __init__(
        self,
        ws: WorkspaceStore,
        router: RouterInterface,
        thinker: ThinkerInterface,
        talker: TalkerInterface,
        mid_trigger_tts_backlog_seconds: float = MID_TRIGGER_TTS_BACKLOG_SECONDS,
        mid_trigger_min_backlog: int = MID_TRIGGER_MIN_BACKLOG,
        mid_trigger_cooldown: float = MID_TRIGGER_COOLDOWN,
        enable_hard_first_bridge: bool = True,
        enable_hard_mid: bool = True,
    ) -> None:
        self._ws = ws
        self._router = router
        self._thinker = thinker
        self._talker = talker
        # Per-instance hard_mid trigger tunables; defaults from module-level
        # constants. Sweepable via Controller construction in §8.5 ablation.
        self._mid_trigger_tts_backlog_seconds = mid_trigger_tts_backlog_seconds
        self._mid_trigger_min_backlog = mid_trigger_min_backlog
        self._mid_trigger_cooldown = mid_trigger_cooldown
        self._enable_hard_first_bridge = enable_hard_first_bridge
        self._enable_hard_mid = enable_hard_mid
        self._events: asyncio.Queue = asyncio.Queue()
        self._active_turn: Optional[TurnId] = None
        self._first_hard_dispatched: Set[TurnId] = set()
        self._last_finished_at: Dict[TurnId, Timestamp] = {}
        # Per-turn flag: True once a TalkerFinished has arrived whose
        # mode is EASY or HARD_FINAL — i.e., the "final answer"
        # dispatch for this turn has played out the LLM side. Replaces
        # the older "look at the last segment in workspace.talker_history"
        # heuristic, which races: a hard_mid dispatch can be queued
        # AFTER the hard_final dispatch finishes (a stale
        # thinker_segment_ready signal still in the controller's event
        # queue) and end up appended last to talker_history. The "last"
        # mode then reads as HARD_MID and _maybe_end_turn refuses to
        # end the turn even though the terminal dispatch had finished
        # and the audio queue had drained — symptom: state stuck at
        # HARD_FINAL, controller_gate_drop events flood the log
        # forever (~87 k events observed in §8.6 R19 turn-1 stuck
        # case).
        self._terminal_dispatch_finished: Set[TurnId] = set()
        self._talker_supervisors: Dict[TurnId, List[asyncio.Task]] = {}
        self._turn_outputs: Dict[TurnId, asyncio.Queue] = {}
        self._thinker_task: Optional[asyncio.Task] = None
        self._classify_task: Optional[asyncio.Task] = None
        self._tts_backlog_seconds: float = 0.0
        # Per-turn talker dispatch sequence counter. Reserved synchronously
        # in _send_talker before the supervisor task is created so two
        # supervisors (e.g., a still-streaming hard_mid and an in-flight
        # hard_final) can never collide on TalkerSegment.seq.
        self._next_talker_seq: Dict[TurnId, int] = {}
        # Pending TTS playback timestamps for dispatches whose TalkerSegment
        # has not yet been appended. Bridge notifications can race ahead of
        # _record_talker_segment because chunks are emitted to the per-turn
        # queue before the segment record is written. Keyed by
        # (turn_id, talker_seg_seq); applied by _record_talker_segment.
        self._pending_tts_updates: Dict[
            Tuple[TurnId, int], Dict[str, Timestamp]
        ] = {}
        # Per-turn cancellation generation. Incremented by _cancel_turn so
        # supervisors that captured the old value can detect they belong to
        # an abandoned turn and refuse to write into workspace. Without this
        # guard, a supervisor that finishes its async iteration while
        # cancellation runs in another coroutine still has a synchronous
        # path between iteration-end and segment-record where asyncio
        # cancellation cannot preempt; it would otherwise persist a
        # TalkerSegment for a turn the rest of the system has discarded.
        self._turn_generation: Dict[TurnId, int] = {}
        # Set of turn ids that have been cancelled in this dialog. Once a
        # turn id appears here, late callbacks (TTS notifications, queued
        # control events) and accidental re-issue (a user-utterance with
        # the same turn id, a follow-up _send_talker) are all rejected.
        # Reusing a cancelled turn id is intentionally unsupported in this
        # dialog: the public chunk / notify / event APIs carry only
        # turn_id+seq, not generation, so accepting late callbacks for
        # "the same id, new issue" would conflate identities. LLMBundle
        # already enforces non-duplicate register_turn at the workspace
        # level; the controller mirrors that for direct callers.
        self._cancelled_turns: Set[TurnId] = set()

    async def run(self) -> None:
        """Main loop. Drains self.events; mutates ws; dispatches talker / thinker."""
        while True:
            ev = await self._events.get()
            try:
                await self._dispatch(ev)
            except Exception:
                logger.exception("controller event handler raised; continuing")
            finally:
                self._events.task_done()

    # ----- ingress -----
    async def on_user_utterance(self, u: UserUtterance) -> None:
        """Queue a UserUtterance event; downstream the router runs and a RouteDecision is queued.

        First-line guard: a utterance whose turn id is already in
        ``_cancelled_turns`` must NOT trigger any side effects, especially
        not the active-turn cancellation below. If a stale caller submits
        an already-cancelled turn id, dropping it here preserves the
        currently active (different) turn from being collateral damage.
        The stronger no-reuse contract on cancelled turn ids is enforced
        downstream in ``_handle_user_utterance`` as well, but the
        cancellation step lives only in this public entry, so the guard
        also has to live here.
        """
        if u.turn_id in self._cancelled_turns:
            logger.warning("dropping reused cancelled turn_id: turn=%s", u.turn_id)
            return
        if self._active_turn is not None and self._active_turn != u.turn_id:
            self._cancel_turn(self._active_turn)
        await self._events.put(u)

    def on_tts_backlog(self, backlog_seconds: float) -> None:
        """Convert TTS backlog reading into a TTSLowWater event when below threshold."""
        self._tts_backlog_seconds = backlog_seconds
        if self._active_turn is None:
            return
        ev = TTSLowWater(
            turn_id=self._active_turn,
            backlog_seconds=backlog_seconds,
            observed_at=self._ws.now(self._active_turn),
        )
        self._events.put_nowait(ev)

    def on_tts_first_audio(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_audio_at: Timestamp,
    ) -> None:
        """Record the first-audio playback time for ``(turn_id, talker_seg_seq)``.

        Two paths:

        * If a TalkerSegment for ``(turn_id, talker_seg_seq)`` already exists
          in workspace, update its ``tts_first_audio_at`` in place via
          ``ws.mutate`` (so the JSONL log captures the update).
        * Else stash the value in ``self._pending_tts_updates[
          (turn_id, talker_seg_seq)]`` under the key ``"tts_first_audio_at"``;
          ``_record_talker_segment`` will apply pending entries when the
          segment is appended.

        Idempotency / ordering guard (R1 review, item 5):
        * If the same value is written twice → ignore silently.
        * If a different value would overwrite an existing one → log a
          structured warning but DO NOT raise. Keeping the experiment
          running is more important than a single bookkeeping bug.
        """
        self._apply_tts_field(turn_id, talker_seg_seq, "tts_first_audio_at", first_audio_at)

    def on_tts_finished_at(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        finished_at: Timestamp,
    ) -> None:
        """Record playback-finished time for ``(turn_id, talker_seg_seq)``.

        Same two-path / idempotency / ordering-guard contract as
        ``on_tts_first_audio``. The bridge supplies a turn-relative
        timestamp computed from its virtual playback queue.
        """
        self._apply_tts_field(turn_id, talker_seg_seq, "tts_finished_at", finished_at)

    def on_tts_first_byte(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_byte_at: Timestamp,
    ) -> None:
        """Record the synth-service first-byte time for ``(turn_id,
        talker_seg_seq)``. Together with ``tts_first_audio_at`` (full
        decode), this lets §8.6 attribute the TTS phase of TTFA to
        service-call vs. decode time. Same two-path / idempotency /
        ordering-guard contract as ``on_tts_first_audio``.
        """
        self._apply_tts_field(turn_id, talker_seg_seq, "tts_first_byte_at", first_byte_at)

    def _apply_tts_field(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        field_name: str,
        value: Timestamp,
    ) -> None:
        """Pure helper used by the two on_tts_* hooks.

        Looks up TalkerSegment by (turn_id, talker_seg_seq):
        * Found and field is None: ``ws.mutate`` to set the field.
        * Found and field equals value: silent no-op (idempotent).
        * Found and field differs: ``logger.warning`` with both values; no
          mutation.
        * Not found: stash in ``self._pending_tts_updates[
          (turn_id, talker_seg_seq)][field_name] = value``; subsequent
          conflicting writes for the same key follow the same warn-but-keep
          policy applied during pending-flush at segment-record time.
        """
        if field_name not in ("tts_first_audio_at", "tts_finished_at", "tts_first_byte_at"):
            logger.warning(
                "dropping unsupported TTS field update: turn=%s seq=%s field=%s value=%s",
                turn_id, talker_seg_seq, field_name, value,
            )
            return

        # Cancelled turn: drop everything before any segment lookup so a
        # late callback cannot mutate an already-recorded segment that
        # belongs to a turn the rest of the system has discarded.
        if turn_id in self._cancelled_turns:
            logger.warning(
                "dropping TTS update for cancelled turn: turn=%s seq=%s field=%s value=%s",
                turn_id, talker_seg_seq, field_name, value,
            )
            return

        # Found-and-update path (works for the existing
        # test_notify_tts_first_audio_updates_talker_segment scenario).
        snapshot = self._ws.snapshot()
        for index, segment in enumerate(snapshot.talker_history):
            if segment.turn_id != turn_id or segment.seq != talker_seg_seq:
                continue

            existing = getattr(segment, field_name)
            if existing == value:
                return
            if existing is not None:
                logger.warning(
                    "tts field conflict: turn=%s seq=%s field=%s existing=%s incoming=%s",
                    turn_id, talker_seg_seq, field_name, existing, value,
                )
                return

            def update(w, idx=index, fname=field_name, val=value):
                w.talker_history[idx] = replace(w.talker_history[idx], **{fname: val})

            self._ws.mutate(update)
            return

        # Segment not yet present. Bridge raced ahead of _record_talker_segment.
        # Two cases:
        # * The turn is still active → stash the update; segment-record will
        #   apply it.
        # * The turn was cancelled → drop the update. _cancel_turn sets
        #   self._active_turn to None and bumps the generation; a late
        #   notification for that turn must NOT recreate per-turn state.
        if self._active_turn != turn_id:
            logger.warning(
                "dropping TTS update for inactive turn: turn=%s seq=%s field=%s value=%s",
                turn_id, talker_seg_seq, field_name, value,
            )
            return

        pending = self._pending_tts_updates.setdefault((turn_id, talker_seg_seq), {})
        existing = pending.get(field_name)
        if existing == value:
            return
        if existing is not None:
            logger.warning(
                "pending tts field conflict: turn=%s seq=%s field=%s existing=%s incoming=%s",
                turn_id, talker_seg_seq, field_name, existing, value,
            )
            return
        pending[field_name] = value

    def attach_turn_output(self, turn_id: TurnId, queue: asyncio.Queue) -> None:
        """Attach a per-turn text chunk queue for LLMBundle stream consumers."""
        if turn_id in self._cancelled_turns:
            logger.warning(
                "dropping output attachment for cancelled turn: turn=%s", turn_id,
            )
            queue.put_nowait(_TURN_END_SENTINEL)
            return
        self._turn_outputs[turn_id] = queue

    # ----- internal handlers -----
    async def _dispatch(self, ev: ControllerEvent) -> None:
        if isinstance(ev, UserUtterance):
            await self._handle_user_utterance(ev)
            return
        if isinstance(ev, RouteDecision):
            await self._handle_route_decision(ev)
            return
        if isinstance(ev, ThinkerSegment):
            await self._handle_thinker_segment(ev)
            return
        if isinstance(ev, EndOfThinking):
            await self._handle_end_of_thinking(ev)
            return
        if isinstance(ev, TalkerFinished):
            await self._handle_talker_finished(ev)
            return
        if isinstance(ev, TalkerFailed):
            await self._handle_talker_failed(ev)
            return
        if isinstance(ev, TTSLowWater):
            await self._handle_tts_low_water(ev)
            return
        if isinstance(ev, TalkerStarted):
            return
        logger.warning("dropping unknown controller event: %r", ev)

    async def _handle_user_utterance(self, u: UserUtterance) -> None:
        # Set active turn and persist the utterance immediately so any concurrent
        # event handlers see the new turn id; spawn router classify as a
        # background task so the event loop is not blocked while RouteLLM runs.
        # Downstream (thinker/talker) only fires once the RouteDecision lands
        # back on the queue, preserving the "router decides first" ordering.
        if u.turn_id in self._cancelled_turns:
            logger.warning("dropping reused cancelled turn_id: turn=%s", u.turn_id)
            return
        self._active_turn = u.turn_id
        self._ws.mutate(lambda w: w.user_history.append(u))
        if self._classify_task is not None and not self._classify_task.done():
            self._classify_task.cancel()
        self._classify_task = asyncio.create_task(self._run_classify(u))

    async def _run_classify(self, u: UserUtterance) -> None:
        try:
            decision = await self._router.classify(u)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("router classify failed for turn %s: %s", u.turn_id, exc)
            return
        await self._events.put(decision)

    async def _handle_route_decision(self, r: RouteDecision) -> None:
        """easy -> set_state(EASY) + dispatch easy. hard -> optionally
        dispatch hard_first, then run thinker and advance through the
        hard path."""
        if r.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping RouteDecision for cancelled turn: turn=%s", r.turn_id,
            )
            return
        if self._active_turn is None:
            self._active_turn = r.turn_id
        elif self._active_turn != r.turn_id:
            logger.warning(
                "dropping stale RouteDecision for turn %s while active turn is %s",
                r.turn_id,
                self._active_turn,
            )
            return

        if r.route == "easy":
            next_state = SessionState.EASY
        elif self._enable_hard_first_bridge:
            next_state = SessionState.HARD_FIRST
        else:
            next_state = SessionState.HARD_MID

        def record_route(w):
            w.route_history.append(r)
            w.state = next_state

        self._ws.mutate(record_route)

        if r.route == "easy":
            self._send_talker(r.turn_id, cause="route_easy", snapshot_seq=0)
            return

        if self._enable_hard_first_bridge:
            self._send_talker(r.turn_id, cause="route_hard_first", snapshot_seq=0)
        prompt = self._ws.current_user_text(r.turn_id) or ""
        self._thinker_task = asyncio.create_task(self._run_thinker(r.turn_id, prompt))

    async def _handle_thinker_segment(self, seg: ThinkerSegment) -> None:
        """Append to ws.thinker_history; possibly dispatch hard_mid per thresholds."""
        if seg.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping ThinkerSegment for cancelled turn: turn=%s", seg.turn_id,
            )
            return
        if seg.turn_id != self._active_turn:
            logger.warning(
                "dropping stale ThinkerSegment for turn %s while active turn is %s",
                seg.turn_id,
                self._active_turn,
            )
            return

        if self._has_thinker_segment(seg.turn_id, seg.seq):
            logger.warning(
                "dropping duplicate ThinkerSegment for turn %s seq %s",
                seg.turn_id,
                seg.seq,
            )
            return

        self._ws.mutate(lambda w: w.thinker_history.append(seg))
        if self._enable_hard_mid:
            self._maybe_dispatch_mid("thinker_segment_ready")

    async def _handle_end_of_thinking(self, eot: EndOfThinking) -> None:
        """Append EoT, set_state(HARD_FINAL), dispatch hard_final."""
        if eot.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping EndOfThinking for cancelled turn: turn=%s", eot.turn_id,
            )
            return
        if eot.turn_id != self._active_turn:
            logger.warning(
                "dropping stale EndOfThinking for turn %s while active turn is %s",
                eot.turn_id,
                self._active_turn,
            )
            return

        def record_eot(w):
            w.thinker_history.append(eot)
            w.state = SessionState.HARD_FINAL

        self._ws.mutate(record_eot)
        self._send_talker(
            eot.turn_id,
            cause="end_of_thinking",
            snapshot_seq=self._current_thinker_history_length(eot.turn_id),
        )

    async def _handle_talker_finished(self, tf: TalkerFinished) -> None:
        """Record talker completion; defer end-of-turn check to _maybe_end_turn.
        Stale tf.turn_id (not equal to active turn) must be dropped, and
        end-turn requires tf.mode in (EASY, HARD_FINAL) — see _maybe_end_turn."""
        if tf.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping TalkerFinished for cancelled turn: turn=%s", tf.turn_id,
            )
            return
        if tf.turn_id != self._active_turn:
            logger.warning(
                "dropping stale TalkerFinished for turn %s while active turn is %s",
                tf.turn_id,
                self._active_turn,
            )
            return
        self._last_finished_at[tf.turn_id] = tf.finished_at
        if tf.mode in (TalkerMode.EASY, TalkerMode.HARD_FINAL):
            # Persist the "terminal dispatch has finished" signal so
            # _maybe_end_turn can fire later from a TTSLowWater path
            # without having to re-derive mode from talker_history (see
            # field docstring on self._terminal_dispatch_finished).
            self._terminal_dispatch_finished.add(tf.turn_id)
        self._maybe_end_turn(tf.turn_id, tf.mode)

    async def _handle_talker_failed(self, tf: TalkerFailed) -> None:
        if tf.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping TalkerFailed for cancelled turn: turn=%s", tf.turn_id,
            )
            return
        if tf.turn_id != self._active_turn:
            logger.warning(
                "dropping stale TalkerFailed for turn %s while active turn is %s",
                tf.turn_id,
                self._active_turn,
            )
            return
        logger.error("talker failed for turn %s: %s", tf.turn_id, tf.error)

    async def _handle_tts_low_water(self, ev: TTSLowWater) -> None:
        """Drop ev when ev.turn_id is not the active turn. Otherwise: if there
        is unspoken thinker work, dispatch hard_mid; also call _maybe_end_turn
        since drain==0 is one of the end-turn triggers."""
        if ev.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping TTSLowWater for cancelled turn: turn=%s", ev.turn_id,
            )
            return
        if ev.turn_id != self._active_turn:
            logger.warning(
                "dropping stale TTSLowWater for turn %s while active turn is %s",
                ev.turn_id,
                self._active_turn,
            )
            return

        self._tts_backlog_seconds = ev.backlog_seconds
        if self._enable_hard_mid:
            self._maybe_dispatch_mid("tts_low_water")
        if ev.backlog_seconds == 0:
            self._maybe_end_turn(ev.turn_id, finished_mode=None)

    def _maybe_end_turn(self, turn_id: TurnId, finished_mode: Optional[TalkerMode]) -> None:
        """Set state=None iff all three §7.2 conditions hold:
        (1) a terminal-mode (EASY or HARD_FINAL) dispatch for this turn has
            finished — checked via ``finished_mode`` if the caller knows it,
            otherwise via the per-turn ``_terminal_dispatch_finished`` flag
            that ``_handle_talker_finished`` sets when a terminal dispatch
            completes;
        (2) TTS backlog == 0 — the user-side audio has fully drained;
        (3) ws.state in (EASY, HARD_FINAL) — global state-machine guard.

        Idempotent; safe to call from both _handle_talker_finished and
        _handle_tts_low_water. ``finished_mode`` is supplied by the
        TalkerFinished caller and is None on the tts-drain path. The
        flag-based recovery replaces an older "look up the latest
        TalkerSegment for turn_id" heuristic that races with stale
        thinker_segment_ready events: a hard_mid dispatch can be
        appended to ``talker_history`` AFTER the hard_final dispatch
        completes, so "latest mode" reports HARD_MID and the turn never
        ends (observed: §8.6 R19 turn-1 stuck with ~87 k
        controller_gate_drop events).
        """
        mode_ok = (
            finished_mode in (TalkerMode.EASY, TalkerMode.HARD_FINAL)
            or turn_id in self._terminal_dispatch_finished
        )
        if not mode_ok:
            return
        if self._tts_backlog_seconds != 0:
            return

        state = self._ws.snapshot().state
        if state not in (SessionState.EASY, SessionState.HARD_FINAL):
            return

        self._set_state(None)
        if self._active_turn == turn_id:
            self._active_turn = None
        self._first_hard_dispatched.discard(turn_id)
        self._last_finished_at.pop(turn_id, None)
        self._talker_supervisors.pop(turn_id, None)
        self._terminal_dispatch_finished.discard(turn_id)
        self._close_turn_output(turn_id)

    def _should_dispatch_mid(self) -> Optional[str]:
        """Tagged version of the hard_mid gate: returns ``None`` if dispatch
        is permitted, otherwise a short reason string identifying why the
        dispatch was blocked.

        Reasons (stable, used as the ``reason`` key in
        ``controller_gate_drop`` events for §8.5):
        * ``"no_active_turn"``
        * ``"state_not_hard_mid"``
        * ``"backlog_above_threshold"``
        * ``"thinker_backlog_below_minimum"``
        * ``"talker_in_flight"``    ← the count §8.5 actually wants
        * ``"within_cooldown"``

        ``_maybe_dispatch_mid`` records a structured ``controller_gate_drop``
        event with the reason whenever this returns non-None, so the metric
        is replayable from the workspace JSONL alone.

        The gate also enforces single-talker-at-a-time (the
        ``"talker_in_flight"`` reason): without it, multiple hard_mid
        dispatches pile up when thinker emits several segments within the
        cooldown — observed as near-duplicate output in real-DashScope
        smokes. hard_first / hard_final / easy are one-shot per turn and
        do not need this gate.
        """
        if self._active_turn is None:
            return "no_active_turn"
        if self._ws.snapshot().state != SessionState.HARD_MID:
            return "state_not_hard_mid"
        if self._tts_backlog_seconds >= self._mid_trigger_tts_backlog_seconds:
            return "backlog_above_threshold"
        if self._ws.thinker_backlog(self._active_turn) < self._mid_trigger_min_backlog:
            return "thinker_backlog_below_minimum"
        if self._has_in_flight_talker(self._active_turn):
            return "talker_in_flight"

        last_finished_at = self._last_finished_at.get(self._active_turn)
        if last_finished_at is None:
            return None
        if self._ws.now(self._active_turn) - last_finished_at < self._mid_trigger_cooldown:
            return "within_cooldown"
        return None

    def _has_in_flight_talker(self, turn_id: TurnId) -> bool:
        """True if any talker supervisor task for `turn_id` is still running.
        Also prunes done tasks from the registry so it doesn't grow unboundedly."""
        tasks = self._talker_supervisors.get(turn_id, [])
        live = [t for t in tasks if not t.done()]
        self._talker_supervisors[turn_id] = live
        return bool(live)

    def _set_state(self, state: Optional[SessionState]) -> None:
        """ws.mutate(lambda w: setattr(w, 'state', state))."""
        self._ws.mutate(lambda w: setattr(w, "state", state))

    def _has_unspoken_thinker(self) -> bool:
        """True if current turn has thinker events past the last consumed snapshot_seq."""
        if self._active_turn is None:
            return False
        return self._ws.thinker_backlog(self._active_turn) > 0

    def _maybe_dispatch_mid(self, cause: ControlCause) -> None:
        """Either dispatch a hard_mid or record a structured gate-drop event.

        On every call:
        * If ``_should_dispatch_mid()`` returns None: dispatch as before.
        * Otherwise: emit a ``controller_gate_drop`` event into the
          workspace JSONL with ``turn_id``, the trigger ``cause``, the
          gate ``reason`` returned by ``_should_dispatch_mid``, the
          observation timestamp, and a small snapshot of the gate
          inputs. The §8.5 metric ``mid_blocked_by_in_flight_count``
          filters these events by ``reason == "talker_in_flight"``.
        """
        reason = self._should_dispatch_mid()
        if reason is None:
            assert self._active_turn is not None
            self._send_talker(
                self._active_turn,
                cause=cause,
                snapshot_seq=self._current_thinker_history_length(self._active_turn),
            )
            return
        if self._active_turn is None:
            return
        turn_id = self._active_turn
        snapshot = self._ws.snapshot()
        state = snapshot.state
        tasks = self._talker_supervisors.get(turn_id, [])
        live_tasks = [task for task in tasks if not task.done()]
        self._talker_supervisors[turn_id] = live_tasks
        self._ws.record_event(
            "controller_gate_drop",
            {
                "turn_id": turn_id,
                "cause": cause,
                "reason": reason,
                "observed_at": self._ws.now(turn_id),
                "state": state.value if state is not None else None,
                "backlog_seconds": self._tts_backlog_seconds,
                "thinker_backlog": self._ws.thinker_backlog(turn_id),
                "in_flight_talker_count": len(live_tasks),
            },
        )

    def _reserve_talker_seq(self, turn_id: TurnId) -> int:
        """Hand out the next TalkerSegment.seq for ``turn_id`` synchronously.

        The counter lives on the controller (in the same coroutine as the
        rest of dispatch), not inside the per-supervisor task, so concurrent
        supervisors (e.g., hard_final dispatched while hard_mid is still
        streaming — the in-flight gate is hard_mid-only and does not
        prevent this overlap) can never read the same ``len(talker_history)``
        and pick the same seq.

        On first call for a turn, seeds the counter from the current
        ``talker_history`` to stay consistent with any segments persisted
        from earlier in the dialog.
        """
        next_seq = self._next_talker_seq.get(turn_id)
        if next_seq is None:
            next_seq = sum(
                1 for s in self._ws.snapshot().talker_history if s.turn_id == turn_id
            )
        self._next_talker_seq[turn_id] = next_seq + 1
        return next_seq

    def _send_talker(
        self,
        turn_id: TurnId,
        cause: ControlCause,
        snapshot_seq: int,
    ) -> None:
        if turn_id in self._cancelled_turns:
            logger.warning("dropping talker dispatch for cancelled turn: turn=%s", turn_id)
            return
        state = self._ws.snapshot().state
        if state is None:
            logger.warning("dropping talker dispatch for turn %s with no active state", turn_id)
            return

        mode = _STATE_TO_MODE[state]
        seg_seq = self._reserve_talker_seq(turn_id)
        turn_generation = self._turn_generation.get(turn_id, 0)
        signal = ControlSignal(
            turn_id=turn_id,
            triggered_at=self._ws.now(turn_id),
            cause=cause,
            snapshot_seq=snapshot_seq,
        )
        task = asyncio.create_task(
            self._supervise_talker(signal, mode, seg_seq, turn_generation)
        )
        self._talker_supervisors.setdefault(turn_id, []).append(task)

        if mode == TalkerMode.HARD_FIRST and turn_id not in self._first_hard_dispatched:
            self._first_hard_dispatched.add(turn_id)
            self._set_state(SessionState.HARD_MID)

    async def _supervise_talker(
        self,
        signal: ControlSignal,
        mode: TalkerMode,
        seg_seq: int,
        turn_generation: int,
    ) -> None:
        """Drain talker.dispatch, wrap each fragment as TalkerChunk on the
        per-turn queue, then record the segment + emit TalkerFinished.

        ``seg_seq`` is reserved synchronously by ``_send_talker`` before the
        supervisor task is scheduled (§6.7 in-flight gate is hard_mid-only;
        hard_final dispatched concurrently with a still-streaming hard_mid
        cannot count history-len safely on its own). The supervisor must
        not recompute the seq from ``talker_history``.

        ``turn_generation`` is captured at dispatch time. ``_cancel_turn``
        increments the per-turn generation; if the captured value is stale
        when this supervisor is about to mutate workspace, the segment-record
        path drops the write so a cancelled turn cannot leak history entries.

        Chunk wrapping: queue carries TalkerChunk(turn_id, talker_seg_seq,
        text, is_last_in_segment, produced_at) so downstream SpeakerBridge
        can identify dispatch boundaries and so smoke / metrics can read
        accurate text-TTFT from ``produced_at`` regardless of any
        downstream synthesis backpressure.

        Boundary marking via trailing sentinel: every chunk yielded by
        the talker is emitted immediately with ``is_last_in_segment=False``
        so the SpeakerBridge can kick off synth on it without waiting for
        a successor (this matters for first-audio latency: the previous
        one-chunk lookahead held the first sentence in a buffer until the
        second sentence arrived). After the talker stream ends — empty
        dispatch or not — a synthetic empty-text ``TalkerChunk`` with
        ``is_last_in_segment=True`` is pushed as the dispatch boundary.
        Mirrors Open-LLM-VTuber's ``conversation_utils`` pattern: each
        sentence is sent to TTS the moment it is produced, and the
        dispatch end is recognised by stream completion rather than a
        last-chunk flag carried by a real chunk.
        """
        started_at = signal.triggered_at
        chunks: List[str] = []
        queue = self._turn_outputs.get(signal.turn_id)

        def _emit(text: str, is_last: bool) -> None:
            if queue is None:
                return
            queue.put_nowait(
                TalkerChunk(
                    turn_id=signal.turn_id,
                    talker_seg_seq=seg_seq,
                    text=text,
                    is_last_in_segment=is_last,
                    produced_at=self._ws.now(signal.turn_id),
                )
            )

        try:
            async for chunk in self._talker.dispatch(signal, mode):
                chunks.append(chunk)
                _emit(chunk, is_last=False)

            _emit("", is_last=True)

            finished_at = self._ws.now(signal.turn_id)
            self._record_talker_segment(
                signal=signal,
                mode=mode,
                text="".join(chunks),
                started_at=started_at,
                finished_at=finished_at,
                seg_seq=seg_seq,
                turn_generation=turn_generation,
            )
            await self._events.put(
                TalkerFinished(
                    turn_id=signal.turn_id,
                    mode=mode,
                    finished_at=finished_at,
                    snapshot_seq=signal.snapshot_seq,
                )
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self._events.put(
                TalkerFailed(
                    turn_id=signal.turn_id,
                    error=repr(exc),
                )
            )

    async def _run_thinker(self, turn_id: TurnId, prompt: str) -> None:
        try:
            async for ev in self._thinker.start(turn_id, prompt):
                await self._events.put(ev)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.exception("thinker raised for turn %s: %s", turn_id, exc)

    def _record_talker_segment(
        self,
        signal: ControlSignal,
        mode: TalkerMode,
        text: str,
        started_at: Timestamp,
        finished_at: Timestamp,
        seg_seq: int,
        turn_generation: int,
    ) -> None:
        """Append the TalkerSegment with the pre-reserved ``seg_seq`` and
        immediately apply any pending TTS playback timestamps stashed from
        early bridge notifications.

        ``seg_seq`` MUST equal the seq used in the chunks emitted to the
        per-turn queue for this dispatch; both are reserved by
        ``_send_talker`` before the supervisor starts.

        ``turn_generation`` is the value captured at dispatch time. If
        cancellation has bumped the per-turn generation since then, drop
        the write — the turn's history must remain frozen at the moment of
        cancellation. Duplicate seq within an active generation is also
        treated as a no-op-with-warning rather than a hard error so a
        single bookkeeping mistake does not crash the controller event loop.
        """
        if signal.turn_id in self._cancelled_turns:
            logger.warning(
                "dropping stale talker segment for cancelled turn: turn=%s seq=%s",
                signal.turn_id, seg_seq,
            )
            return
        current_generation = self._turn_generation.get(signal.turn_id, 0)
        if current_generation != turn_generation:
            logger.warning(
                "dropping stale talker segment for cancelled turn: turn=%s seq=%s",
                signal.turn_id, seg_seq,
            )
            return

        if any(
            s.turn_id == signal.turn_id and s.seq == seg_seq
            for s in self._ws.snapshot().talker_history
        ):
            logger.warning(
                "duplicate TalkerSegment seq; dropping segment: turn=%s seq=%s",
                signal.turn_id, seg_seq,
            )
            return

        def append_segment(w):
            w.talker_history.append(
                TalkerSegment(
                    turn_id=signal.turn_id,
                    seq=seg_seq,
                    mode=mode,
                    text=text,
                    started_at=started_at,
                    finished_at=finished_at,
                    tts_first_audio_at=None,
                    triggered_by=signal,
                )
            )

        self._ws.mutate(append_segment)
        pending = self._pending_tts_updates.pop((signal.turn_id, seg_seq), None)
        if pending is None:
            return
        for field_name, value in pending.items():
            self._apply_tts_field(signal.turn_id, seg_seq, field_name, value)

    def _cancel_turn(self, turn_id: TurnId) -> None:
        # Bump the per-turn generation BEFORE any task cancellation so a
        # supervisor that is between async-iter end and segment-record
        # (where asyncio cancellation cannot preempt) observes the bump and
        # drops its write. Add to _cancelled_turns FIRST so late bridge
        # callbacks for already-recorded segments are rejected by
        # _apply_tts_field's pre-lookup check (a cancelled turn that
        # already had a TalkerSegment append committed before cancellation
        # must not have its TTS fields filled in by a stale callback).
        self._cancelled_turns.add(turn_id)
        self._turn_generation[turn_id] = self._turn_generation.get(turn_id, 0) + 1
        if self._active_turn == turn_id:
            self._active_turn = None
        self._thinker.cancel(turn_id)
        if self._thinker_task is not None and not self._thinker_task.done():
            self._thinker_task.cancel()
        if self._classify_task is not None and not self._classify_task.done():
            self._classify_task.cancel()
        for task in self._talker_supervisors.pop(turn_id, []):
            task.cancel()
        self._first_hard_dispatched.discard(turn_id)
        self._last_finished_at.pop(turn_id, None)
        self._terminal_dispatch_finished.discard(turn_id)
        # Drop seq counter + any pending TTS updates for the cancelled turn.
        # Reusing a cancelled turn id within the same dialog is intentionally
        # unsupported — see the _cancelled_turns docstring.
        self._next_talker_seq.pop(turn_id, None)
        for key in list(self._pending_tts_updates.keys()):
            if key[0] == turn_id:
                self._pending_tts_updates.pop(key, None)
        self._close_turn_output(turn_id)

    def _current_thinker_history_length(self, turn_id: TurnId) -> int:
        return len(self._ws.current_turn_thinker(turn_id))

    def _has_thinker_segment(self, turn_id: TurnId, seq: int) -> bool:
        return any(
            isinstance(event, ThinkerSegment) and event.turn_id == turn_id and event.seq == seq
            for event in self._ws.current_turn_thinker(turn_id)
        )

    def _latest_talker_mode(self, turn_id: TurnId) -> Optional[TalkerMode]:
        for segment in reversed(self._ws.current_turn_talker(turn_id)):
            return segment.mode
        return None

    def _close_turn_output(self, turn_id: TurnId) -> None:
        queue = self._turn_outputs.pop(turn_id, None)
        if queue is not None:
            queue.put_nowait(_TURN_END_SENTINEL)

    async def shutdown(self) -> None:
        """Cancel all in-flight subtasks (thinker / classify / talker
        supervisors) and await their completion. Called by LLMBundle.end_dialog
        before the runner itself is cancelled to keep tasks from outliving the
        dialog and leaking references to the workspace."""
        if self._active_turn is not None:
            self._thinker.cancel(self._active_turn)
        pending: List[asyncio.Task] = []
        for task in (self._thinker_task, self._classify_task):
            if task is not None and not task.done():
                task.cancel()
                pending.append(task)
        for tasks in list(self._talker_supervisors.values()):
            for t in tasks:
                if not t.done():
                    t.cancel()
                    pending.append(t)
        for turn_id in list(self._turn_outputs.keys()):
            self._close_turn_output(turn_id)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
