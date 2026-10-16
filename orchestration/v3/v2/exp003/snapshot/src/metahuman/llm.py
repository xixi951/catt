"""LLM bundle interface and default implementation (§4.2)."""
from __future__ import annotations

import abc
import asyncio
from typing import AsyncIterator, Callable, Optional

from pathlib import Path

from .controller import Controller, _TURN_END_SENTINEL as _CHUNK_SENTINEL
from .router import RouterInterface
from .talker import TalkerInterface
from .thinker import ThinkerInterface
from .types import AudioMeta, DialogId, TalkerChunk, Timestamp, TurnId, UserUtterance
from .workspace import WorkspaceStore

# Per-dialog factories: thinker / talker need a workspace reference at
# construction time, but each begin_dialog creates a fresh workspace.
# The factory contract is "given the dialog's workspace, build a
# wired-up adapter instance"; LLMBundle calls the factories inside
# begin_dialog and discards the instances on end_dialog.
ThinkerFactory = Callable[[WorkspaceStore], ThinkerInterface]
TalkerFactory = Callable[[WorkspaceStore], TalkerInterface]
# Optional Controller factory. When supplied, LLMBundle.begin_dialog calls it
# to construct the per-dialog Controller; the factory takes the four
# already-built dependencies and returns a fully-configured Controller
# (typical use: bake §8.5 hard_mid threshold tuples into a closure so each
# sweep cell can run with its own thresholds while reusing the rest of the
# pipeline). Default is None — LLMBundle then falls back to the project-wide
# default Controller(...) constructor.
ControllerFactory = Callable[
    [WorkspaceStore, RouterInterface, ThinkerInterface, TalkerInterface],
    Controller,
]


class LLMInterface(abc.ABC):
    """Boundary interface for Router + Controller + Thinker + Talker + WS (§4.2)."""

    @abc.abstractmethod
    def begin_dialog(self, dialog_id: DialogId) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def end_dialog(self) -> None:
        """Cancel every in-flight subtask (controller runner, thinker, router
        classify, talker supervisors) and wait for them to drain before
        returning. Async because cleanup needs to await the cancelled tasks;
        a synchronous variant could leave tasks alive after return and leak
        references to the dialog's workspace."""
        raise NotImplementedError

    @abc.abstractmethod
    def handle_user_text(
        self,
        turn_id: TurnId,
        text: str,
        audio_meta: AudioMeta,
        turn_anchor_monotonic: float,
        turn_anchor_walltime: float,
    ) -> AsyncIterator[TalkerChunk]:
        """Yield TalkerChunks for the orchestrator to forward to TTS.

        Each chunk carries the text fragment, its dispatch identifier
        (``talker_seg_seq``) and an ``is_last_in_segment`` flag so a
        SpeakerBridge can detect dispatch boundaries from the stream alone
        without LLM knowing TTS exists.

        The turn anchors are copied from STTResult. LLMBundle registers them
        before submitting the user utterance so WorkspaceStore.now(turn_id) is
        valid throughout the controller path.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def notify_tts_backlog(self, backlog_seconds: float) -> None:
        """Orchestrator pushes TTS backlog readings; LLM converts to TTSLowWater events."""
        raise NotImplementedError

    @abc.abstractmethod
    def notify_tts_first_audio(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_audio_at: Timestamp,
    ) -> None:
        """Orchestrator reports first-audio playback time for one specific dispatch.

        Identifies the dispatch by ``(turn_id, talker_seg_seq)`` rather than
        "latest unfilled segment" so notifications remain correct even when
        the bridge fires before the controller has appended the TalkerSegment
        record (chunks are emitted to the per-turn queue *before*
        ``_record_talker_segment`` runs; in normal async scheduling the
        bridge can race ahead). The controller updates the matching segment
        if it exists, or stashes the timestamp keyed by
        ``(turn_id, talker_seg_seq)`` and applies it at segment-record time.

        Idempotency contract: same value is silently accepted; conflicting
        value is logged but does NOT raise (skeleton-review R1: per user
        request, do not stop the experiment for a single bookkeeping bug).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def notify_tts_finished_at(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        finished_at: Timestamp,
    ) -> None:
        """Orchestrator reports playback-finish time for one specific dispatch.

        Same keyed contract and idempotency rules as
        ``notify_tts_first_audio``; the bridge calls this when it sees a
        TalkerChunk with ``is_last_in_segment=True`` and computes the
        finish wall-clock from its virtual playback queue.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def notify_tts_first_byte(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_byte_at: Timestamp,
    ) -> None:
        """Orchestrator reports the synth-service first-byte time for one
        dispatch (the moment the TTS backend returned its first audio
        chunk, BEFORE full-segment decode). Optional: bridges that wrap
        a TTS adapter without this signal can simply not call this.
        Same keyed contract and idempotency rules as
        ``notify_tts_first_audio``.
        """
        raise NotImplementedError


class LLMBundle(LLMInterface):
    """Default impl wiring WorkspaceStore, Router, Thinker, Talker, Controller.

    Constructor takes a pre-built router (router has no workspace dependency)
    and *factory* callables for thinker / talker. Each `begin_dialog` creates
    a fresh WorkspaceStore and immediately calls the factories with it so the
    thinker / talker / controller all share the same store. `end_dialog`
    discards the per-dialog instances; the next `begin_dialog` rebuilds.

    Earlier versions stored pre-built thinker / talker on the bundle; that
    coupled them to whichever workspace the caller had used at construction
    time, while `begin_dialog` would silently create a *different* workspace
    for the controller. The per-dialog factories close that gap.
    """

    def __init__(
        self,
        router: RouterInterface,
        thinker_factory: ThinkerFactory,
        talker_factory: TalkerFactory,
        controller_factory: Optional[ControllerFactory] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        """``controller_factory`` is the §8.5-driven extension: when supplied,
        ``begin_dialog`` calls it with ``(ws, router, thinker, talker)`` and
        uses the returned Controller instead of ``Controller(ws, ...)`` with
        defaults. Sweep drivers bake threshold tuples into the closure so
        each cell runs with its own hard_mid trigger thresholds. Default
        None: existing callers (smoke scripts, current tests) get the same
        behavior as before.

        ``log_dir`` is forwarded to ``WorkspaceStore`` for every dialog this
        bundle opens. Default None preserves the historical ``./logs``
        behavior. The §8.5 sweep driver supplies a per-run directory so each
        sweep does not pollute the project root.
        """
        self._router = router
        self._thinker_factory = thinker_factory
        self._talker_factory = talker_factory
        self._controller_factory = controller_factory
        self._log_dir = log_dir
        self._ws: Optional[WorkspaceStore] = None
        self._controller: Optional[Controller] = None
        self._runner: Optional[asyncio.Task] = None
        self._dialog_id: Optional[DialogId] = None
        # Per-dialog adapters; rebuilt in begin_dialog, discarded in end_dialog.
        self._thinker: Optional[ThinkerInterface] = None
        self._talker: Optional[TalkerInterface] = None

    def begin_dialog(self, dialog_id: DialogId) -> None:
        if self._ws is not None:
            raise RuntimeError("dialog already active; call end_dialog first")
        self._dialog_id = dialog_id
        self._ws = WorkspaceStore(dialog_id, log_dir=self._log_dir)
        self._thinker = self._thinker_factory(self._ws)
        self._talker = self._talker_factory(self._ws)
        if self._controller_factory is not None:
            self._controller = self._controller_factory(
                self._ws, self._router, self._thinker, self._talker,
            )
        else:
            self._controller = Controller(
                self._ws,
                self._router,
                self._thinker,
                self._talker,
            )
        self._runner = asyncio.create_task(self._controller.run())

    async def end_dialog(self) -> None:
        # Drain controller subtasks first so they release ws references before
        # the runner is cancelled; otherwise the cancelled runner could leave
        # background work pinning the workspace and emitting writes against a
        # dialog that callers have already considered closed.
        if self._controller is not None:
            await self._controller.shutdown()
        if self._runner is not None and not self._runner.done():
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        self._runner = None
        self._controller = None
        self._thinker = None
        self._talker = None
        self._ws = None
        self._dialog_id = None

    def handle_user_text(
        self,
        turn_id: TurnId,
        text: str,
        audio_meta: AudioMeta,
        turn_anchor_monotonic: float,
        turn_anchor_walltime: float,
    ) -> AsyncIterator[TalkerChunk]:
        return self._stream_turn(
            turn_id,
            text,
            audio_meta,
            turn_anchor_monotonic,
            turn_anchor_walltime,
        )

    async def _stream_turn(
        self,
        turn_id: TurnId,
        text: str,
        audio_meta: AudioMeta,
        turn_anchor_monotonic: float,
        turn_anchor_walltime: float,
    ) -> AsyncIterator[TalkerChunk]:
        if self._controller is None or self._ws is None:
            raise RuntimeError("LLMBundle.handle_user_text called outside of a dialog")

        self._ws.register_turn(
            turn_id,
            monotonic_anchor=turn_anchor_monotonic,
            walltime_anchor=turn_anchor_walltime,
        )
        queue: asyncio.Queue = asyncio.Queue()
        self._controller.attach_turn_output(turn_id, queue)

        utterance = UserUtterance(
            turn_id=turn_id,
            text=text,
            audio_meta=audio_meta,
            received_at=self._ws.now(turn_id),
        )
        await self._controller.on_user_utterance(utterance)

        while True:
            chunk = await queue.get()
            if chunk is _CHUNK_SENTINEL:
                return
            yield chunk

    def notify_tts_backlog(self, backlog_seconds: float) -> None:
        if self._controller is None:
            return
        self._controller.on_tts_backlog(backlog_seconds)

    def notify_tts_first_audio(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_audio_at: Timestamp,
    ) -> None:
        if self._controller is None:
            return
        self._controller.on_tts_first_audio(turn_id, talker_seg_seq, first_audio_at)

    def notify_tts_finished_at(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        finished_at: Timestamp,
    ) -> None:
        if self._controller is None:
            return
        self._controller.on_tts_finished_at(turn_id, talker_seg_seq, finished_at)

    def notify_tts_first_byte(
        self,
        turn_id: TurnId,
        talker_seg_seq: int,
        first_byte_at: Timestamp,
    ) -> None:
        if self._controller is None:
            return
        self._controller.on_tts_first_byte(turn_id, talker_seg_seq, first_byte_at)

    def workspace_log_path(self) -> Optional[Path]:
        """Public accessor for the active dialog's JSONL log path.

        Returns None when no dialog is active. The §8.5 driver reads this to
        feed ``per_turn_metrics`` after each turn; using a public accessor
        avoids the encapsulation hole of reaching into ``bundle._ws._log_path``.
        """
        if self._ws is None:
            return None
        return self._ws.log_path
