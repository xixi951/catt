"""WorkspaceStore: locked mutate + snapshot + JSONL persistence (§5.2, §6.3)."""
from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .types import (
    DialogId,
    EndOfThinking,
    RouteDecision,
    SessionState,
    TalkerSegment,
    ThinkerEvent,
    ThinkerSegment,
    TurnId,
    Workspace,
)


class WorkspaceStore:
    """Wraps a Workspace with a write lock, snapshot, and per-mutate JSONL append."""

    def __init__(self, dialog_id: DialogId, log_dir: Optional[Path] = None) -> None:
        self._ws = Workspace(dialog_id=dialog_id)
        self._lock = threading.RLock()
        self._log_dir = Path(log_dir) if log_dir is not None else Path("./logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"{dialog_id}.ws.jsonl"
        # Per-turn (monotonic, walltime) anchors. Monotonic is captured at
        # VAD-detected speech start by STT; walltime is the same instant in
        # Unix seconds. Anchors are in-memory only and do not persist across
        # process restarts; replay tools should derive walltime correlation
        # from the JSONL `ts` field of the user_utterance event.
        self._turn_anchors: Dict[TurnId, Tuple[float, float]] = {}

    def register_turn(
        self,
        turn_id: TurnId,
        monotonic_anchor: float,
        walltime_anchor: float,
    ) -> None:
        """Register the turn's clock anchors before any now()/walltime_for_turn()
        call for the same turn_id. Subsequent calls with the same turn_id
        raise ValueError to prevent silent re-anchoring."""
        with self._lock:
            if turn_id in self._turn_anchors:
                raise ValueError(f"turn already registered: {turn_id!r}")
            self._turn_anchors[turn_id] = (monotonic_anchor, walltime_anchor)

    def now(self, turn_id: TurnId) -> Timestamp:
        """Return seconds elapsed since the turn's monotonic anchor."""
        with self._lock:
            anchors = self._turn_anchors.get(turn_id)
            if anchors is None:
                raise KeyError(f"turn not registered: {turn_id!r}")
            monotonic_anchor, _ = anchors
            return time.monotonic() - monotonic_anchor

    @property
    def log_path(self) -> Path:
        """Public path to the JSONL log for this dialog."""
        return self._log_path

    def record_event(self, kind: str, data: Dict[str, Any]) -> None:
        """Append a non-state-mutating structured event to the JSONL log.

        Used for events that are not changes to ``Workspace`` (state /
        history fields) but are still part of the dialog's recoverable
        timeline. The §8.5 ablation uses this for ``controller_gate_drop``
        events emitted when a would-be hard_mid dispatch is blocked; the
        metrics reader only needs to scan one file.

        Caller is responsible for ensuring ``data`` is JSON-serializable;
        non-jsonable values pass through ``_jsonable`` first.
        """
        with self._lock:
            self._append_events([self._event(kind, data)])

    def walltime_for_turn(self, turn_id: TurnId) -> Optional[float]:
        """Unix epoch seconds at this turn's monotonic anchor, or None if
        the turn has not been registered."""
        with self._lock:
            anchors = self._turn_anchors.get(turn_id)
            if anchors is None:
                return None
            _, walltime_anchor = anchors
            return walltime_anchor

    def mutate(self, fn: Callable[[Workspace], None]) -> None:
        """Apply a write under lock; append one JSONL line after.

        `fn` MUST be atomic: either it succeeds and the workspace ends in a
        consistent state, or it raises before mutating anything. If `fn`
        partially mutates the workspace and then raises, the in-memory state
        will drift from the JSONL log (no events are written on exception)
        and subsequent diffs will misattribute the leftover changes. The
        store does not deep-copy on entry to avoid the per-write cost; the
        atomicity guarantee is the caller's responsibility."""
        with self._lock:
            before_state = self._ws.state
            before_user_len = len(self._ws.user_history)
            before_route_len = len(self._ws.route_history)
            before_thinker_len = len(self._ws.thinker_history)
            before_talker_len = len(self._ws.talker_history)
            before_talker = list(self._ws.talker_history)

            fn(self._ws)

            events: List[Dict[str, Any]] = []
            if self._ws.state != before_state:
                events.append(self._event("state", {"value": self._state_value(self._ws.state)}))

            for item in self._ws.user_history[before_user_len:]:
                events.append(self._event("user_utterance", self._dataclass_dict(item)))
            for item in self._ws.route_history[before_route_len:]:
                events.append(self._event("route_decision", self._dataclass_dict(item)))
            for item in self._ws.thinker_history[before_thinker_len:]:
                events.append(self._event(self._thinker_kind(item), self._dataclass_dict(item)))

            for old, new in zip(before_talker[:before_talker_len], self._ws.talker_history[:before_talker_len]):
                update = self._talker_update(old, new)
                if update is not None:
                    events.append(self._event("talker_segment_update", update))
            for item in self._ws.talker_history[before_talker_len:]:
                events.append(self._event("talker_segment", self._dataclass_dict(item)))

            self._append_events(events)

    def snapshot(self) -> Workspace:
        """Return a detached deep copy. Mutations to the returned object do not
        affect the store, but the type is structurally mutable; readers are by
        contract (gentleman's agreement, not enforced) expected not to mutate it."""
        with self._lock:
            return deepcopy(self._ws)

    def current_turn_thinker(self, turn_id: TurnId) -> List[ThinkerEvent]:
        """Subset of thinker_history with matching turn_id."""
        with self._lock:
            return deepcopy([event for event in self._ws.thinker_history if event.turn_id == turn_id])

    def current_turn_talker(self, turn_id: TurnId) -> List[TalkerSegment]:
        """Subset of talker_history with matching turn_id."""
        with self._lock:
            return deepcopy([segment for segment in self._ws.talker_history if segment.turn_id == turn_id])

    def current_user_text(self, turn_id: TurnId) -> Optional[str]:
        """Latest user_history.text for the given turn, or None."""
        with self._lock:
            for utterance in reversed(self._ws.user_history):
                if utterance.turn_id == turn_id:
                    return utterance.text
            return None

    def current_route(self, turn_id: TurnId) -> Optional[RouteDecision]:
        """Latest route_history entry for the given turn, or None."""
        with self._lock:
            for route in reversed(self._ws.route_history):
                if route.turn_id == turn_id:
                    return deepcopy(route)
            return None

    def thinker_backlog(self, turn_id: TurnId) -> int:
        """Count of thinker segments past the most recent talker dispatch's snapshot_seq."""
        with self._lock:
            produced = sum(
                1
                for event in self._ws.thinker_history
                if isinstance(event, ThinkerSegment) and event.turn_id == turn_id
            )
            consumed = max(
                (
                    segment.triggered_by.snapshot_seq
                    for segment in self._ws.talker_history
                    if segment.turn_id == turn_id
                ),
                default=0,
            )
            return max(0, produced - consumed)

    def _append_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        with self._log_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")

    def _event(self, kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"ts": time.time(), "kind": kind, "data": self._jsonable(data)}

    def _dataclass_dict(self, item: Any) -> Dict[str, Any]:
        if not is_dataclass(item) or isinstance(item, type):
            raise TypeError(f"expected dataclass instance, got {type(item)!r}")
        return self._jsonable({field.name: getattr(item, field.name) for field in fields(item)})

    def _thinker_kind(self, item: ThinkerEvent) -> str:
        if isinstance(item, ThinkerSegment):
            return "thinker_segment"
        if isinstance(item, EndOfThinking):
            return "end_of_thinking"
        raise TypeError(f"unsupported thinker event: {type(item)!r}")

    def _talker_update(self, old: TalkerSegment, new: TalkerSegment) -> Optional[Dict[str, Any]]:
        # Non-finalize fields must remain identical across an in-place replace.
        # If callers attempt to change them, fail loud rather than silently
        # dropping the change from the log.
        for field_name in ("turn_id", "seq", "mode", "text", "started_at", "triggered_by"):
            if getattr(old, field_name) != getattr(new, field_name):
                raise ValueError(
                    f"TalkerSegment.{field_name} cannot change after creation "
                    f"(turn_id={old.turn_id!r}, seq={old.seq})"
                )
        data: Dict[str, Any] = {"turn_id": new.turn_id, "seq": new.seq}
        if old.finished_at != new.finished_at:
            data["finished_at"] = new.finished_at
        if old.tts_first_audio_at != new.tts_first_audio_at:
            data["tts_first_audio_at"] = new.tts_first_audio_at
        if old.tts_finished_at != new.tts_finished_at:
            data["tts_finished_at"] = new.tts_finished_at
        if old.tts_first_byte_at != new.tts_first_byte_at:
            data["tts_first_byte_at"] = new.tts_first_byte_at
        if len(data) == 2:
            return None
        return data

    def _state_value(self, state: Optional[SessionState]) -> Optional[str]:
        if state is None:
            return None
        return state.value

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: self._jsonable(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._jsonable(item) for item in value]
        return value
