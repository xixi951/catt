"""Data types for the metahuman LLM bundle (research-plan.md §5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Union

DialogId = str
TurnId = str
Timestamp = float  # seconds since this turn's monotonic anchor (= when user started
                   # speaking). The anchor is captured by STT at VAD-detected speech
                   # start and registered with WorkspaceStore.register_turn(); use
                   # WorkspaceStore.now(turn_id) to obtain a fresh Timestamp.


@dataclass(frozen=True)
class AudioMeta:
    duration_s: float
    sample_rate: int
    started_at: Timestamp
    finished_at: Timestamp


class SessionState(str, Enum):
    """LLM-internal state; controls Talker dispatch mode (§5.6)."""
    EASY = "easy"
    HARD_FIRST = "hard_first"
    HARD_MID = "hard_mid"
    HARD_FINAL = "hard_final"


class TalkerMode(str, Enum):
    EASY = "easy"
    HARD_FIRST = "hard_first"
    HARD_MID = "hard_mid"
    HARD_FINAL = "hard_final"


@dataclass(frozen=True)
class UserUtterance:
    turn_id: TurnId
    text: str
    audio_meta: AudioMeta
    received_at: Timestamp


@dataclass(frozen=True)
class RouteDecision:
    """Router output (§5.7). `confidence` is RouteLLM's strong-model win rate
    (the `calculate_strong_win_rate` output, semantically P(strong model wins) ∈ [0, 1])."""
    turn_id: TurnId
    route: Literal["easy", "hard"]
    confidence: float  # in [0, 1]; >= hard_threshold => route='hard'
    rationale: str
    decided_at: Timestamp


@dataclass(frozen=True)
class ThinkerSegment:
    turn_id: TurnId
    seq: int
    content: str
    created_at: Timestamp


EoTReason = Literal["natural"]
# Streaming Thinker only ever emits "natural" today: one streaming chat
# call drains to completion (or is cancelled, in which case no EoT is
# emitted at all). The Literal is kept as a single-value enum so future
# branches (cancellation-with-summary, upstream-error tail, etc.) can
# extend it without a schema break.


@dataclass(frozen=True)
class EndOfThinking:
    turn_id: TurnId
    final_summary: str
    total_segments: int
    created_at: Timestamp
    reason: EoTReason = "natural"


ThinkerEvent = Union[ThinkerSegment, EndOfThinking]


ControlCause = Literal[
    "route_easy",
    "route_hard_first",
    "tts_low_water",
    "thinker_segment_ready",
    "end_of_thinking",
]


@dataclass(frozen=True)
class ControlSignal:
    """Controller-to-Talker dispatch (§5.7); mode is read from ws.state."""
    turn_id: TurnId
    triggered_at: Timestamp
    cause: ControlCause
    snapshot_seq: int  # thinker_history len at trigger; idempotency key


@dataclass(frozen=True)
class TalkerSegment:
    turn_id: TurnId
    seq: int
    mode: TalkerMode
    text: str
    started_at: Timestamp
    finished_at: Optional[Timestamp]
    tts_first_audio_at: Optional[Timestamp]
    triggered_by: ControlSignal
    # Wall-clock-equivalent moment (turn-relative) at which the audio for this
    # dispatch finishes playing under the continuous-playback model in §6.6.
    # Filled by the orchestrator-side bridge via LLMInterface.notify_tts_finished_at;
    # None until set. Pair with tts_first_audio_at to compute §8.5's audio-gap
    # metric.
    tts_finished_at: Optional[Timestamp] = None
    # Turn-relative moment the TTS service returned its first audio byte for
    # this segment, BEFORE the synth adapter performed full-segment decode
    # (mp3 → PCM via pydub for EdgeTTS). Together with
    # ``tts_first_audio_at`` this lets us split the TTS phase of TTFA into
    # "service call + network" (started_at → first_byte_at) and "decode"
    # (first_byte_at → first_audio_at). Optional because not all TTS
    # adapters expose first-chunk timing; None preserves back-compat.
    tts_first_byte_at: Optional[Timestamp] = None


@dataclass(frozen=True)
class TalkerChunk:
    """One text fragment from a Talker dispatch's streaming output (§6.6, §8.5).

    A single talker dispatch (one TalkerSegment) yields several TalkerChunk
    items in order; the per-turn output queue carries them. ``is_last_in_segment``
    flags the final chunk so consumers can detect dispatch boundaries without
    a separate side channel — this is what lets SpeakerBridge stay an
    OLV-style decoupled middleware (LLM yields chunks, TTS synthesizes them,
    Bridge bridges).

    ``talker_seg_seq`` matches the eventual TalkerSegment.seq written at
    dispatch end, so JSONL log readers can correlate chunk-stream events to
    the final segment record.

    ``produced_at`` is the turn-relative moment the controller put this
    chunk on the per-turn output queue (i.e., the LLM-produced text became
    visible to downstream consumers). Stamped here rather than measured at
    consumer side so §8 H2's text-TTFT metric stays unaffected by any
    downstream synthesis backpressure (SpeakerBridge.run awaits TTS synth
    before yielding, so consumer-side timing reflects synth latency, not
    LLM latency; ``produced_at`` reads the LLM-side moment regardless).
    """
    turn_id: TurnId
    talker_seg_seq: int
    text: str
    is_last_in_segment: bool
    produced_at: Timestamp


@dataclass
class Workspace:
    """Shared work space (§5.2). Controller is sole writer."""
    dialog_id: DialogId
    state: Optional[SessionState] = None
    user_history: List[UserUtterance] = field(default_factory=list)
    route_history: List[RouteDecision] = field(default_factory=list)
    thinker_history: List[ThinkerEvent] = field(default_factory=list)
    talker_history: List[TalkerSegment] = field(default_factory=list)
