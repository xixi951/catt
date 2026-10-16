"""metahuman LLM bundle (research-plan.md v5.0 skeleton).

Public API surface: STT/TTS/LLM interfaces and shared dataclasses for the
three-way decoupling described in §4. The Controller class and its internal
event types (TalkerStarted / TalkerFinished / TalkerFailed / TTSLowWater)
are intentionally not re-exported: they are LLMBundle-internal wiring; outside
callers should interact with the LLM subsystem only via LLMInterface.
"""
from .llm import LLMBundle, LLMInterface
from .router import RouterInterface
from .stt import STTInterface, STTResult
from .talker import TalkerInterface
from .thinker import ThinkerInterface
from .tts import TTSInterface, TTSResult
from .types import (
    AudioMeta,
    ControlSignal,
    DialogId,
    EndOfThinking,
    RouteDecision,
    SessionState,
    TalkerMode,
    TalkerSegment,
    ThinkerEvent,
    ThinkerSegment,
    Timestamp,
    TurnId,
    UserUtterance,
    Workspace,
)
from .workspace import WorkspaceStore

__all__ = [
    "AudioMeta",
    "ControlSignal",
    "DialogId",
    "EndOfThinking",
    "LLMBundle",
    "LLMInterface",
    "RouteDecision",
    "RouterInterface",
    "STTInterface",
    "STTResult",
    "SessionState",
    "TTSInterface",
    "TTSResult",
    "TalkerInterface",
    "TalkerMode",
    "TalkerSegment",
    "ThinkerEvent",
    "ThinkerInterface",
    "ThinkerSegment",
    "Timestamp",
    "TurnId",
    "UserUtterance",
    "Workspace",
    "WorkspaceStore",
]
