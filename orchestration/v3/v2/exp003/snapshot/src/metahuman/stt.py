"""STT interface (research-plan.md §5.5; mirrors OLV ASRInterface)."""
from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass

import numpy as np

from .types import AudioMeta, Timestamp


@dataclass(frozen=True)
class STTResult:
    text: str
    is_final: bool  # OLV's ASRInterface lacks this; added for streaming
    audio_meta: AudioMeta  # turn-relative (.started_at == 0, .finished_at == duration_s)

    # Raw clock anchors captured by STT at VAD-detected speech start. The
    # orchestrator passes these to WorkspaceStore.register_turn() before any
    # other call to WorkspaceStore.now(turn_id) for this turn.
    turn_anchor_monotonic: float  # raw time.monotonic() at speech start
    turn_anchor_walltime: float   # raw time.time() (Unix seconds) at speech start

    # Turn-relative seconds for STT inference span; semantically distinct from
    # audio physical times.
    transcribe_started_at: Timestamp
    transcribe_finished_at: Timestamp


class STTInterface(abc.ABC):
    """Sync transcribe API + async wrapper. Concrete impls live in adapters_impl/."""

    SAMPLE_RATE: int = 16000
    NUM_CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2

    @abc.abstractmethod
    def transcribe_np(
        self,
        audio: np.ndarray,
        turn_anchor_monotonic: float,
        turn_anchor_walltime: float,
    ) -> STTResult:
        """Transcribe a float32 mono audio array.

        The two anchors are captured by the orchestrator or VAD layer when
        speech starts. STT implementations use them to populate
        STTResult.turn_anchor_monotonic / .turn_anchor_walltime and compute
        turn-relative transcribe_started_at / transcribe_finished_at.
        """
        ...

    async def async_transcribe_np(
        self,
        audio: np.ndarray,
        turn_anchor_monotonic: float,
        turn_anchor_walltime: float,
    ) -> STTResult:
        return await asyncio.to_thread(
            self.transcribe_np,
            audio.astype(np.float32),
            turn_anchor_monotonic,
            turn_anchor_walltime,
        )
