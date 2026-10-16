"""TTS interface (research-plan.md §5.5; mirrors OLV)."""
from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Timestamp


@dataclass(frozen=True)
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    started_at: Timestamp
    finished_at: Timestamp
    # Turn-relative moment the synthesis service returned its first audio
    # byte. Optional because not all backends expose this. EdgeTTS sets it
    # from ``communicate.stream()``'s first audio chunk; the full decode
    # then runs and ``finished_at`` is recorded. Used to split the TTS
    # phase of TTFA into service-call vs decode time. Default None
    # preserves existing call-site contracts.
    first_byte_at: Optional[Timestamp] = None


class TTSInterface(abc.ABC):
    SAMPLE_RATE: int = 16000
    NUM_CHANNELS: int = 1

    @abc.abstractmethod
    def synthesize_text(self, text: str, turn_anchor_monotonic: float) -> TTSResult:
        """Synthesize ``text`` into audio. ``turn_anchor_monotonic`` is the raw
        ``time.monotonic()`` reading at the active turn's zero-point (captured
        by STT at speech start and shared via WorkspaceStore.register_turn);
        implementations rebase ``TTSResult.started_at`` and ``finished_at``
        against it so all downstream timestamp math stays turn-relative,
        consistent with §5.1's time convention."""
        ...

    async def async_synthesize_text(
        self,
        text: str,
        turn_anchor_monotonic: float,
    ) -> TTSResult:
        return await asyncio.to_thread(
            self.synthesize_text,
            text,
            turn_anchor_monotonic,
        )

    @abc.abstractmethod
    def backlog_seconds(self) -> float:
        """Audio queue duration; orchestrator polls this and feeds it to LLM (§6.6)."""
        ...

    def reset(self) -> None:
        """Drop any internal playback-cursor state accumulated across synth
        calls so ``backlog_seconds()`` returns 0 immediately after.

        Called by ``SpeakerBridge.reset_playback_cursor()`` between turns
        in experimental drivers. The contract is: invoke only when no
        synth task is in flight (the bridge enforces this by calling
        ``wait_until_drained`` first); a reset issued mid-pipeline can be
        overwritten by a later in-flight synth completion. Default is a
        no-op so adapters that do not maintain such state keep working.

        Concrete adapters that maintain a virtual ``_play_until`` cursor
        (SherpaOnnxTTS, EdgeTTS) must override this to zero the cursor;
        without the override the cursor accumulates ``duration_s`` per
        synth across the entire run, which in long sweeps drives
        ``backlog_seconds()`` far ahead of wall clock and prevents the
        controller's ``_close_turn_if_done`` (requires exact
        ``backlog == 0``) from firing until wall time finally catches up.
        """
        return
