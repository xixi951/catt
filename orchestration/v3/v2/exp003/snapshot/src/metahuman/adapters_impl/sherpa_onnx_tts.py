"""Local sherpa-onnx VITS TTS adapter (replaces network-bound edge-tts).

The §8.6 router-threshold sweep showed that ``EdgeTTS``'s service call
dominated TTFA (mean ~7 s, p95 ~19 s) and occasionally returned no
audio (Microsoft Edge TTS service flakiness). Both characteristics
made router-decision latency noise impossible to isolate. This adapter
runs a local sherpa-onnx VITS model end-to-end on CPU/GPU, producing
audio in 100-400 ms for typical LLM-response-length text. Mean TTFA
in the §8.6 setup drops by an order of magnitude and the
``NoAudioReceived`` failure mode disappears.

Pip install: ``pip install sherpa-onnx``. The package bundles its
ONNX runtime; no separate install required.

Model: pass ``OfflineTtsVitsModelConfig`` paths via constructor. The
default (``vits-piper-en_US-amy-low``) is the Piper TTS Amy voice (low
quality / small model); the model files are NOT bundled in the repo
because they are ~50 MB. Each environment is responsible for
downloading them once::

    mkdir -p ~/sherpa-onnx-models && cd ~/sherpa-onnx-models && \\
        curl -sL -o vits-piper-en_US-amy-low.tar.bz2 \\
            https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2 && \\
        tar -xjf vits-piper-en_US-amy-low.tar.bz2

Environment variable ``SHERPA_ONNX_TTS_MODEL_DIR`` overrides the default
model directory (``~/sherpa-onnx-models/vits-piper-en_US-amy-low``).
The constructor reads it at instantiation time so swapping models is a
one-line env change without code edits.

First-byte semantics: sherpa-onnx's ``OfflineTts.generate(text)`` is
synchronous and returns the full audio buffer in one call. There is no
streaming "first byte" the way edge-tts has from its WebSocket service.
We therefore set ``first_byte_at`` equal to ``finished_at`` minus the
synth elapsed — semantically "the moment we have audio bytes ready"
collapses to "synth complete" for an offline batch model. Per-stage
analysis can still distinguish "router decided → talker dispatch
start → synth complete" by comparing ``started_at``, ``finished_at``,
and the existing ``talker_segment_update`` events.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from metahuman.tts import TTSInterface, TTSResult


_DEFAULT_MODEL_DIR = Path.home() / "sherpa-onnx-models" / "vits-piper-en_US-amy-low"
_DEFAULT_MODEL_FILENAME = "en_US-amy-low.onnx"
_DEFAULT_TOKENS_FILENAME = "tokens.txt"
_DEFAULT_DATA_DIRNAME = "espeak-ng-data"


class SherpaOnnxTTS(TTSInterface):
    """sherpa-onnx VITS TTS with lazy model load and thread-safe synth.

    The underlying ``sherpa_onnx.OfflineTts`` instance is constructed
    once per adapter instance (the model load is the slow step at
    ~3-5 seconds; synth itself is hundreds of ms). A per-instance lock
    serialises ``generate`` calls so concurrent dispatches do not race
    on the C++ object's internal state.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        model_filename: str = _DEFAULT_MODEL_FILENAME,
        tokens_filename: str = _DEFAULT_TOKENS_FILENAME,
        data_dirname: str = _DEFAULT_DATA_DIRNAME,
        speaker_id: int = 0,
        speed: float = 1.0,
        num_threads: int = 2,
    ) -> None:
        env_dir = os.environ.get("SHERPA_ONNX_TTS_MODEL_DIR")
        if model_dir is not None:
            self._model_dir = Path(model_dir)
        elif env_dir:
            self._model_dir = Path(env_dir)
        else:
            self._model_dir = _DEFAULT_MODEL_DIR
        self._model_filename = model_filename
        self._tokens_filename = tokens_filename
        self._data_dirname = data_dirname
        self._speaker_id = int(speaker_id)
        self._speed = float(speed)
        self._num_threads = int(num_threads)

        self._tts: Optional[Any] = None
        self._sample_rate: int = 0
        self._lock = threading.Lock()
        self._play_until: float = 0.0

    def synthesize_text(self, text: str, turn_anchor_monotonic: float) -> TTSResult:
        tts = self._load()
        started_at_monotonic = time.monotonic()
        with self._lock:
            audio = tts.generate(text, sid=self._speaker_id, speed=self._speed)
        finished_at_monotonic = time.monotonic()

        # sherpa-onnx returns float32 samples in [-1, 1]. Convert to int16
        # PCM to match the rest of the pipeline's audio expectations
        # (EdgeTTS adapter uses int16; SpeakerBridge does not care, but
        # downstream consumers might).
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size:
            samples_int16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
        else:
            samples_int16 = samples.astype(np.int16)

        sample_rate = int(audio.sample_rate)
        duration_s = float(samples_int16.shape[0]) / float(sample_rate) if sample_rate else 0.0

        self._play_until = max(self._play_until, finished_at_monotonic) + duration_s

        # Offline model: no real "first byte" event from a stream — the
        # entire audio buffer becomes available at synth-complete. Set
        # first_byte_at == finished_at so the field is populated and
        # downstream analysis can still recognise "synth-service-time"
        # (= finished_at - started_at) as a single phase.
        return TTSResult(
            audio=samples_int16,
            sample_rate=sample_rate,
            duration_s=duration_s,
            started_at=started_at_monotonic - turn_anchor_monotonic,
            finished_at=finished_at_monotonic - turn_anchor_monotonic,
            first_byte_at=finished_at_monotonic - turn_anchor_monotonic,
        )

    def backlog_seconds(self) -> float:
        now = time.monotonic()
        return max(0.0, self._play_until - now)

    def reset(self) -> None:
        """Drop the accumulated virtual-playback cursor.

        ``_play_until`` is advanced by every ``synthesize_text`` call's
        ``duration_s`` and is never time-decayed; over many turns it grows
        past wall clock by however much synthesized-audio-duration
        exceeded turn wall time. Once it leads wall clock by tens of
        minutes, ``backlog_seconds()`` reports nonzero for the same span,
        which prevents the controller's ``_close_turn_if_done`` (requires
        exact ``backlog == 0``) from ever firing until wall time finally
        catches up. The bridge calls this between turns to keep the
        cursor representing only the current turn's queued audio.
        """
        self._play_until = 0.0

    def _load(self) -> Any:
        if self._tts is not None:
            return self._tts
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ImportError(
                "SherpaOnnxTTS requires `pip install sherpa-onnx`."
            ) from exc

        model_path = self._model_dir / self._model_filename
        tokens_path = self._model_dir / self._tokens_filename
        data_path = self._model_dir / self._data_dirname
        for path, label in (
            (model_path, "model file"),
            (tokens_path, "tokens file"),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"sherpa-onnx TTS {label} not found: {path}. "
                    "See module docstring for download instructions."
                )

        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_path) if data_path.is_dir() else "",
                ),
                num_threads=self._num_threads,
            ),
            max_num_sentences=1,
        )
        self._tts = sherpa_onnx.OfflineTts(cfg)
        self._sample_rate = int(self._tts.sample_rate)
        return self._tts
