"""SW Ranking router with Qwen3 embeddings, sklearn full-similarity solver.

Drop-in replacement for ``RouteLLMBertRouter``. Decides easy / hard for
the controller by estimating ``P(thinking-on wins | query)`` over a
training set of labelled battles, weighted by cosine similarity from
Qwen3-Embedding-0.6B.

Pipeline at inference time (``classify``):

1. Encode the user's utterance with Qwen3-Embedding-0.6B (instruction-
   wrapped, last-token pooling, L2-normalised) -> query embedding.
2. Cosine similarity between the query embedding and the pre-computed
   battle embedding matrix (full N rows, no top-k truncation).
3. Use the similarities as ``sample_weight`` for a weighted Bradley-
   Terry / logistic regression fit on the battle labels (one fit per
   request; with only N_models=2 the design matrix has 2 columns and
   the fit is sub-millisecond).
4. ``strong_win_rate = sigmoid(rating[strong] - rating[weak])`` where
   strong = thinking-on, weak = thinking-off.

Build pipeline (offline, ``scripts/build_router_artifacts.py``):

* Reads a battle CSV with columns ``prompt_id, prompt, winner`` (winner
  is ``"strong"``, ``"weak"``, or ``"tie"``); writes:
    - ``battles.parquet`` (or .csv): one row per battle, with model_a
      and model_b indices.
    - ``embeddings.npy``: ``(N, dim)`` float32 L2-normalised embeddings
      for each battle prompt.
    - ``manifest.json``: metadata with embedding_model name,
      instruction_version, dim, normalize, embedding count, etc.
      Online and offline must use **exactly the same** instruction +
      pooling for cosine similarities to be meaningful.

The ``hard_threshold`` constructor argument matches the existing
``RouteLLMBertRouter``'s API: ``route='hard'`` if
``strong_win_rate >= hard_threshold``.

Embedding model: ``Qwen/Qwen3-Embedding-0.6B`` (HF cache pre-loaded on
srv18 / srv73). Uses sentence-transformers' ``SentenceTransformer``
class which already implements last-token pooling for this checkpoint
since v3+. Default device follows ``CUDA_VISIBLE_DEVICES`` (will fall
back to CPU if no GPU available; CPU latency is ~150-300 ms / query
on srv73 vs ~5-15 ms on a free A100).

Memory profile (N=820 battles, dim=1024, fp32):
    embeddings: 820 * 1024 * 4 = 3.4 MB
    Qwen3-Embedding-0.6B weights (fp16): ~1.2 GB
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..router import RouterInterface
from ..types import RouteDecision, UserUtterance


# Same instruction text used for both offline embedding precompute and
# online query encoding. Changing this string invalidates pre-computed
# embeddings; bump ``ROUTER_QUERY_INSTRUCTION_VERSION`` together.
ROUTER_QUERY_INSTRUCTION = (
    "Represent the user request for LLM routing. Similar requests "
    "should require similar levels of reasoning, domain expertise, "
    "coding ability, mathematical ability, and need for a stronger "
    "reasoning model."
)
ROUTER_QUERY_INSTRUCTION_VERSION = "route_query_v1"


@dataclass
class _BattleData:
    """In-memory battle table.

    ``label`` semantics (matches routellm.SWRankingRouter convention):

    * ``+1.0`` -> strong (thinking-on) won
    * ``0.0``  -> weak (thinking-off) won
    * ``0.5``  -> tie

    ``weights_seed`` is a per-battle base weight (defaults to 1.0); the
    online cosine similarity is multiplied into this to form the
    sklearn ``sample_weight`` argument.
    """

    labels: np.ndarray  # shape (N,), float in {0.0, 0.5, 1.0}
    base_weights: np.ndarray  # shape (N,), float >= 0
    embeddings: np.ndarray  # shape (N, dim), float32 L2-normalised


class SWRankingQwen3Router(RouterInterface):
    """SW Ranking router backed by Qwen3-Embedding-0.6B.

    Parameters
    ----------
    artifacts_dir:
        Directory produced by ``scripts/build_router_artifacts.py``.
        Must contain ``battles.parquet`` (or ``battles.csv``),
        ``embeddings.npy``, and ``manifest.json``.
    hard_threshold:
        ``[0, 1]``. ``route='hard'`` iff strong-win-rate >=
        threshold. Default 0.5 mirrors RouteLLM's documented binary
        cutoff and ``RouteLLMBertRouter`` for parity in the §8.6
        ablation table.
    embedding_model_name:
        Override the manifest's embedding model. Useful for
        experiments. Default ``None`` -> use whatever the manifest
        was built with.
    embedding_device:
        ``"cuda"``, ``"cpu"``, or ``None`` to autodetect. CUDA is
        ~10x faster but adds ~1.2 GB GPU memory.
    similarity_floor:
        Clip any cosine similarity below this to zero before using
        as sample_weight. Default 0.0 (no clipping). Setting >= 0.1
        prunes near-orthogonal battles (which contribute almost no
        signal anyway).
    """

    def __init__(
        self,
        artifacts_dir: Path,
        hard_threshold: float = 0.5,
        embedding_model_name: Optional[str] = None,
        embedding_device: Optional[str] = None,
        similarity_floor: float = 0.0,
    ) -> None:
        if not 0.0 <= hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be in [0, 1]")
        self.artifacts_dir = Path(artifacts_dir)
        self.hard_threshold = hard_threshold
        self.similarity_floor = float(similarity_floor)
        self._override_model = embedding_model_name
        self._device = embedding_device
        self._battles: Optional[_BattleData] = None
        self._encoder: Any = None  # sentence_transformers.SentenceTransformer
        self._manifest: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Idempotent load of embeddings, battles, and the embedding
        encoder. Validates that the runtime instruction / model match
        the manifest so cosine similarities are meaningful.
        """
        if self._battles is not None and self._encoder is not None:
            return

        manifest_path = self.artifacts_dir / "manifest.json"
        embeddings_path = self.artifacts_dir / "embeddings.npy"
        battles_csv_path = self.artifacts_dir / "battles.csv"
        battles_parquet_path = self.artifacts_dir / "battles.parquet"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Router artifacts manifest missing: {manifest_path}. "
                "Run scripts/build_router_artifacts.py first."
            )
        self._manifest = json.loads(manifest_path.read_text())

        # Cross-check: instruction must match (otherwise embeddings
        # are not comparable).
        manifest_instr = self._manifest.get("instruction_version")
        if manifest_instr != ROUTER_QUERY_INSTRUCTION_VERSION:
            raise ValueError(
                f"router artifacts use instruction_version "
                f"{manifest_instr!r} but the runtime is "
                f"{ROUTER_QUERY_INSTRUCTION_VERSION!r}; rebuild "
                "artifacts with the new instruction."
            )

        embeddings = np.load(embeddings_path).astype(np.float32)
        # Sanity: rows should be L2-normalised already.
        if not bool(self._manifest.get("normalize", False)):
            raise ValueError(
                "router artifacts were built without L2 normalisation; "
                "rebuild with `--normalize true`."
            )

        # Load battles (prefer parquet; fall back to CSV).
        if battles_parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(battles_parquet_path)
                df = table.to_pandas()
            except ImportError:
                import pandas as pd

                df = pd.read_csv(battles_csv_path)
        elif battles_csv_path.exists():
            import pandas as pd

            df = pd.read_csv(battles_csv_path)
        else:
            raise FileNotFoundError(
                f"battles.parquet / battles.csv missing in "
                f"{self.artifacts_dir}"
            )

        if len(df) != len(embeddings):
            raise ValueError(
                f"battle count {len(df)} does not match embedding count "
                f"{len(embeddings)} in {self.artifacts_dir}"
            )

        # Encode winner column to labels (1.0 strong-wins / 0.0 weak-wins
        # / 0.5 tie). Accept the canonical literals; raise on anything
        # else so dirty data does not silently corrupt the routing.
        winner_map = {
            "strong": 1.0,  # thinking-on
            "thinking_on": 1.0,
            "weak": 0.0,  # thinking-off
            "thinking_off": 0.0,
            "tie": 0.5,
            "both": 0.5,
        }
        labels = np.array(
            [winner_map[str(w).lower()] for w in df["winner"]],
            dtype=np.float64,
        )
        if "weight" in df.columns:
            base_weights = df["weight"].to_numpy(dtype=np.float64)
        else:
            base_weights = np.ones(len(df), dtype=np.float64)

        self._battles = _BattleData(
            labels=labels,
            base_weights=base_weights,
            embeddings=embeddings,
        )

        # Load encoder.
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SWRankingQwen3Router requires `pip install "
                "sentence-transformers`."
            ) from exc

        model_name = (
            self._override_model
            or self._manifest.get("embedding_model")
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self._encoder = SentenceTransformer(
            model_name,
            device=self._device,
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # RouterInterface
    # ------------------------------------------------------------------

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        """Run encoder + similarity + weighted BT in a worker thread.

        Diagnostic timing split: dispatch_ms is the asyncio overhead
        (time from to_thread() being awaited until _infer actually
        starts executing on a worker thread). infer_ms is the time
        spent inside _infer (encoder forward + similarity + Bradley
        Terry). Their sum equals total elapsed_ms.
        """
        started_at = time.monotonic()
        timing_box: list = []  # appended by _infer_with_timing
        score = await asyncio.to_thread(
            self._infer_with_timing, utterance.text, timing_box
        )
        elapsed_s = time.monotonic() - started_at
        infer_start = timing_box[0] if len(timing_box) > 0 else None
        infer_end = timing_box[1] if len(timing_box) > 1 else None
        dispatch_ms = (
            (infer_start - started_at) * 1000 if infer_start is not None else None
        )
        infer_ms = (
            (infer_end - infer_start) * 1000
            if (infer_end is not None and infer_start is not None)
            else None
        )
        score = max(0.0, min(1.0, float(score)))

        route = "hard" if score >= self.hard_threshold else "easy"
        timing_suffix = ""
        if dispatch_ms is not None and infer_ms is not None:
            timing_suffix = (
                f", dispatch_ms={dispatch_ms:.1f}, infer_ms={infer_ms:.1f}"
            )
        if len(timing_box) >= 4:
            gc_pause_ms = timing_box[2]
            collected = timing_box[3]
            timing_suffix += f", gc_pause_ms={gc_pause_ms:.1f}, gc_collected={collected}"
        rationale = (
            f"sw_qwen3: strong_win_prob={score:.3f} "
            f"(N={len(self._battles.labels)}, "
            f"latency_ms={elapsed_s * 1000:.1f}{timing_suffix})"
        )
        return RouteDecision(
            turn_id=utterance.turn_id,
            route=route,
            confidence=score,
            rationale=rationale,
            decided_at=utterance.received_at + elapsed_s,
        )

    def _infer_with_timing(self, text: str, timing_box: list) -> float:
        """Wrapper around _infer that captures start and end timestamps
        in the caller-provided list. The caller (classify) computes
        dispatch_ms from started_at to infer_start and infer_ms from
        infer_start to infer_end.

        Also captures pre-call gc.collect() pause time to test whether
        cyclic GC is the cause of router latency spikes under heavy
        thinker streaming load. METAHUMAN_ROUTER_GC_PROBE=1 enables
        a synchronous gc.collect() right before the encoder call;
        the time spent inside that call is appended to timing_box[2].
        Without the probe, no GC is forced and timing_box has only
        2 entries.
        """
        gc_probe = os.environ.get("METAHUMAN_ROUTER_GC_PROBE") == "1"
        if gc_probe:
            gc_t0 = time.monotonic()
            collected = gc.collect()
            gc_pause_ms = (time.monotonic() - gc_t0) * 1000
            timing_box.append(time.monotonic())  # infer_start
            try:
                result = self._infer(text)
            finally:
                timing_box.append(time.monotonic())  # infer_end
                timing_box.append(gc_pause_ms)
                timing_box.append(collected)
            return result
        timing_box.append(time.monotonic())
        try:
            return self._infer(text)
        finally:
            timing_box.append(time.monotonic())

    # ------------------------------------------------------------------
    # core inference
    # ------------------------------------------------------------------

    def _infer(self, text: str) -> float:
        """Synchronous embed + similarity + weighted-BT solve.

        Returns ``P(strong wins | query)`` in [0, 1].
        """
        self._load()
        battles = self._battles
        assert battles is not None  # post-load invariant

        # 1. encode query (instruction-wrapped, L2-normalised).
        wrapped = (
            f"Instruct: {ROUTER_QUERY_INSTRUCTION}\nQuery: {text}"
        )
        query_emb = self._encoder.encode(
            [wrapped],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        # query_emb: shape (1, dim)

        # 2. cosine similarity. Both sides L2-normalised → dot product.
        sims = (query_emb @ battles.embeddings.T)[0]  # shape (N,)
        if self.similarity_floor > 0:
            sims = np.where(sims >= self.similarity_floor, sims, 0.0)
        # Negative similarities are unusual for L2-normalised
        # sentence embeddings but possible; clip to zero so they do
        # not flip the sign of the weighted average.
        sims = np.clip(sims, 0.0, None)

        # 3. weighted "strong wins" rate (Bradley-Terry on 2 models
        # collapses to a weighted Bernoulli mean; the logit form
        # adds nothing here so we use the closed-form mean).
        weights = sims * battles.base_weights
        # Tied battles count half toward each side (label 0.5).
        denom = float(weights.sum())
        if denom <= 0:
            # No similar-enough battles in the training set. Fall
            # back to the marginal rate of strong wins so we always
            # return a valid probability.
            denom = float(battles.base_weights.sum()) or 1.0
            numer = float((battles.labels * battles.base_weights).sum())
            return numer / denom
        numer = float((battles.labels * weights).sum())
        return numer / denom
