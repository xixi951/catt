"""Standalone HTTP service that wraps SWRankingQwen3Router.

Why a separate process: empirical evidence (Plan D experiment, 2026-05)
showed that running the SentenceTransformer encoder in the main
metahuman process causes 6-16 s router-latency spikes from turn ~15
onward when the talker/thinker run on DashScope qwen3-32b with
thinking_budget >= 16000. A sidecar probe in a separate Python process
on the same physical GPU 0 (Qwen3-Embedding-0.6B, identical model and
identical input prompt) stayed under 80 ms throughout an 8-hour run
that produced 5400+ samples. The contamination is process-internal
(httpx/openai SDK + asyncio + PyTorch interaction), not GPU-level.

This module implements a small HTTP server that holds the encoder
exactly once and serves a single endpoint:

  POST /classify
  body:  {"turn_id": str, "text": str, "received_at": float}
  reply: {"turn_id": str (echoed),
          "route": "easy"|"hard",
          "confidence": float,
          "rationale": str,
          "decided_at": float,
          "elapsed_ms": float,
          "queue_wait_ms": float,
          "infer_ms": float,
          "server_threshold": float}

  GET /healthz
  reply: {"status": "ok", "battles": int, "model": str,
          "threshold": float}

Listens on a Unix domain socket by default to avoid TCP loopback
overhead. TCP is also supported for cross-host setups.

Per Codex skeleton review (2026-05-14):
  - encoder forward is serialised via an asyncio.Lock because
    sentence_transformers is not thread-safe (M3)
  - reply schema echoes turn_id + server_threshold so the client can
    reconcile config drift (M4)
  - TCP defaults to 127.0.0.1 (Low)
  - stale UDS detected by trying a connect() before unlinking (Low)

Usage:

    # Unix socket (preferred, no TCP overhead)
    python -m metahuman.adapters_impl.router_rpc_server \\
        --uds /tmp/metahuman-router.sock \\
        --artifacts-dir data/router \\
        --threshold 0.534 \\
        --device cuda:0

    # TCP loopback
    python -m metahuman.adapters_impl.router_rpc_server \\
        --port 8001 \\
        --artifacts-dir data/router \\
        --threshold 0.534 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("metahuman.router_rpc_server")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--uds",
        type=Path,
        default=None,
        help="Unix domain socket path. If set, --host/--port are ignored.",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="TCP bind host (default 127.0.0.1; only override for cross-host).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port (only used if --uds is not set).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("data/router"),
        help="SW Ranking artifacts directory (battles + embeddings + manifest).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hard-route threshold; route='hard' iff strong_win_prob >= threshold.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Embedder device (cuda:0 / cpu / None for autodetect).",
    )
    p.add_argument(
        "--similarity-floor",
        type=float,
        default=0.0,
        help="Clip cosine similarities below this to zero before BT solve.",
    )
    args = p.parse_args()
    if args.uds is None and args.port is None:
        p.error("either --uds <path> or --port <num> is required")
    return args


def _is_socket_alive(uds_path: Path) -> bool:
    """Probe whether an existing UDS path is being held by a live
    server. Fail-closed: only ECONNREFUSED (no listener) is treated
    as stale; timeouts and other errors are treated as alive so we
    do not accidentally unlink a busy socket (Codex impl-review M4).

    Caller is expected to check `S_ISSOCK` before calling; this
    function only handles the connect-probe.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(str(uds_path))
        return True
    except ConnectionRefusedError:
        return False
    except FileNotFoundError:
        return False
    except (socket.timeout, OSError):
        # Treat any other error (timeout, EPERM, etc.) as alive so
        # we do not unlink a socket whose owner we cannot probe.
        return True
    finally:
        s.close()


def _build_router(
    artifacts_dir: Path,
    threshold: float,
    device: Optional[str],
    similarity_floor: float,
) -> Any:
    """Construct a single SWRankingQwen3Router; force eager load of
    embedder + battles by calling _load() so the first request does
    not pay the cold-start cost."""
    from .sw_ranking_qwen3_router import SWRankingQwen3Router

    router = SWRankingQwen3Router(
        artifacts_dir=artifacts_dir,
        hard_threshold=threshold,
        embedding_device=device,
        similarity_floor=similarity_floor,
    )
    router._load()
    return router


def _build_app(router: Any, threshold: float) -> Any:
    """Build aiohttp app with /classify and /healthz endpoints. The
    encoder is serialised by an asyncio.Lock because sentence
    transformers is not thread-safe (Codex review M3)."""
    from aiohttp import web

    encoder_lock = asyncio.Lock()
    model_name = (
        router._encoder.__class__.__name__
        if router._encoder is not None
        else "unknown"
    )

    async def classify_endpoint(request: web.Request) -> web.Response:
        recv_at_server = time.monotonic()
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": f"invalid json: {e}"}, status=400
            )
        turn_id = body.get("turn_id")
        text = body.get("text")
        client_received_at = float(body.get("received_at") or 0.0)
        if not isinstance(text, str):
            return web.json_response(
                {"error": "missing or non-string 'text'"}, status=400
            )
        if not isinstance(turn_id, str):
            return web.json_response(
                {"error": "missing or non-string 'turn_id'"}, status=400
            )

        # Serialise encoder forward (Codex M3)
        async with encoder_lock:
            queue_wait_ms = (time.monotonic() - recv_at_server) * 1000
            infer_t0 = time.monotonic()
            score = await asyncio.to_thread(router._infer, text)
            infer_ms = (time.monotonic() - infer_t0) * 1000

        score = max(0.0, min(1.0, float(score)))
        route = "hard" if score >= router.hard_threshold else "easy"
        elapsed_ms = (time.monotonic() - recv_at_server) * 1000
        rationale = (
            f"sw_qwen3_rpc: strong_win_prob={score:.3f} "
            f"(N={len(router._battles.labels)}, "
            f"latency_ms={elapsed_ms:.1f}, queue_wait_ms={queue_wait_ms:.1f}, "
            f"infer_ms={infer_ms:.1f})"
        )
        # decided_at uses the client's received_at as the per-turn
        # timeline base so the response is drop-in for the existing
        # RouteDecision.decided_at convention.
        decided_at = client_received_at + (elapsed_ms / 1000.0)
        return web.json_response(
            {
                "turn_id": turn_id,
                "route": route,
                "confidence": score,
                "rationale": rationale,
                "decided_at": decided_at,
                "elapsed_ms": elapsed_ms,
                "queue_wait_ms": queue_wait_ms,
                "infer_ms": infer_ms,
                "server_threshold": router.hard_threshold,
            }
        )

    async def healthz_endpoint(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "battles": len(router._battles.labels) if router._battles else 0,
                "model": model_name,
                "threshold": router.hard_threshold,
            }
        )

    app = web.Application()
    app.router.add_post("/classify", classify_endpoint)
    app.router.add_get("/healthz", healthz_endpoint)
    return app


def _run_uds(app: Any, uds_path: Path) -> None:
    """Run the server on a Unix domain socket. Detect and unlink stale
    sockets; refuse to start if a live server is already bound (Codex
    Low finding)."""
    from aiohttp import web

    if uds_path.exists():
        # Refuse to unlink anything that isn't a socket (Codex impl
        # review M4). Prevents accidentally rm'ing a real file or
        # named pipe at the same path.
        import stat as stat_mod
        st = uds_path.stat()
        if not stat_mod.S_ISSOCK(st.st_mode):
            raise SystemExit(
                f"path {uds_path} exists and is not a socket; "
                f"refusing to start"
            )
        if _is_socket_alive(uds_path):
            raise SystemExit(
                f"socket {uds_path} is in use by another process; "
                f"refusing to start"
            )
        logger.info("removing stale socket: %s", uds_path)
        uds_path.unlink()
    uds_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("listening on unix:%s", uds_path)
    web.run_app(app, path=str(uds_path), access_log=None)
    # web.run_app() blocks; on exit, clean up the socket file.
    if uds_path.exists():
        try:
            uds_path.unlink()
        except OSError:
            pass


def _run_tcp(app: Any, host: str, port: int) -> None:
    from aiohttp import web

    logger.info("listening on tcp://%s:%d", host, port)
    web.run_app(app, host=host, port=port, access_log=None)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    router = _build_router(
        args.artifacts_dir, args.threshold, args.device, args.similarity_floor
    )
    n_battles = len(router._battles.labels) if router._battles else 0
    logger.info(
        "router ready: %d battles, threshold=%.3f, device=%s",
        n_battles,
        router.hard_threshold,
        args.device or "auto",
    )
    app = _build_app(router, args.threshold)
    if args.uds is not None:
        _run_uds(app, args.uds)
    else:
        _run_tcp(app, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
