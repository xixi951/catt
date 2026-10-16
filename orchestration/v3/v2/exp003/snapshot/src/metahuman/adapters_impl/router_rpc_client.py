"""HTTP client adapter that satisfies RouterInterface by delegating
to a router_rpc_server running in a separate Python process.

Drop-in replacement for SWRankingQwen3Router. The wire protocol is
defined in router_rpc_server.py.

Why: the in-process SWRankingQwen3Router suffers from process-internal
contamination under heavy DashScope SSE load (Plan D, 2026-05). This
adapter side-steps it by sending each ``classify`` call to a clean
external process that holds the encoder.

Per-call overhead vs. in-process router:
  - aiohttp client request:       ~0.5-2 ms (UDS), ~1-3 ms (TCP)
  - JSON serialisation:            ~0.1 ms
  - Server-side encoder forward:  ~35-70 ms (Plan D sidecar p95 = 51 ms)
  - Total expected:                ~40-75 ms per call
  - vs. in-process worst case:    occasional 6-16 s spikes

Construction:

    # UDS (preferred for local-only setups)
    RouterRPCClient(uds_path="/tmp/metahuman-router.sock", hard_threshold=0.534)

    # TCP loopback
    RouterRPCClient(base_url="http://127.0.0.1:8001", hard_threshold=0.534)

The client lazy-creates aiohttp objects on first ``classify`` call so
that constructing it does not require an event loop (sweep harness
constructs adapters before starting the loop).

Per Codex skeleton review (2026-05-14):
  - UDS handled via aiohttp.UnixConnector + dummy http origin (M1)
  - close() exists; sweep harness wraps construction in try/finally
    and awaits close on shutdown (M2)
  - Wire schema includes turn_id echo + server_threshold; client
    validates the echoed turn_id and warns on threshold drift (M4)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from ..router import RouterInterface
from ..types import RouteDecision, UserUtterance


logger = logging.getLogger("metahuman.router_rpc_client")


# Dummy HTTP origin used with UnixConnector. aiohttp still wants a
# valid http URL even when the underlying transport is a Unix socket;
# the host part is ignored by UnixConnector.
_UDS_DUMMY_BASE = "http://router"


class RouterRPCClient(RouterInterface):
    """Thin RouterInterface implementation backed by an HTTP RPC."""

    def __init__(
        self,
        uds_path: Optional[Path | str] = None,
        base_url: Optional[str] = None,
        hard_threshold: float = 0.5,
        request_timeout_s: float = 30.0,
    ) -> None:
        if (uds_path is None) == (base_url is None):
            raise ValueError(
                "exactly one of uds_path or base_url must be provided"
            )
        self._uds_path: Optional[Path] = Path(uds_path) if uds_path else None
        self._base_url: str = base_url or _UDS_DUMMY_BASE
        self.hard_threshold = float(hard_threshold)
        self._request_timeout_s = float(request_timeout_s)
        self._session: Any = None
        self._session_lock = asyncio.Lock()
        self._threshold_drift_warned = False

    async def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        async with self._session_lock:
            if self._session is not None:
                return self._session
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self._request_timeout_s)
            if self._uds_path is not None:
                connector = aiohttp.UnixConnector(path=str(self._uds_path))
            else:
                connector = aiohttp.TCPConnector()
            self._session = aiohttp.ClientSession(
                base_url=self._base_url,
                connector=connector,
                timeout=timeout,
            )
            return self._session

    async def classify(self, utterance: UserUtterance) -> RouteDecision:
        session = await self._get_session()
        try:
            async with session.post(
                "/classify",
                json={
                    "turn_id": utterance.turn_id,
                    "text": utterance.text,
                    "received_at": utterance.received_at,
                },
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"router rpc returned {resp.status}: {body[:300]}"
                    )
                data = await resp.json()
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"router rpc transport error: {e}") from e

        # Validate echoed turn_id (Codex skeleton M4)
        echoed = data.get("turn_id")
        if echoed != utterance.turn_id:
            raise RuntimeError(
                f"router rpc reply turn_id mismatch: "
                f"sent {utterance.turn_id!r}, got {echoed!r}"
            )

        # Server threshold is required (Codex impl-review M3): a
        # missing field would silently disable drift detection and
        # mask future schema regressions. Server's value is
        # authoritative for the route field per the wire contract.
        if "server_threshold" not in data:
            raise RuntimeError(
                "router rpc reply missing 'server_threshold' field"
            )
        try:
            server_threshold = float(data["server_threshold"])
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"router rpc 'server_threshold' is not a finite float: "
                f"{data['server_threshold']!r}"
            ) from e
        if not (0.0 <= server_threshold <= 1.0):
            raise RuntimeError(
                f"router rpc 'server_threshold' out of [0,1]: "
                f"{server_threshold}"
            )
        if (
            not self._threshold_drift_warned
            and abs(server_threshold - self.hard_threshold) > 1e-6
        ):
            logger.warning(
                "router rpc threshold drift: client=%.3f server=%.3f; "
                "server value is used for route assignment",
                self.hard_threshold,
                server_threshold,
            )
            self._threshold_drift_warned = True

        # Validate route field (Codex impl-review L3).
        route_str = str(data.get("route", ""))
        if route_str not in ("easy", "hard"):
            raise RuntimeError(
                f"router rpc returned invalid route {route_str!r}; "
                f"expected 'easy' or 'hard'"
            )

        return RouteDecision(
            turn_id=utterance.turn_id,
            route=route_str,
            confidence=float(data["confidence"]),
            rationale=str(data["rationale"]),
            decided_at=float(data["decided_at"]),
        )

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
