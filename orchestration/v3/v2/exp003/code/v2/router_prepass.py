"""Score all 220 fixture items with the router RPC service (CPU) before generation.
Router scores are deterministic properties of the pre-specified router, not outcomes of the
new run; this file supports threshold replay (R1-2) and checks router identity against May.
Usage: python router_prepass.py CONFIG OUT.jsonl
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2 as R  # noqa: E402
from metahuman.adapters_impl.router_rpc_client import RouterRPCClient  # noqa: E402
from metahuman.types import AudioMeta, UserUtterance  # noqa: E402
import baseline_direct_llm as B  # noqa: E402


async def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    questions = B.load_fixture(R.SNAPSHOT / cfg['fixture'])
    rpc = RouterRPCClient(uds_path=cfg['router_uds'], hard_threshold=cfg['router_threshold'])
    with open(sys.argv[2], 'w') as out:
        for q in questions:
            t0 = time.monotonic()
            d = await rpc.classify(UserUtterance(turn_id=f'prepass/{q.id}', text=q.question,
                                                 audio_meta=AudioMeta(0.0, 16000, 0.0, 0.0), received_at=0.0))
            out.write(json.dumps({'question_id': q.id, 'source': q.source, 'strong_win_prob': d.confidence,
                                  'route': d.route, 'seconds': time.monotonic() - t0}) + '\n')
    close = getattr(rpc, 'close', None)
    if close:
        r = close()
        if asyncio.iscoroutine(r):
            await r


asyncio.run(main())
