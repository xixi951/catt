#!/usr/bin/env bash
# EXP-20261009-001 (from EXP-20261007-001): recovered router RPC service on CPU (no GPU), tau* = 0.5444878691642917
# (EXP-20261002-001, unrounded), same artifacts and settings as v3 (read-only copy under pkg/v3/v2/exp003).
set -euo pipefail
ENV=/data/userhome/xi/miniconda3/envs/metahuman
PKG=/data/userhome/xi/exp20261009/pkg/v3/v2
OUT=/data/userhome/xi/exp20261009/logs
SOCK=${SOCK:-/tmp/exp20261009-router.sock}
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/data/userhome/xi/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PKG/exp003/snapshot/src
TAG=$(date +%Y%m%dT%H%M%S)
nohup "$ENV/bin/python" -u -m metahuman.adapters_impl.router_rpc_server --uds "$SOCK" \
  --artifacts-dir "$PKG/exp003/snapshot/data/router" --threshold 0.5444878691642917 --device cpu --similarity-floor 0.0 \
  > "$OUT/router_${TAG}.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/router.pid"
echo "started router pid $! sock $SOCK log $OUT/router_${TAG}.log"
