#!/usr/bin/env bash
# EXP-20260930-003 v2: router RPC service, recovered snapshot, CPU, threshold 0.55 (as in May).
set -euo pipefail
ENV=/data/userhome/xi/miniconda3/envs/metahuman
PKG=/data/userhome/xi/exp003_matched/pkg
OUT=/data/userhome/xi/exp003_matched/logs
SOCK=/tmp/exp003v2-router.sock
export CUDA_VISIBLE_DEVICES=""
export HF_HOME=/data/userhome/xi/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONPATH=$PKG/snapshot/src
export PYTHONDONTWRITEBYTECODE=1
TAG=$(date +%Y%m%dT%H%M%S)
nohup "$ENV/bin/python" -u -m metahuman.adapters_impl.router_rpc_server --uds "$SOCK" \
  --artifacts-dir "$PKG/snapshot/data/router" --threshold 0.55 --device cpu --similarity-floor 0.0 \
  > "$OUT/router_${TAG}.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/router_${TAG}.pid"
echo "started router pid $(cat $OUT/router_${TAG}.pid) sock $SOCK"
