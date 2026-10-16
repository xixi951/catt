#!/usr/bin/env bash
# EXP-20261009-001 (from EXP-20261007-001): user-level CUDA MPS control daemon for this EXP's GPUs only (default 2,3,4,5).
# Only processes that set this EXP's CUDA_MPS_PIPE_DIRECTORY connect to it. Same foreign-process guard
# as launch_engines.sh. Stop with: CUDA_MPS_PIPE_DIRECTORY=... echo quit | nvidia-cuda-mps-control
set -euo pipefail
GPUS=${GPUS:-2,3,4,5}
BASE=/data/userhome/xi/exp20261009
OUT=$BASE/logs
mkdir -p "$BASE/mps/pipe" "$BASE/mps/log" "$OUT"
GPU_TABLE=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits) || { echo "REFUSED: nvidia-smi failed"; exit 3; }
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader) || { echo "REFUSED: nvidia-smi failed"; exit 3; }
for idx in $(echo "$GPUS" | tr ',' ' '); do
  uuid=$(awk -F', ' -v i="$idx" '$1==i{print $2}' <<< "$GPU_TABLE")
  used=$(awk -F', ' -v i="$idx" '$1==i{print $3}' <<< "$GPU_TABLE")
  if [ -z "$uuid" ]; then echo "REFUSED: GPU $idx not found"; exit 3; fi
  if [[ "$APPS" == *"$uuid"* ]] || [ "$used" -gt 1000 ]; then echo "REFUSED: GPU $idx in use (${used} MiB)"; exit 3; fi
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS
export CUDA_MPS_PIPE_DIRECTORY=$BASE/mps/pipe CUDA_MPS_LOG_DIRECTORY=$BASE/mps/log
nvidia-cuda-mps-control -d
sleep 1
pgrep -n -u "$(id -u)" -f "^nvidia-cuda-mps-control" > "$OUT/engine_mps.pid"
echo "MPS control daemon pid $(cat $OUT/engine_mps.pid) for GPUs $GPUS"
