#!/usr/bin/env bash
# EXP-20261009-001 (from EXP-20261007-001): two independent vLLM 0.10.2 engines (Qwen3-32B 9216db57, bf16, same KV dtype and parser
# as EXP-20261005-001), each tensor-parallel 4 over the same four GPUs (user allocation: newsrv 2,3,4,5).
#   ROLE=thinker  port 18011: CATT reasoning and both baselines (runs alone).
#   ROLE=talker   port 18012: bridge, progress, final and easy-path talker calls.
# Refuses to start if any process other than this EXP's own engine is on a target GPU.
# Output is captured before matching ('cmd | grep -q' under pipefail misreads busy cards as free).
set -euo pipefail
ROLE=${ROLE:?thinker or talker}
GPUS=${GPUS:-2,3,4,5}
MAXLEN=${MAXLEN:-40960}
MEMUTIL=${MEMUTIL:?per-instance gpu memory utilization}
MAXSEQS=${MAXSEQS:-32}
EAGER=${EAGER:-0}
ENV=/data/userhome/xi/miniconda3/envs/metahuman
WEIGHTS=/data/userhome/xi/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
BASE=/data/userhome/xi/exp20261009
OUT=$BASE/logs
mkdir -p "$OUT"
case "$ROLE" in thinker) PORT=18011 ;; talker) PORT=18012 ;; *) echo "bad ROLE"; exit 2 ;; esac
TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OWN_PIDS=$(cat "$OUT"/engine_*.pid 2>/dev/null | tr '\n' ' ' || true)
GPU_TABLE=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits) || { echo "REFUSED: nvidia-smi failed"; exit 3; }
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader) || { echo "REFUSED: nvidia-smi failed"; exit 3; }
# Descendants (workers) of this EXP's own engines; any other process on a target GPU refuses the launch.
PS_TABLE=$(ps -eo pid=,ppid=) || { echo "REFUSED: ps failed"; exit 3; }
OWN_TREE=$(awk -v roots="$OWN_PIDS" 'BEGIN{n=split(roots,r," "); for(i=1;i<=n;i++) own[r[i]]=1}
  {par[$1]=$2} END{changed=1; while(changed){changed=0; for(p in par) if(!(p in own) && (par[p] in own)){own[p]=1; changed=1}}
  for(p in own) printf "%s ", p}' <<< "$PS_TABLE")
for idx in $(echo "$GPUS" | tr ',' ' '); do
  uuid=$(awk -F', ' -v i="$idx" '$1==i{print $2}' <<< "$GPU_TABLE")
  if [ -z "$uuid" ]; then echo "REFUSED: GPU $idx not found"; exit 3; fi
  while IFS=', ' read -r au apid amem; do
    [ -z "$au" ] && continue
    if [ "$au" = "$uuid" ]; then
      if [[ " $OWN_TREE " != *" $apid "* ]]; then echo "REFUSED: GPU $idx has foreign process $apid ($amem)"; exit 3; fi
    fi
  done <<< "$APPS"
done
if [ "${DRYRUN:-0}" = "1" ]; then echo "GUARD OK: GPUs $GPUS free of foreign processes (dry run, nothing started)"; exit 0; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "${MPS:-0}" = "1" ]; then
  # Client of this EXP's own MPS daemon (start_mps.sh), whose CUDA_VISIBLE_DEVICES is $GPUS; client
  # device indices are relative to the daemon's set.
  export CUDA_MPS_PIPE_DIRECTORY=$BASE/mps/pipe CUDA_MPS_LOG_DIRECTORY=$BASE/mps/log
  [ -S "$CUDA_MPS_PIPE_DIRECTORY/control" ] || { echo "REFUSED: MPS daemon not running"; exit 3; }
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1)))
  # v4 change 2: an MPS client may be limited to a share of the SMs; set only for the talker engine
  if [ -n "${MPS_PCT:-}" ]; then export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT; fi
else
  export CUDA_VISIBLE_DEVICES=$GPUS
fi
export HF_HUB_OFFLINE=1
export VLLM_SERVER_DEV_MODE=1
export PATH=$ENV/bin:$PATH
export HF_HOME=/data/userhome/xi/.cache/huggingface
TAG=$(date +%Y%m%dT%H%M%S)
CMD=("$ENV/bin/vllm" serve "$WEIGHTS" --served-model-name Qwen3-32B --host 127.0.0.1 --port $PORT
     --dtype bfloat16 --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --reasoning-parser qwen3
     --gpu-memory-utilization "$MEMUTIL" --max-num-seqs "$MAXSEQS" --seed 0)
if [ "$EAGER" = "1" ]; then CMD+=(--enforce-eager); fi
printf '%q ' "${CMD[@]}" > "$OUT/engine_${ROLE}_${TAG}.cmd"; echo >> "$OUT/engine_${ROLE}_${TAG}.cmd"
env | grep -E '^(CUDA_|HF_|VLLM_|NCCL_)' | sort > "$OUT/engine_${ROLE}_${TAG}.env"
nohup "${CMD[@]}" > "$OUT/engine_${ROLE}_${TAG}.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/engine_${ROLE}.pid"
echo "started $ROLE pid $! on GPUs $GPUS (TP$TP) port $PORT log $OUT/engine_${ROLE}_${TAG}.log"
