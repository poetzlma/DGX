#!/bin/bash
# EVAL ONLY — Qwen3.8-27B NVFP4 with the DSpark drafter instead of native MTP.
# NOT production. Prod is bin/launch-vllm-qwen38-prod.sh (MTP n=3, port 9030).
#
# WHY: 0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark reports, on the SAME
# weights (unsloth NVFP4) and the same engine family (vLLM) on a GB10:
#
#   workload            FP8+DSpark k7   NVFP4+DSpark k7   NVFP4+DSpark k14
#   fresh gen (400 tok)     20.00            29.23             29.55
#   edit-heavy (3k tok)     45-47            58.6-59.1         72.6-75.0
#
# Our prod MTP n=3 measures 18.01 tok/s on the same fresh-generation shape
# (3-run median, 2026-08-17, bench in scratchpad). If their number reproduces,
# DSpark is worth ~1.6x on fresh generation and far more on edit-heavy work —
# which is what our ~100k:4k coding traffic actually looks like.
#
# TWO ENV VARS THEY CALL NON-OPTIONAL ON SM121 — note our prod sets NEITHER:
#   VLLM_MARLIN_USE_ATOMIC_ADD=1  a documented Marlin race on SM121 yields
#                                 INCORRECT OUTPUT (not an error). 4-bit weights
#                                 dequant through Marlin, so it is in the decode
#                                 path. A race looks exactly like "bad quant".
#   VLLM_USE_FLASHINFER_MOE_FP4=0 keeps MoE off CUTLASS FP4 (silent garbage on
#                                 this arch). Unused by this dense model; cheap.
#
# K: draft depth. k=14 wins both workloads for them but needs the batch budget
# raised (draft slots come out of it; k * max_num_seqs over budget makes
# max_num_scheduled_tokens negative at startup). We already run 16384.
# Their §3 warns k=14 is NOT simply better once concurrency rises.
#
# UTIL: they run 0.85. WE DO NOT. Our pin is 0.70 — 0.80+ logged 60x NVRM
# NV_ERR_NO_MEMORY during warmup, the precursor to host-OOM hard-hang #3
# (2026-08-15, power cycle). Overriding is a deliberate act, not a default.
#
# Usage:  bin/park-prod-ds4.sh   # park prod FIRST — two engines OOM-hang the box
#         K=7 bin/eval-qwen38-dspark.sh
#         (bench, then) docker rm -f vllm-qwen38-dspark ; bin/restore-prod-ds4.sh
set -e
NAME="vllm-qwen38-dspark"
PORT="${PORT:-9031}"
K="${K:-7}"
UTIL="${UTIL:-0.70}"
SPEC="${SPEC:-dspark}"          # dspark | mtp | off
IMAGE="${IMAGE:-vllm-qwen38:prod-20260816}"
DRAFTER="${DRAFTER:-Doopeworld/Qwen3.8-27B-DSpark-vLLM}"

# ---- guards: never a second big engine on this box ----
if pgrep -x ds4-server >/dev/null 2>&1 || ss -ltn 2>/dev/null | grep -q ':9010[[:space:]]'; then
  echo "REFUSING: ds4 resident on 9010. Run bin/park-prod-ds4.sh first." >&2; exit 1
fi
if ss -ltn 2>/dev/null | grep -q ':9030[[:space:]]'; then
  echo "REFUSING: prod qwen3.8 is live on 9030. Two engines OOM-hang this box." >&2
  echo "Run bin/park-prod-ds4.sh first (it also clears the startup preload)." >&2; exit 1
fi
others=$(docker ps --format '{{.Names}}' 2>/dev/null | grep vllm | grep -v "^${NAME}$" || true)
[ -n "$others" ] && { echo "REFUSING: another vLLM engine running: $others" >&2; exit 1; }
avail_gb=$(free -g | awk '/^Mem:/{print $7}')
[ "$avail_gb" -lt 60 ] && { echo "REFUSING: only ${avail_gb} GB available." >&2; exit 1; }

case "$SPEC" in
  dspark) SPEC_CFG="{\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}" ;;
  mtp)    SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$K}" ;;
  off)    SPEC_CFG="" ;;
  *)      echo "SPEC must be dspark|mtp|off" >&2; exit 2 ;;
esac

echo "launching $NAME :: spec=$SPEC k=$K util=$UTIL port=$PORT image=$IMAGE"
docker rm -f "$NAME" 2>/dev/null || true

ARGS=(
  serve /home/max/models/qwen3.8-27b-nvfp4
  --served-model-name qwen3.8-27b-dspark
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$UTIL"
  --max-model-len 262144
  --max-num-seqs 4
  --max-num-batched-tokens 16384
  --kv-cache-dtype fp8
  --mamba-cache-dtype float32
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --trust-remote-code
  --limit-mm-per-prompt '{"image":4,"video":1}'
)
[ -n "$SPEC_CFG" ] && ARGS+=( --speculative-config "$SPEC_CFG" )

exec docker run --name "$NAME" \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:"$PORT":"$PORT" \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/models:/home/max/models:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  --entrypoint vllm \
  "$IMAGE" "${ARGS[@]}"
