#!/bin/bash
# Qwen3.8-27B NVFP4 — PRODUCTION launcher. Staged 2026-08-16, cut over: (not yet).
# This is the pinned, no-env-vars twin of launch-vllm-qwen38-27b.sh (the eval
# launcher, which keeps its knobs). Change THIS file only via a measured A/B.
#
# ── WHY EACH VALUE (all measured on this box, 2026-08-14..16) ────────────
# Weights  : local dir (22.4 GB unsloth NVFP4, compressed-tensors). Mixed quant:
#            FP8 attn/lm_head + 4-bit MLPs. MMLU-recovery and our smoke are
#            clean (13/14, 0 hard fails, logs/smoke-qwen38-nvfp4-tuned.json).
# Image    : vllm-qwen38:prod-20260816 = nightly-aarch64 frozen at digest
#            sha256:677afd5bf3b4...  NEVER point prod at the mutable nightly tag.
# UTIL 0.70: 0.80 produced 60x NVRM NV_ERR_NO_MEMORY during warmup (the hang
#            precursor — box hard-hung 2026-08-15 from this state). 0.70 = 0 hits
#            and matches the Qwen3.6 concurrency work. KV pool at 0.70 is still
#            841,870 tok = 3.21x full-256k requests.
# MTP n=3  : +25-50% decode vs n=1 (83-90% draft acceptance measured). No
#            DFlash/Eagle drafter exists for 3.8; the native head is the play.
# Marlin   : default backend. Do NOT set VLLM_NVFP4_GEMM_BACKEND. sm_121 has no
#            usable native-FP4 path in vLLM today; Marlin dequant->BF16 is the
#            community-converged fastest, and output is verified clean (the
#            3.6-era "Marlin garbage" applied to MODELOPT packing, not this).
# KV bf16  : flash_attn refuses kv-cache-dtype=auto on this quant
#            ("kv_cache_dtype not supported"). fp8 KV untested — potential
#            future win at long ctx, A/B before touching.
# BATCH 16384, SEQS 4: community-validated; SEQS=2 A/B pending (task #5) —
#            aggregate is flat past c=2, so SEQS may drop to 2 after the test.
# NO --reasoning-parser: thinking arrives inline in content (blank-line
#            separated on this engine; no <think> tag leak on 0.27.2). Parser
#            ON buffers the think block (feedback_reasoning_parser_ttft).
#            DECISION PENDING — A/B streaming behavior before cutover.
#
# MEASURED (single-run sweep; matrix medians in logs/q38-matrix-final.log):
#   ctx      prefill   decode   coldTTFT
#   2k       2,065     20.9      1.1s
#   19k      1,351     25.8     13.9s
#   70k      1,080     18.4     65s
#   142k       861     13.2    165s          (sustained-1k-out: 10.5)
#   260k       663      9.6    392s
# vs ds4 incumbent: 18.0 tok/s @34.6k, ctx cap 131072, cold TTFT ~370s @100k.
#
# OOM RULES (this box hard-hangs, 3 power cycles to date):
#   - never start while ds4 or any other engine is resident (guards below)
#   - never run downloads/page-cache-heavy jobs during load or bench
#   - after launch check: journalctl -k --since <launch> | grep NV_ERR_NO_MEMORY
set -e
NAME="vllm-qwen38-27b"

ds4_running=0
pgrep -x ds4-server >/dev/null 2>&1 && ds4_running=1
ss -ltn 2>/dev/null | grep -q ':9010[[:space:]]' && ds4_running=1
if [ "$ds4_running" = "1" ]; then
  echo "REFUSING: ds4 resident on 9010. Two engines OOM-hang this box." >&2
  echo "Run bin/park-prod-ds4.sh first." >&2
  exit 1
fi
others=$(docker ps --format '{{.Names}}' 2>/dev/null | grep vllm | grep -v "^${NAME}$" || true)
[ -n "$others" ] && { echo "REFUSING: another vLLM engine running: $others" >&2; exit 1; }
avail_gb=$(free -g | awk '/^Mem:/{print $7}')
[ "$avail_gb" -lt 60 ] && { echo "REFUSING: only ${avail_gb} GB available." >&2; exit 1; }

docker rm -f "$NAME" 2>/dev/null || true
exec docker run --name "$NAME" \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9030:9030 \
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
  --entrypoint vllm \
  vllm-qwen38:prod-20260816 \
  serve /home/max/models/qwen3.8-27b-nvfp4 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 9030 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype bfloat16 \
  --mamba-cache-dtype float32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --attention-backend flash_attn \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  --language-model-only \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
