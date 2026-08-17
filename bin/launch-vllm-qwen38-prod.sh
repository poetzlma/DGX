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
# VLLM_MARLIN_USE_ATOMIC_ADD=1 — added 2026-08-17 (decisions.md §44). There is a
#            race in the Marlin kernel on SM121 that yields INCORRECT OUTPUT
#            rather than an error. Our 4-bit weights dequantize through Marlin,
#            so it sits directly in the decode path. A race is intermittent, so
#            the "output verified clean" note above does NOT clear it — a clean
#            smoke run is not evidence of absence. Measured cost: none. On the
#            code prompt, with vs without was 26.73 vs 26.69 tok/s, i.e. inside
#            noise; an earlier apparent +16% was engine warmup, not this flag.
#            Treat as a correctness fix with no performance component.
# KV fp8   : per the official vLLM recipe (live 2026-08-16). Halves KV bytes
#            = long-ctx decode + pool gains. flash_attn refuses fp8 KV, so the
#            attention-backend pin is DROPPED — vLLM auto-picks an fp8-capable
#            backend. If startup fails on attention selection, revert BOTH
#            (kv bfloat16 + --attention-backend flash_attn) together.
# BATCH 16384, SEQS 8: raised from 4 on 2026-08-17 for a live coding-agent test.
#            Measured at SEQS=4 on a code prompt: c=4 aggregate 93.67 tok/s vs
#            c=1 26.69 (3.5x scaling) with per-request barely down (25.13), so
#            c=4 was clearly not the ceiling at short/moderate context. The
#            §42-era "aggregate flat past c=2" was measured at 120k, NOT
#            universally — see §44.
#            WHY THIS IS SAFE-ISH: the KV pool is allocated ONCE at startup and
#            does not grow with SEQS; on this hybrid model the SSM state is paged
#            INSIDE that pool (vLLM forces attention block size to 1664 tokens so
#            the attention page >= mamba page). Raising SEQS divides a fixed pool
#            among more sequences, and vLLM re-profiles at startup and shrinks the
#            pool to fit the larger activation. So this is not the memory-growth
#            shape that hard-hangs the box.
#            TWO REAL RISKS, both invisible to health checks:
#            1. SEQS=8 DEADLOCKED the laguna lane twice (token counter frozen
#               while /health returned 200). Different model+drafter, so it may
#               not reproduce — but if completions stall while the process looks
#               healthy, this is why. Roll back FIRST, diagnose after.
#            2. The pool holds 5.83 full-256k requests. c=8 CROSSES that, so
#               eight genuinely long requests will preempt/recompute. Watch
#               vllm:num_preemptions_total (0 at SEQS=4). At ~100k prompts the
#               pool holds ~15, so this only bites the long tail.
#            ROLLBACK: set this back to 4 and restart the engine. One line.
# VISION ENABLED 2026-08-16 (was --language-model-only at first cutover):
#            the model is natively multimodal and the stack had NO vision lane
#            since laguna went dark 2026-08-01. limit-mm image:4/video:1 is the
#            community-validated setting for this model on GB10. Encoder costs
#            ~2-4 GB from the same UTIL budget (KV pool shrinks slightly).
#            3.6-era caveat to watch: thinking-mode spirals on image inputs
#            (fix was enable_thinking=false per request) — verify on 3.8.
# --reasoning-parser qwen3: per the official recipe (both its configs use it).
#            Separates thinking into reasoning_content. The 3.6-era buffering
#            concern (feedback_reasoning_parser_ttft) was re-tested on this
#            engine at deploy — see decisions.md §42 addendum for the verdict.
#            NO --generation-config flag: the model's generation_config.json
#            (temp 1.0 / top_p 0.95 / top_k 20 thinking-mode) must be the
#            default; `--generation-config vllm` silently replaced it with
#            vLLM generics — that was a copy-through bug from the 3.6 lanes.
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
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  --entrypoint vllm \
  vllm-qwen38:prod-20260816 \
  serve /home/max/models/qwen3.8-27b-nvfp4 \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 9030 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.70 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --mamba-cache-dtype float32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":4,"video":1}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
