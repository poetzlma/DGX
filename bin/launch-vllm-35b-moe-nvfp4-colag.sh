#!/bin/bash
# Qwen3.6-35B-A3B NVFP4 — FAST LANE, co-resident with Laguna S-2.1 (2026-07-24).
#
# Derived from launch-vllm-35b-moe-nvfp4.sh (Scargall recipe) with three changes:
#   1. Weights: unsloth/Qwen3.6-35B-A3B-NVFP4-Fast — the RedHatAI NVFP4 repo was
#      evicted in the 2026-07-13 offload; the unsloth "-Fast" NVFP4 is what's in
#      the local HF cache (23 GB, same compressed-tensors packaging).
#   2. --gpu-memory-utilization 0.24 (was 0.70 solo): co-resident budget.
#      Laguna runs at 0.66 / ctx 131072; sum 0.90 x 119.7 = ~109 GB steady,
#      ~10 GB host headroom. Was 0.25 until 2026-07-24 09:18 when the OOM
#      guard killed this lane at MemAvailable=4GB during a laguna LOAD peak —
#      co-load transients are ~3-4 GB above steady state. Operational rule:
#      whenever laguna reloads, this lane may be sacrificed by the guard and
#      relaunches on demand via llama-swap. DO NOT raise either util without
#      lowering the other — the GB10 hard-hangs on host OOM (needs a power
#      cycle, no remote recovery).
#   3. --max-model-len 65536 / seqs 4 / batched 8192: KV pool at this util is
#      ~3.4 GB = ~71k fp8 tokens (2 kv-heads x 256 head-dim = ~48 KB/token).
#      This lane is for SHORT/MEDIUM fast requests (fan-out subtasks, <=16k x4);
#      it holds at most ONE ~48k-prompt session at a time. Big-context traffic
#      belongs on laguna.
#
# Role: speed lane for multi-agent fan-out (Shannon et al) + coding default for
# small prompts. Bench basis (RedHat weights, 2026-05-17): 56 tok/s c=1 short,
# 169 tok/s c=4 short. Re-verify on unsloth weights.
#
# IMAGE: v0.25.1-aarch64 (same as laguna), NOT the cu130-nightly-20260423 the
# solo script used — the old nightly (vLLM 0.19.2) can't load the unsloth
# checkpoint's quantized lm_head ("no parameter named lm_head.weight_scale",
# first-start failure 2026-07-24 08:20). On v0.25.1: --moe-backend left auto
# (as laguna does on GB10) + VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass env.
set -e
docker rm -f vllm-qwen-35b-moe-nvfp4 2>/dev/null || true
exec docker run --name vllm-qwen-35b-moe-nvfp4 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9019:9019 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/llm-stack/etc:/llm-stack-etc:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint vllm \
  vllm/vllm-openai:v0.25.1-aarch64-ubuntu2404 \
  serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name qwen3.6-35b-a3b-nvfp4 \
  --host 0.0.0.0 --port 9019 \
  --quantization compressed-tensors \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.24 \
  --max-model-len 49152 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
