#!/bin/bash
# Qwen3.6-27B sakamakismile clean instruct + MTP head (NVFP4 modelopt) — eval slot.
#
# Goal of this launch: A/B against prod's AEON-7 + DFlash on real-traffic-shaped
# coding prompts (13k / 60k / 100k input). Sakamakismile is the modelopt-format
# NVFP4 quant with the 1-layer Qwen3.6 MTP head explicitly grafted back in bf16
# — vLLM's qwen3_5_mtp speculator handler then drives n=3 in-target speculation.
# Per the model card, per-position acceptance ~87 / 72 / 61 % (vs DFlash's 51 /
# 28 / 16 % on this user's real opencode traffic).
#
# Image: ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 — same image as the
# DFlash prod launcher and the dormant MTP launcher. Known-working with
# qwen3_5_mtp method.
#
# Three deliberate deltas vs prod:
#   1. --quantization modelopt           (sakamakismile is modelopt-format, NOT
#                                         compressed-tensors; per model-card
#                                         discussion #5 the modelopt path is
#                                         meaningfully faster on Blackwell SM120)
#   2. --kv-cache-dtype fp8              (needed for headroom at 200K context
#                                         with the with-MTP-head variant)
#   3. --max-num-seqs 2                  (model card warning: max-num-seqs 4 +
#                                         fp8 KV + spec n=3 + long ctx silently
#                                         OOMs during cuda-graph capture on
#                                         vLLM 0.19.x. Conservative for the
#                                         bench, revisit if MTP wins production.)
#
# Same as prod (per AEON-7 image conventions):
#   - NO --reasoning-parser qwen3        (buffered <think> adds ~3 s TTFT — memory)
#   - --language-model-only              (text-only workload, frees mm encoder mem)
#   - --attention-backend flash_attn     (Qwen3.6 hybrid; non-flash layers fall back)
#   - Spark/Blackwell env vars (TORCH_CUDA_ARCH_LIST=12.1a, NVFP4 SM100 off)
#
# Cold start: ~10–12 min on first boot (FlashInfer NVFP4 GEMM autotuner +
# CUDA-graph capture; cached to /root/.cache/vllm).
#
# Port 9015 (avoids 9013 prod / 9008 mtp-dormant / 9014 clean-dflash).
set -e
docker rm -f vllm-qwen-27b-sakamaki-mtp 2>/dev/null || true
exec docker run --name vllm-qwen-27b-sakamaki-mtp \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9015:9015 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 \
  bash -c '
    exec vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
      --served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-sakamaki-mtp \
      --host 0.0.0.0 --port 9015 \
      --tensor-parallel-size 1 \
      --dtype auto \
      --quantization modelopt \
      --kv-cache-dtype fp8 \
      --max-model-len 200000 \
      --max-num-seqs 2 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format safetensors \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --language-model-only \
      --generation-config vllm \
      --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}"
  '
