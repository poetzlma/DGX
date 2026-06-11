#!/bin/bash
# DiffusionGemma 26B-A4B-it NVFP4 (NVIDIA ModelOpt) — EVAL slot, added 2026-06-11.
#
# Google DeepMind's first diffusion LLM (dLLM), released 2026-06-10 on the
# Gemma 4 MoE backbone: 26B total / ~3.8B active. Generates by iteratively
# refining 256-token "canvas" blocks (up to 48 denoising steps) instead of
# autoregressive token-by-token. NVIDIA published ~150 tok/s single-stream on
# DGX Spark — the reason this slot exists (clears the >100 tok/s bar the
# bandwidth-bound dense 27B never could). User wants it for NON-CODING tasks;
# Google notes output quality is below AR Gemma 4, so it's a speed/breadth
# lane, not a coding-quality lane.
#
# Serve command is from the model card (nvidia/diffusiongemma-26B-A4B-it-NVFP4):
#   - VLLM_USE_V2_MODEL_RUNNER=1   diffusion uses model-runner-v2 ModelState
#   - --attention-backend TRITON_ATTN   (card-specified; FlashAttn path n/a)
#   - --max-num-seqs 4             MANDATORY — diffusion state buffers are heavy
#   - --tool-call-parser gemma4 / --reasoning-parser gemma4
#   - NVFP4 quant auto-detected from config (ModelOpt format); no --quantization
#
# Image: vllm/vllm-openai:gemma-aarch64-cu130 — the day-0 (2026-06-10) gemma
# build for aarch64 (Grace) + CUDA 13. The generic cu130-nightly tag is stale
# (deprecated upstream, frozen at 2026-04-23) and does NOT have the diffusion
# runner — do not substitute it.
#
# MEMORY: this is in the `main` swap-exclusive group. The qwen co-resident pair
# (~104 GB) leaves only ~15 GB free, so triggering this WILL need the pair
# freed first (docker rm -f vllm-qwen-27b-int4-dflash vllm-qwen-35b-moe-nvfp4).
# Port 9021 (next free above the 9008–9020 range).
set -e
docker rm -f vllm-diffusiongemma-nvfp4 2>/dev/null || true
exec docker run --name vllm-diffusiongemma-nvfp4 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9021:9021 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  vllm/vllm-openai:gemma-aarch64-cu130 \
  nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
  --served-model-name diffusiongemma-26b \
  --host 0.0.0.0 --port 9021 \
  --trust-remote-code \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --override-generation-config '{"max_new_tokens": null}' \
  --default-chat-template-kwargs '{"enable_thinking":true}'
