#!/bin/bash
# nvidia/Qwen3.6-27B-NVFP4 — VISION-ENABLED variant of the text-only 9024 lane.
# Cloned from launch-vllm-27b-nvidia-nvfp4.sh on 2026-07-05 (smoke test: does
# the AEON sm_121a image handle the qwen3_5_vision encoder on GB10?).
#
# Architecture: Qwen3_5ForConditionalGeneration (model_type qwen3_5) — a HYBRID
# multimodal model, 48 linear-attention (Gated DeltaNet) + 16 full-attention
# layers, PLUS a qwen3_5_vision encoder (depth 27, patch 16, Qwen2VL image
# processor). The text lane serves it with --language-model-only; THIS lane
# loads the vision tower so you can send image_url content blocks.
#
# ── DELTA vs the 9024 text lane ──────────────────────────────────────────
#   - REMOVED --language-model-only  → vision encoder is loaded.
#   - ADDED --limit-mm-per-prompt '{"image":4,"video":0}' → up to 4 images per
#     request, video disabled (saves the video-preproc path / KV; re-enable if
#     you want clips). Bump image count if you need more.
#   - Port 9025, container vllm-qwen-27b-nvidia-nvfp4-vision, served-model-name
#     qwen3.6-27b-nvfp4-vision (distinct slot so it can co-exist with 9024).
#   - Context UNCHANGED at 262144 native — the vision encoder is small (~0.4 GB
#     of activations at these dims), so no ctx/concurrency reduction expected.
#     If OOM shows up at load, first knob to drop is --max-model-len, not util.
#
# All the NVFP4-on-sm_121 machinery is identical to the text lane — see that
# script's header for the full Marlin-vs-CUTLASS saga. Same AEON image, same
# VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass override, same DFlash n=10 drafter.
#
# CAVEAT: the qwen3_5 vision preprocessing + encoder path on GB10/sm_121a under
# this image is UNVERIFIED. If images produce garbage or the encoder errors on
# load, that's the thing being tested here — fall back to nemotron-3-nano-omni
# (port 9012) for multimodal in the meantime.
set -e
docker rm -f vllm-qwen-27b-nvidia-nvfp4-vision 2>/dev/null || true
exec docker run --name vllm-qwen-27b-nvidia-nvfp4-vision \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9025:9025 \
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
  ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve nvidia/Qwen3.6-27B-NVFP4 \
  --served-model-name qwen3.6-27b-nvfp4-vision \
  --host 0.0.0.0 --port 9025 \
  --quantization modelopt \
  --mamba-cache-dtype float32 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --attention-backend flash_attn \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}'
