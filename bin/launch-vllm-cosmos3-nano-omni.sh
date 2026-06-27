#!/bin/bash
# Cosmos3-Nano (NVIDIA) omni world model via vLLM-Omni — EVAL slot, added 2026-06-27.
#
# NVIDIA Cosmos3-Nano: ~15B omni-modal world model (text / image / video /
# audio / action) for Physical AI. This is the SMALL Cosmos3 variant — it fits
# in the 119 GB unified pool (~30 GB BF16) and runs natively on the Spark,
# unlike Cosmos3-Super (64B, multi-GPU only → stays standalone in ~/cosmos3).
#
# WHY in the gateway: web research 2026-06-27 found vLLM-Omni now ships an
# OpenAI-compatible server for Cosmos3 with an aarch64/GB10 image, so Nano can
# be a first-class swap lane instead of a standalone script. See memory
# project_cosmos3_serving.md.
#
# ENDPOINTS (NOT chat/completions — this is a generation model):
#   - POST /v1/videos        async text-to-video / image-to-video (multipart)
#   - POST /v1/videos/sync   synchronous video gen (benchmark-oriented)
#   - POST /v1/images/generations   text-to-image (OpenAI image API shape)
# Example:
#   curl -s http://192.168.1.12:8079/v1/videos/sync \
#     -F prompt="a cinematic tracking shot of a mountain lake at sunrise" \
#     -F width=1280 -F height=720 -F num_frames=80 -F fps=16 \
#     -F num_inference_steps=40 --output out.mp4
#
# Image: vllm/vllm-omni:cosmos3-aarch64 — the Cosmos3 vLLM-Omni build for
# aarch64 (Grace/GB10). The generic vllm-openai omni tags do NOT carry the
# Cosmos3 generator path; do not substitute.
#
# Serves the LOCAL copy at ~/cosmos3/models/Cosmos3-Nano (33 GB, already on
# disk) mounted read-only, so no HF re-download. To pull fresh instead, swap
# the served path for the HF id nvidia/Cosmos3-Nano.
#
# MEMORY: swap-exclusive `experiments` member — loads alone with the full
# 119 GB and evicts whatever was resident. Port 9023 (next free above 9022).
set -e
docker rm -f vllm-cosmos3-nano-omni 2>/dev/null || true
exec docker run --name vllm-cosmos3-nano-omni \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9023:9023 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/cosmos3/models/Cosmos3-Nano:/models/Cosmos3-Nano:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  vllm/vllm-omni:cosmos3-aarch64 \
  /models/Cosmos3-Nano \
  --omni \
  --served-model-name cosmos3-nano-omni \
  --host 0.0.0.0 --port 9023 \
  --trust-remote-code \
  --init-timeout 1800
