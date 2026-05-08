#!/bin/bash
# Qwen3.6-27B official FP8 + native MTP — production candidate 2026-05-08.
#
# Goal: replace sakamakismile NVFP4+MTP-graft as production due to intermittent
# `</think>C` truncation bug in real opencode coding traffic. Bug is suspected
# to be a combination of (a) the upstream Qwen3.6 chat_template.jinja issue
# documented in sakamakismile model card discussion #9 and (b) NVFP4 tail
# variance at the `</think>` distribution-shift boundary. Switching to the
# official Qwen FP8 with **natively-trained MTP** addresses both:
#
#   - FP8 (block-128 fine-grained) vs NVFP4: removes 4-bit weight noise; quality
#     "nearly identical to original" per Qwen card. ~28 GB on disk vs 19 GB.
#   - Native MTP head (no graft hack): eliminates the MTP-graft fragility in one
#     step. Uses vLLM's `qwen3_next_mtp` method (different from `qwen3_5_mtp`
#     used by sakamakismile graft).
#   - `--reasoning-parser qwen3` (NEW vs sakamaki): vLLM splits <think>...</think>
#     into a separate `reasoning` field, so opencode reads only `content`. Fixes
#     the inline-think parsing ambiguity directly. Costs +3-7s TTFT at short
#     context, irrelevant at the user's 60k+ coding traffic.
#
# Image: vllm/vllm-openai:cu130-nightly — vanilla upstream nightly, NOT the
# AEON-7 DFlash fork. Qwen FP8 path uses stock vLLM features only (qwen3_next_mtp
# is mainline since 0.19); no need for the DFlash drafter image.
#
# Port 9016 (avoids 9008 mtp / 9012 nemotron / 9013 dflash / 9014 clean / 9015 sakamaki).
set -e
docker rm -f vllm-qwen-27b-fp8 2>/dev/null || true
exec docker run --name vllm-qwen-27b-fp8 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9016:9016 \
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
  vllm/vllm-openai:v0.20.1-aarch64-ubuntu2404 \
  Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-fp8 \
  --host 0.0.0.0 --port 9016 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --max-model-len 200000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 32768 \
  --max-cudagraph-capture-size 256 \
  --gpu-memory-utilization 0.85 \
  --async-scheduling \
  -O3 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format safetensors \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only \
  --generation-config vllm \
  --enable-log-requests \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
