#!/bin/bash
# Nemotron-3-Puzzle-75B-A9B-NVFP4 — PHASE 2: MTP speculative decoding + CUDA graphs.
#
# Phase-1 (launch-vllm-nemotron-puzzle-75b.sh) proved it loads & runs but was
# eager-only (Mamba sm121 crash workaround) with MTP off -> 17.8/58.8 short,
# 16.6/10.7 @125k: beaten by the 35B MoE lane. This variant chases the two levers:
#   1. MTP: --speculative-config mtp n=1 (config declares num_nextn_predict_layers=1,
#      mtp.safetensors auto-loads; no num_nextn patch needed unlike Qwen).
#   2. CUDA graphs: DROP --enforce-eager + --no-async-scheduling, betting AEON
#      v0.23 sm121a fixed the Mamba graph-capture crash (#37431 / Nemotron #125)
#      that forced eager. If it crashes during graph capture -> fall back to the
#      eager+MTP variant (re-add --enforce-eager; MTP still helps via accept rate).
set -e
docker rm -f vllm-nemotron-puzzle-75b 2>/dev/null || true
exec docker run --name vllm-nemotron-puzzle-75b \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9027:9027 \
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
  serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
  --served-model-name nemotron-3-puzzle-75b \
  --host 0.0.0.0 --port 9027 \
  --quantization modelopt \
  --trust-remote-code \
  --mamba-cache-dtype float32 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization ${NEMO_UTIL:-0.85} \
  --max-model-len ${NEMO_CTX:-131072} \
  --max-num-seqs ${NEMO_SEQS:-8} \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --generation-config vllm \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
  # --reasoning-parser nemotron_v3 (added 2026-07-08): splits <think>…</think>
  # into the OpenAI reasoning_content field so coding clients read a clean
  # `content`. Costs a few seconds of TTFT — it buffers the think block before
  # emitting (see memory feedback_reasoning_parser_ttft). Drop this flag to
  # revert to raw inline <think> streaming.
