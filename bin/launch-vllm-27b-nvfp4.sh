#!/bin/bash
# Production launcher — Qwen3.6-27B-NVFP4 (solo, 2026-04-24).
#
# Replaces the previous MoE+GGUF co-resident pair. MoE is disabled in
# llama-swap.yaml (kept commented for reactivation on other hardware).
#
# Decisions:
#   - Solo → full 119 GiB budget, --gpu-memory-utilization 0.85 (~101 GB, ~18 GB OS headroom).
#   - 131072 ctx: ample headroom with MoE gone (worst-case KV at c=10 is
#     ~16 GB * per-user-ctx-fraction; paged-attn handles eviction).
#   - NO --reasoning-parser qwen3: raw <think> streamed to client.
#     opencode parses <think> natively; buffered parser added 2-5 s TTFT.
#   - MTP n=3: winner of 2026-04-24 sweep (154 > 136 > 120 tok/s @c=10).
#   - No --rm: container persists after crash for `docker logs` post-mortem.
#   - Port 9008: drop-in for the old GGUF slot → LiteLLM/opencode unchanged.
#
# NOTE: Needs num_nextn_predict_layers=1 patch in the AlphaOxO cache config.json
# (see ~/llm-stack/README.md §MTP-patch). Without it, MTP silently no-ops.
set -e
docker rm -f llama-qwen-27b 2>/dev/null || true
docker rm -f vllm-qwen-27b 2>/dev/null || true
exec docker run --name vllm-qwen-27b \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9008:9008 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  vllm/vllm-openai:cu130-nightly \
  AlphaOxO/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 --port 9008 \
  --served-model-name qwen3.6-27b \
  --max-model-len 262144 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --language-model-only \
  --attention-backend FLASHINFER \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
