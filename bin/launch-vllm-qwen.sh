#!/bin/bash
# Launcher for the Qwen3.6-35B-A3B MoE container.
# NO --rm: container persists after crash so `docker logs vllm-qwen` gives a
# post-mortem. The docker rm below cleans any prior dead/orphaned container
# before we respawn so --name doesn't collide.
set -e
docker rm -f vllm-qwen 2>/dev/null || true
exec docker run --name vllm-qwen \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9001:9001 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  vllm/vllm-openai:cu130-nightly \
  RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --tokenizer Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 --port 9001 \
  --served-model-name qwen3.6-35b-a3b \
  --max-model-len 131072 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.55 \
  --moe-backend flashinfer_cutlass \
  --language-model-only \
  --reasoning-parser qwen3 \
  --attention-backend FLASHINFER \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
