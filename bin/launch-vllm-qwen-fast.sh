#!/bin/bash
# Qwen3.6-35B-A3B-NVFP4-Fast (unsloth) — vLLM lane replacing the llama.cpp GGUF lane.
# WHY: NVFP4 + native MTP (n=2 per model card) => ~1.79x throughput claim, WITH
# vision and parallel — the combo llama.cpp MTP cannot do (PR #22673 limits).
# Sized to co-reside beside nemotron-75B (util 0.55): QWENFAST_UTIL=0.28 ≈ 34 GB.
# Rollback: llama-swap.yaml qwen entry -> launch-llamacpp-35b-moe-vision.sh (Q4 GGUF).
docker rm -f vllm-qwen-fast 2>/dev/null || true
exec docker run --name vllm-qwen-fast \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9026:9026 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NVIDIA_FORWARD_COMPAT=1 -e NVIDIA_DISABLE_REQUIRE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --served-model-name qwen3.6-35b-a3b-vision \
  --host 0.0.0.0 --port 9026 \
  --trust-remote-code \
  --gpu-memory-utilization ${QWENFAST_UTIL:-0.28} \
  --max-model-len ${QWENFAST_CTX:-262144} \
  --max-num-seqs ${QWENFAST_SEQS:-2} \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":2}"
