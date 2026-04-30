#!/bin/bash
# Bench launcher — Qwen3.6-27B AEON-7 NVFP4 + native Qwen3.6-DFlash drafter (k=15).
#
# Side-by-side bench vs prod NVFP4+MTP via llama-swap (main, swap-exclusive group).
# Mirrors AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash docker-compose flags
# verbatim with three deliberate deltas vs their compose:
#   1. NO --reasoning-parser qwen3   → matches prod; per memory adds ~3 s TTFT for buffered <think>
#   2. --language-model-only         → matches prod (text-only workload, frees mm encoder mem)
#   3. Port 9013, container vllm-qwen-27b-dflash, --served-model-name qwen3.6-27b-dflash
#      → does not collide with prod 9008/vllm-qwen-27b
#
# Cold start: ~10–12 min first boot per AEON-7 README (FlashInfer NVFP4 GEMM autotuner
# + CUDA-graph capture, both cache to /root/.cache/vllm/...).
set -e
docker rm -f vllm-qwen-27b-dflash 2>/dev/null || true
exec docker run --name vllm-qwen-27b-dflash \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9013:9013 \
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
    exec vllm serve AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4 \
      --served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-dflash \
      --host 0.0.0.0 --port 9013 \
      --tensor-parallel-size 1 \
      --dtype auto \
      --quantization compressed-tensors \
      --kv-cache-dtype auto \
      --max-model-len 200000 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format safetensors \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --attention-backend flash_attn \
      --language-model-only \
      --generation-config vllm \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":15}"
  '
