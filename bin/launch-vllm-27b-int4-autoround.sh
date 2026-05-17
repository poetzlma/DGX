#!/bin/bash
# Qwen3.6-27B Intel AutoRound INT4 — eval slot 2026-05-17.
#
# Hypothesis: bandwidth-bound 27B on Spark (per concurrency-ceiling memory
# 2026-05-11) should win decode tok/s from INT4 weight bandwidth (~14 GB)
# vs prod FP8 (~27 GB). Intel's AutoRound recipe leaves linear_attn.in_proj_a/b
# (126 layers) and mtp.fc at 16-bit, so the kaitchup 2026-05-12 evals rank it
# at FP8 parity on accuracy — unlike the NVFP4 variants we already rolled back.
#
# Quantization: auto-round (packing_format auto_round:auto_gptq, bits 4,
# group_size 128, sym). vLLM auto-detects from config.json — no --quantization
# flag needed. Loads via GPTQ-Marlin backend.
#
# Speculative: native qwen3_next_mtp head (Intel kept mtp.fc at fp16).
# Intel card recommends n=2 specifically; FP8 prod runs n=3. Start n=2, tune.
#
# Image: vllm/vllm-openai:v0.20.1-aarch64-ubuntu2404 — same vanilla used by
# FP8 prod. No AEON DFlash fork needed.
#
# Port 9017 (next free; 9008/9010-9016 all taken).
set -e
docker rm -f vllm-qwen-27b-int4-autoround 2>/dev/null || true
exec docker run --name vllm-qwen-27b-int4-autoround \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9017:9017 \
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
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  vllm/vllm-openai:v0.20.1-aarch64-ubuntu2404 \
  Intel/Qwen3.6-27B-int4-AutoRound \
  --served-model-name qwen3.6-27b-int4-autoround \
  --host 0.0.0.0 --port 9017 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --kv-cache-dtype auto \
  --max-model-len 200000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.70 \
  --attention-backend FLASH_ATTN \
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
  --chat-template /llm-stack-etc/qwen3.6-chat-template-froggeric.jinja \
  --default-chat-template-kwargs '{"preserve_thinking": true}'
