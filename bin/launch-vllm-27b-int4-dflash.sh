#!/bin/bash
# Qwen3.6-27B Intel INT4 (AutoRound) + z-lab DFlash drafter — eval slot 2026-05-17.
#
# Pivot from vanilla Intel INT4 + MTP n=2 (capped at 22 tok/s single-stream)
# to the AEON-7 DFlash-patched vLLM image so we can drive n=15 speculative
# decoding off the z-lab Qwen3.6-27B-DFlash drafter. AEON's own measurements
# show DFlash + NVFP4 = 37.6 tok/s (3.58× over their 10.5 tok/s raw baseline)
# on Spark; our INT4 raw baseline measured 13.4 tok/s, so the same 3.5× speedup
# would put us in the 45-50 tok/s zone.
#
# Why this combo, not stock vLLM:
#   - DFlash needs the interleaved-sliding-window-attention patch (vLLM PR
#     #40898). Vanilla v0.20.1 doesn't have it; AEON v3 image bundles it.
#   - DFlash drafter (z-lab) is a separate small model that shares E/LM-head
#     with the base. Embeddings/LM-head are unchanged across INT4 vs NVFP4 quant
#     of the same base, so the drafter should attach to Intel INT4 cleanly.
#
# Open risks:
#   - GPTQ-Marlin INT4 backend may not coexist cleanly with DFlash drafter
#     loader in this image (image was built for compressed-tensors NVFP4). If
#     it fails to start, fall back to Intel INT4 + MTP n=2 at 22 tok/s.
#   - DFlash drafter style was distilled against the Qwen3.6-27B base; INT4 of
#     the same base should produce ~the same logits, so acceptance should hold.
#     Real-traffic A/B is the only way to confirm.
#
# Port 9018 (next free; 9017 is the vanilla INT4+MTP entry, kept as rollback).
set -e
docker rm -f vllm-qwen-27b-int4-dflash 2>/dev/null || true
exec docker run --name vllm-qwen-27b-int4-dflash \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9018:9018 \
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
  ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4 \
  bash -c '
    exec vllm serve Intel/Qwen3.6-27B-int4-AutoRound \
      --chat-template /llm-stack-etc/qwen3.6-chat-template-froggeric.jinja \
      --served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-int4-dflash \
      --host 0.0.0.0 --port 9018 \
      --tensor-parallel-size 1 \
      --dtype auto \
      --kv-cache-dtype auto \
      --max-model-len 120000 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.50 \
      --enable-chunked-prefill \
      --enable-prefix-caching \
      --load-format safetensors \
      --trust-remote-code \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --attention-backend flash_attn \
      --language-model-only \
      --generation-config vllm \
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs "{\"preserve_thinking\": true}" \
      --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\"}" \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":4}"
  '
