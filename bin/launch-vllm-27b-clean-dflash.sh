#!/bin/bash
# Qwen3.6-27B clean (AlphaOxO) NVFP4 + native Qwen3.6-DFlash drafter (k=15).
#
# Goal: combine clean (non-abliterated) AlphaOxO weights — which we expect to
# resolve the coding-quality complaints attributed to AEON-7 — with the DFlash
# drafter that drives prod's c=1 41 tok/s figure. If acceptance rate holds when
# the drafter (distilled against the base) is paired with AlphaOxO, this is a
# strict upgrade over the AEON-7 prod path.
#
# Image: ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 — same as prod.
#        It bundles vLLM with DFlash PR #40898 patched in. Mainline
#        vllm/vllm-openai:cu130-nightly may or may not carry that patch yet,
#        so reusing AEON-7's image removes the engine-side risk and keeps the
#        only variable the target weights.
#
# Differences vs prod (launch-vllm-27b-dflash.sh):
#   - Target model:   AlphaOxO/Qwen3.6-27B-NVFP4 (clean instruct, was the
#                     dormant MTP target on port 9008)
#   - Drafter:        z-lab/Qwen3.6-27B-DFlash  (unchanged, k=15)
#   - Port:           9014  (does not collide with 9013 prod or 9008 mtp)
#   - Container:      vllm-qwen-27b-clean-dflash
#   - --served-model-name:  qwen3.6-27b-clean-dflash
#
# Same as prod (per AEON-7 README + z-lab DFlash card + prod tuning):
#   - NO --reasoning-parser qwen3   (buffered <think> adds ~3 s TTFT — memory)
#   - --language-model-only         (text-only workload, frees mm encoder mem)
#   - --quantization compressed-tensors  (AlphaOxO ships in this format too)
#   - --attention-backend flash_attn     (z-lab card requirement for DFlash)
#   - --max-num-batched-tokens 32768     (z-lab card recommendation)
#   - Spark/Blackwell env vars (TORCH_CUDA_ARCH_LIST=12.1a, NVFP4 SM100 off)
#
# Cold start: ~10–12 min on first boot per the prod image notes (FlashInfer
# NVFP4 GEMM autotuner + CUDA-graph capture, both cache to /root/.cache/vllm).
set -e
docker rm -f vllm-qwen-27b-clean-dflash 2>/dev/null || true
exec docker run --name vllm-qwen-27b-clean-dflash \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9014:9014 \
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
  ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 \
  bash -c '
    exec vllm serve AlphaOxO/Qwen3.6-27B-NVFP4 \
      --chat-template /llm-stack-etc/qwen3.6-chat-template-froggeric.jinja \
      --served-model-name qwen3.6-27b-clean-dflash \
      --host 0.0.0.0 --port 9014 \
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
