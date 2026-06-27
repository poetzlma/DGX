#!/bin/bash
# Qwopus3.6-27B-v2 (Jackrong, Claude-Opus-trace-distilled fine-tune of
# Qwen3.6-27B) — INT4 AutoRound + z-lab DFlash drafter n=4. EVAL slot.
#
# Cloned from launch-vllm-27b-int4-dflash.sh (prod) so the comparison is
# apples-to-apples. Differences vs prod:
#   - model: locally AutoRound-quantized Qwopus (Intel recipe: int4 g128 sym,
#     all linear_attn.in_proj_a/b + mtp.fc kept fp16).
#   - --max-num-seqs 8  (prod=1) so a c1..c8 concurrency sweep isn't capped.
#   - --max-model-len 128000 so the 120k-context bench cell fits.
#   - port 9020.
#
# DFlash drafter z-lab/Qwen3.6-27B-DFlash shares E/LM-head with the *base*
# Qwen3.6-27B. Qwopus is a fine-tune; if it shifted those weights, acceptance
# (hence tok/s) drops. The benchmark will expose that.
set -e
MODEL=/models/Qwopus3.6-27B-v2-int4-AutoRound/66a4ee9d49dbcef3f83528a400ef6bec93684d6b-w4g128
docker rm -f vllm-qwopus-int4-dflash 2>/dev/null || true
exec docker run --name vllm-qwopus-int4-dflash \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9020:9020 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/llm-stack/models:/models:ro \
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
    exec vllm serve '"$MODEL"' \
      --chat-template /llm-stack-etc/qwen3.6-chat-template-froggeric.jinja \
      --served-model-name qwopus3.6-27b qwopus3.6-27b-int4-dflash \
      --host 0.0.0.0 --port 9020 \
      --tensor-parallel-size 1 \
      --dtype auto \
      --kv-cache-dtype auto \
      --max-model-len 128000 \
      --max-num-seqs 8 \
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
      --reasoning-parser qwen3 \
      --default-chat-template-kwargs "{\"preserve_thinking\": true}" \
      --compilation-config "{\"cudagraph_mode\":\"PIECEWISE\"}" \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":4}"
  '
