#!/bin/bash
# Qwen3.6-35B-A3B-NVFP4 MoE — re-eval slot 2026-05-17.
#
# Per Steve Scargall's 2026-04 DGX Spark recipe, RedHatAI/Qwen3.6-35B-A3B-NVFP4
# achieves 55.9 tok/s single-user (MTP n=1, 85% accept) and 433 tok/s at c=32
# on Spark with the vanilla cu130-nightly image. This is faster than our INT4
# dense path (27 tok/s c=1) — MoE activates only ~3B params per token, so the
# bandwidth wall that caps the dense 27B doesn't apply.
#
# Memory `project_qwen36_35b_moe_rejected` (2026-05-11) noted the MoE loses on
# coding accuracy vs 27B dense (≥4 pts SWE-bench / -15.5 SkillsBench). We are
# re-running here at the user's request to validate Scargall's published
# numbers and explore throughput at c=2/4 with 125k context. Quality A/B vs
# the dense INT4+DFlash entry is the separate question.
#
# Critical flags from Scargall's recipe:
#   - --quantization compressed-tensors (matches Red Hat NVFP4 packaging)
#   - --moe-backend flashinfer_cutlass (sm121a verified MoE routing)
#   - --gpu-memory-utilization 0.87 (much higher than old 0.55 default)
#   - --max-model-len 131072 (Qwen3.6 native 128K, covers user's 125k need)
#   - --kv-cache-dtype fp8_e4m3 (halves KV vs bf16; needed for c=4 at 125k)
#   - --speculative-config mtp n=1 (Qwen3.6 MTP is 1-layer; n>1 just chains it)
#
# Image: vllm/vllm-openai:cu130-nightly (already in cache, Scargall confirms
# this is the working image for Blackwell NVFP4 + MoE routing).
#
# Port 9019 (next free; 9018 = int4-dflash prod, 9017 = int4-mtp eval, 9016 =
# fp8 rollback, 9015 = sakamaki rollback, 9014 = clean-dflash, 9013 = aeon,
# 9012 = nemotron, 9011 = ds4-llama, 9010 = ds4-antirez, 9008 = mtp-dormant).
set -e
docker rm -f vllm-qwen-35b-moe-nvfp4 2>/dev/null || true
exec docker run --name vllm-qwen-35b-moe-nvfp4 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9019:9019 \
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
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  vllm/vllm-openai:cu130-nightly \
  RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b-a3b-nvfp4 \
  --host 0.0.0.0 --port 9019 \
  --quantization compressed-tensors \
  --moe-backend flashinfer_cutlass \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.40 \
  --max-model-len 80000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
