#!/bin/bash
# poolside/Laguna-S-2.1-NVFP4 — EVAL LANE (added 2026-07-22).
#
# 117.6B total / 8.5B active MoE (256 experts + 1 shared, 10 exp/tok), 48 layers
# (36 sliding-window sw=512 + 12 global), native 262k ctx (no RoPE scaling),
# compressed-tensors nvfp4-pack-quantized (~71 GB on disk), FP8 KV cache.
#
# Poolside targets a single DGX Spark and their own DFlash speculative decoder
# (poolside/Laguna-S-2.1-DFlash-NVFP4, 6-layer BF16 drafter). Upstream vLLM
# v0.25.1 has BOTH native LagunaForCausalLM and the dflash method (PR #46853,
# merged 2026-07-03, verified contained in the v0.25.1 tag) — the AEON sm121a
# back-port image is no longer needed for this lane.
#
# STANDALONE on port 9030 — deliberately NOT in llama-swap config so the swap
# manager cannot evict this container mid-benchmark when traffic hits :8080.
#
# Env knobs:
#   LAG_SPEC=0   -> detach DFlash drafter (default 1 = drafter on; prod config)
#   LAG_CTX      -> --max-model-len (default 262144)
#   LAG_UTIL     -> --gpu-memory-utilization (default 0.85)
#   LAG_SEQS     -> --max-num-seqs (default 8; need >=4 for the concurrency bench)
#   LAG_IMAGE    -> container image (default upstream v0.25.1 aarch64)
#   LAG_MOE      -> --moe-backend override. LEAVE UNSET on GB10: A/B'd 2026-07-22,
#                   flashinfer_cutlass (auto) is the ONLY NVFP4 MoE backend that
#                   starts on sm_121 — triton is "not supported for NvFP4 MoE",
#                   flashinfer_trtllm/cutedsl have no sm_121 kernels (all three
#                   hard-error at startup -> llama-swap crash-loops prod). Knob
#                   kept for future images/backends only.
#   LAG_SPEC_N   -> num_speculative_tokens when LAG_SPEC=1 (default 15; measured
#                   2026-07-22: n=15 beats n=7 at 100k ctx (33.3 vs 25.9 tok/s)
#                   with no TTFT penalty vs baseline)
#
# GENERATION CONFIG (2026-07-22): model's generation_config.json now APPLIES
# (keeps poolside's eval-certified top_k=20 — the old `--generation-config vllm`
# flag discarded it). The max_new_tokens override sets the DEFAULT output budget
# when a client omits max_tokens: THINKING NEEDS HEADROOM or it eats the whole
# budget and content comes back null. Explicit client max_tokens still wins.
# Upstream ships no max_new_tokens cap (verified both poolside repos) — the
# override is insurance against one appearing in a future snapshot.
set -e

SPEC_ARGS=()
if [ "${LAG_SPEC:-1}" = "1" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"poolside/Laguna-S-2.1-DFlash-NVFP4\",\"num_speculative_tokens\":${LAG_SPEC_N:-15}}")
fi

docker rm -f vllm-laguna-s21 2>/dev/null || true
exec docker run --name vllm-laguna-s21 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9030:9030 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/.cache/vllm-laguna:/root/.cache/vllm \
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
  "${LAG_IMAGE:-vllm/vllm-openai:v0.25.1-aarch64-ubuntu2404}" \
  serve poolside/Laguna-S-2.1-NVFP4 \
  --served-model-name laguna-s-2.1 \
  --host 0.0.0.0 --port 9030 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  ${LAG_MOE:+--moe-backend ${LAG_MOE}} \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization ${LAG_UTIL:-0.85} \
  --max-model-len ${LAG_CTX:-262144} \
  --max-num-seqs ${LAG_SEQS:-4} \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tool-call-parser poolside_v1 \
  --enable-auto-tool-choice \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --override-generation-config '{"max_new_tokens": 131072}' \
  "${SPEC_ARGS[@]}"
