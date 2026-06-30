#!/bin/bash
# nvidia/Qwen3.6-27B-NVFP4 (NVIDIA official ModelOpt NVFP4) — EVAL slot 2026-06-30.
#
# Architecture: Qwen3_5ForConditionalGeneration (model_type qwen3_5) — a HYBRID
# multimodal model, 48 linear-attention (Gated DeltaNet) + 16 full-attention
# layers, vision encoder, native MTP head, 262144 native ctx. Quant is
# MIXED_PRECISION: W4A16-NVFP4 MLPs (block=16, FP8 per-block + FP32 per-tensor
# scales) + FP8 linear_attn/self_attn. Served TEXT-ONLY as a throughput lane:
# concurrency up to c=4 with a full 120k+ context window.
#
# ── WHY THIS IMAGE (the whole saga) ──────────────────────────────────────
# On a STOCK vLLM image the W4A16-NVFP4 path is hardcoded to the Marlin FP4
# kernel (modelopt.py pins MarlinNvFp4LinearKernel). Marlin's FP4 kernel is
# SM80-targeted; on GB10/sm_121 it computes wrong logits → the model emits pure
# garbage ("!!!!" / "d d d") regardless of KV dtype or eager/graph mode. This is
# the documented DGX-Spark bug (vLLM issues #37030, #43906; fix PR #38126).
# CUTLASS FP4 *is* supported on sm_121 (cutlass_fp4_supported() == True) — the
# stock image just never routes W4A16 to it.
#
# ghcr.io/aeon-7/aeon-vllm-ultimate:latest = vLLM 0.23.0 built from source for
# GB10/sm_121a WITH the CUTLASS NVFP4 kernels + the VLLM_NVFP4_GEMM_BACKEND
# override that forces W4A16 onto flashinfer-cutlass instead of Marlin. This is
# the community-validated production path for Qwen3.6-27B-NVFP4 on Spark. (Same
# AEON image family as our int4-dflash lane.)
#
# Critical knobs:
#   - VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass : route NVFP4 GEMM to CUTLASS
#     (the fix — without it you get Marlin garbage).
#   - --mamba-cache-dtype float32 : the GDN/linear-attn recurrent state cache
#     must be fp32 for numerical stability (AEON recipe).
#   - --quantization modelopt : NVIDIA ModelOpt packaging (auto modelopt_mixed).
#   - --kv-cache-dtype bfloat16 : MANDATORY for DFlash (the drafter path needs
#     bf16 KV; without it the engine defaults to fp8_e4m3 and DFlash misbehaves).
#   - --max-num-seqs 4 / --max-model-len 262144 : c=4, native 256k ctx (no RoPE
#     scaling — max_position_embeddings=262144). KV pool ~626k tok holds ~2.4x
#     concurrent 256k reqs; c=4 fine up to ~150k each. Bumped from 131072 on
#     2026-06-30 (hybrid model: only 16/64 layers carry growing KV).
#   - --gpu-memory-utilization 0.85 : solo swap-exclusive, full 119 GB budget.
#   - SPECULATIVE: DFlash n=10 via z-lab/Qwen3.6-27B-DFlash (a separate 5-layer
#     drafter reading target hidden states @ layers 1/16/31/46/61), NOT the
#     model's native MTP head. DFlash n=10 measured ~2-3x faster single-stream
#     than native MTP n=1 (which drafts only 1 tok/step) — Weschera/spark-bench
#     recipe + our A/B. AEON image is the `+dflash` build; drafter is cached.
#
# Port 9024 (next free).
set -e
docker rm -f vllm-qwen-27b-nvidia-nvfp4 2>/dev/null || true
exec docker run --name vllm-qwen-27b-nvidia-nvfp4 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9024:9024 \
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
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint vllm \
  ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve nvidia/Qwen3.6-27B-NVFP4 \
  --served-model-name qwen3.6-27b-nvfp4 \
  --host 0.0.0.0 --port 9024 \
  --quantization modelopt \
  --mamba-cache-dtype float32 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --language-model-only \
  --attention-backend flash_attn \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}'
