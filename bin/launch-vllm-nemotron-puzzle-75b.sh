#!/bin/bash
# Nemotron-3-Puzzle-75B-A9B-NVFP4 — EVAL / smoke-test lane, added 2026-07-08.
#
# nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 — 75.3B total / 9.3B active,
# hybrid Mamba + MoE + Attention (NemotronHPuzzleForCausalLM, heterogeneous blocks),
# 256k ctx, NVFP4 (Blackwell), MTP baked in (mtp.safetensors). Derived by NVIDIA's
# "Iterative Puzzle" compression of Nemotron-3-Super-120B. ~53 GB on disk, fits ONE
# GB10 (the Super-120B needed TP=2 across two Sparks — Puzzle is the single-Spark cut).
#
# This is PHASE 1: prove the arch loads and generates on sm_121. Speed tuning, MTP
# speculative decoding, reasoning/tool parsers are deferred to phase 2.
#
# sm_121 landmines this script works around (all confirmed on this NemotronH arch):
#   1. Mamba-2 Triton illegal instruction on sm_121 (vLLM #37431)
#        -> --enforce-eager + --mamba-cache-dtype float32
#   2. NVFP4 + CUDA-graph-capture illegal instruction (NVIDIA-NeMo/Nemotron #125,
#      Nano's Mamba+NVFP4 graph replay) -> --enforce-eager + --no-async-scheduling
#   3. NVFP4 MoE falls back to Marlin garbage on SM_12x (vLLM #43906)
#        -> AEON sm121a image + VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
#          (same path that makes the nvidia/Qwen3.6-27B-NVFP4 256k lane correct)
#
# Cost of #1/#2: no CUDA graphs => modest tok/s until the upstream Mamba sm121
# kernel bug lands. That's expected; the win here is the flat-KV hybrid-Mamba
# concurrency/long-context story, not raw single-stream speed. Retest without
# --enforce-eager whenever vLLM #37431 closes.
#
# --trust-remote-code is load-bearing: the repo ships custom modeling files
# (configuration_nemotron_h_puzzle / modeling_nemotron_h_puzzle). Whether vLLM
# accepts the *Puzzle* heterogeneous variant is the actual go/no-go of phase 1.
#
# Port 9027 (next free after the 35B vision lane on 9026). Not wired into
# llama-swap/LiteLLM yet — launch by hand, watch `docker logs -f`.
set -e
docker rm -f vllm-nemotron-puzzle-75b 2>/dev/null || true
exec docker run --name vllm-nemotron-puzzle-75b \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9027:9027 \
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
  serve nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 \
  --served-model-name nemotron-3-puzzle-75b \
  --host 0.0.0.0 --port 9027 \
  --quantization modelopt \
  --trust-remote-code \
  --enforce-eager \
  --no-async-scheduling \
  --mamba-cache-dtype float32 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --generation-config vllm
  # --- phase 2 (add once phase 1 boots clean): ---
  # --speculative-config '{"method":"mtp","num_speculative_tokens":1}'  # mtp.safetensors; may need CUDA graphs (drop --enforce-eager)
  # --reasoning-parser nemotron  --enable-auto-tool-choice --tool-call-parser <nemotron>   # buffers TTFT; confirm parser names
