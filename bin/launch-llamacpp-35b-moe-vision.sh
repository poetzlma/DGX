#!/bin/bash
# Qwen3.6-35B-A3B (unsloth GGUF) + mmproj vision encoder — mainline llama.cpp.
# Added 2026-07-07 (experimentation phase). VISION lane for the 35B MoE:
# the vLLM NVFP4 lane (qwen3.6-35b-a3b-nvfp4, :9019) is text-only; this lane
# loads unsloth's mmproj-F16 so image_url content blocks work.
#
# Runtime: mainline llama.cpp CUDA build for GB10 (sm_121) at
#   ~/llama.cpp/build/bin/llama-server (same binary as the ornith lane).
# GGUF: ~/models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf (22.4 GB,
#   unsloth dynamic 4-bit) + mmproj-F16.gguf (vision encoder).
#
# QWEN35B_CTX = TOTAL engine ctx shared across QWEN35B_PARALLEL slots.
# Default 262144 = 2 slots x 131072 → c=2 @ 131k per request.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="$HOME/models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MMPROJ="$HOME/models/qwen3.6-35b-a3b-gguf/mmproj-F16.gguf"
CTX="${QWEN35B_CTX:-524288}"
PARALLEL="${QWEN35B_PARALLEL:-2}"

exec "$LLAMA_SERVER" \
  --host 127.0.0.1 --port 9026 \
  --alias qwen3.6-35b-a3b-vision \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  --n-gpu-layers 99 \
  -fa on \
  -b 2048 -ub 512 \
  --no-mmap \
  --jinja \
  --reasoning-budget -1
