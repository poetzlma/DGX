#!/bin/bash
# Ornith-1.0-35B-MoE (DeepReinforce) — GGUF via mainline llama.cpp.
# Added 2026-06-27 (experimentation phase). Arch: qwen35moe (Qwen3.5 MoE,
# 256 experts, 40 layers, ~3B active, native ctx 262144). MIT, agentic coding.
#
# Runtime: mainline llama.cpp built with CUDA for GB10 (sm_121) at
#   ~/llama.cpp/build/bin/llama-server  (the deepseek-patched fork does NOT
#   support qwen35moe — needs current llama.cpp).
# GGUF: ~/models/ornith-1.0-35b/ornith-1.0-35b-Q4_K_M.gguf  (Q4_K_M, 21 GB)
#
# ORNITH_CTX env overrides context (default 32768). Max is 262144 but KV at
# full ctx is large — raise deliberately while benching context size.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="$HOME/models/ornith-1.0-35b/ornith-1.0-35b-Q4_K_M.gguf"
CTX="${ORNITH_CTX:-32768}"

exec "$LLAMA_SERVER" \
  --host 127.0.0.1 --port 9022 \
  --alias ornith-1.0-35b \
  -m "$MODEL" \
  --ctx-size "$CTX" \
  --n-gpu-layers 99 \
  -fa on \
  -b 2048 -ub 512 \
  --no-mmap \
  --jinja \
  --reasoning-budget -1
