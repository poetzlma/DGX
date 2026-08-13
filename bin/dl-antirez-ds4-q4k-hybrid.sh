#!/bin/bash
# Download antirez's 0731 Q4K-hybrid GGUF (97.6 GB) — the higher-bpw upgrade
# candidate for the ds4 lane.
#
# WHY THIS ONE: its recipe is exactly inside what ~/ds4-q4/ds4.c accepts —
#   routed experts  Q4_K (layers 37-42) / IQ2_XXS gate+up / Q2_K down
#                   -> tensor_is_routed_expert_type() allows IQ2_XXS|Q2_K|Q4_K
#   shexp + attn + out  Q8_0   -> hard-pinned DS4_TENSOR_Q8_0 at ds4.c:2338+
#   ffn_gate_inp        F16    -> hard-pinned DS4_TENSOR_F16
# Header-verified 2026-08-02 by range-fetching the first 12 MB.
# vs current prod IQ2XXS-0731 (86.7 GB): +11 GB, top-6 expert layers at Q4_K.
#
# (For contrast: NO unsloth UD-* quant can load here — bf16 router, q6_k
# shexp, iq3_xxs/iq2_xs experts are all rejected. Those are llama.cpp-only.)
set -u
DEST=/home/max/ds4/gguf
F="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
BASE=https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main
mkdir -p "$DEST"
curl -sL -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$DEST/$F" "$BASE/$F" \
  && echo "DONE $F" || echo "FAIL $F"
echo "=== antirez q4k-hybrid finished ==="
ls -la "$DEST/$F"
