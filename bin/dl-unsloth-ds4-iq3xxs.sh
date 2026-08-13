#!/bin/bash
# Download unsloth/DeepSeek-V4-Flash-0731-GGUF UD-IQ3_XXS (104 GB, 4 shards).
# NOTE: this quant is for llama.cpp ONLY — it cannot load on the antirez ds4
# engine (bf16 router, q6_k shexp, iq3_xxs/iq2_xs experts are all rejected by
# ds4.c's hard-pinned layout checks). Serve via ~/llama.cpp-gx10-dsv4.
# Resumable: re-run to continue partial shards.
set -u
DEST=/home/max/models/ds4-unsloth-UD-IQ3_XXS
BASE=https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ3_XXS
mkdir -p "$DEST"
pids=()
for i in 1 2 3 4; do
  f=$(printf "DeepSeek-V4-Flash-0731-UD-IQ3_XXS-%05d-of-00004.gguf" "$i")
  ( curl -sL -C - --retry 10 --retry-delay 5 --retry-all-errors \
        -o "$DEST/$f" "$BASE/$f" && echo "DONE $f" || echo "FAIL $f" ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "=== all shards finished ==="
ls -la "$DEST"
du -sh "$DEST"
