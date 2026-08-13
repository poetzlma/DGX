#!/bin/bash
# DeepSeek-V4-Flash-0731 — unsloth UD-IQ3_XXS (104 GB, 3.06 bpw) on llama.cpp.
# Eval lane, port 9098. Added 2026-08-02.
#
# WHY llama.cpp AND NOT THE ds4 ENGINE: this quant physically cannot load on
# ~/ds4-q4/ds4-server. Verified against the file's own GGUF header:
#   ffn_*_exps    iq3_xxs + iq2_xs  -> ds4.c tensor_is_routed_expert_type()
#                                      allows only IQ2_XXS | Q2_K | Q4_K
#   ffn_*_shexp   q6_k              -> ds4.c:2338+ hard-pins DS4_TENSOR_Q8_0
#   ffn_gate_inp  bf16              -> hard-pinned DS4_TENSOR_F16
# See memory reference_ds4_engine_quant_compat.
#
# ============================ BINARY: USE MAINLINE ==========================
# Official deepseek4 support landed in MAINLINE llama.cpp via PR #24162 on
# 2026-06-29, plus CUDA kernels for the V4-specific ops:
#   ggml/src/ggml-cuda/dsv4-hc.cu          (hyper-connections / Sinkhorn)
#   ggml/src/ggml-cuda/lightning-indexer.cu
#
# DO NOT USE ~/llama.cpp-gx10-dsv4 (phuongncn fork, b8796). It predates official
# support by two months and implements ALL FIVE dsv4 ops CPU-ONLY — there is no
# ggml-cuda/ implementation in it at all. With -ngl 999 the scheduler therefore
# round-trips GPU->Grace->GPU at each of 43 layers, every token. Measured on
# this box 2026-08-02 with that fork, UD-IQ3_XXS at c=131072:
#       prefill  41.5 tok/s      decode  2.28 tok/s
# versus ~371-411 / ~15.5-16.6 tok/s reported for the same file on mainline
# b10216 on a single DGX Spark. ~9x and ~7x slower respectively.
#
# (For the record: 2.28 tok/s is NOT a bandwidth roofline. V4-Flash decode is
# compute-bound — an M3 Ultra gets ~16 tok/s on both a 282 GiB Q8_0 and a
# 99 GiB Q2_K. Do not reason about this model from weight size / GB-per-s.)
# ============================================================================
#
# FLAG NOTES:
#   --no-mmap   load resident; avoids unified-memory paging games on GB10.
#   -b/-ub      4096/2048 is the Spark-community setting; the default -ub 512
#               leaves prefill throughput on the table.
#   KV dtype    LEAVE UNQUANTIZED. -ctk/-ctv q8_0 silently CORRUPTS V4
#               inference (reported by multiple people; a fix landed ~2026-07-07
#               but is unverified here). f16 default is correct.
#   -fit off    skip the auto-fit device-memory probe; we set -c and -ngl
#               explicitly, and the probe was what tripped the old fork's
#               GGML_ASSERT(ubatch.n_seqs == 1).
#   -np         fork required 1; mainline handles concurrency fine. Kept at 1
#               because this lane is being benched single-stream.
#
# MEMORY: ~99 GB weights + ~1 GB KV/compute at 131072 (V4-Flash compressed
# attention makes KV nearly free: 215 MiB at 32k -> 860 MiB at 128k). Fits the
# 121.6 GB pool with ~18 GB spare. Do not run co-resident with anything.
set -e
LCTX="${LCTX:-131072}"
LPORT="${LPORT:-9098}"
LBIN="${LBIN:-/home/max/llama.cpp/build-cuda/bin/llama-server}"
MODEL=/home/max/models/ds4-unsloth-UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
# Shard 00001 is metadata-only (0 tensors) — unsloth splits with a metadata-first
# shard. llama.cpp follows split.count and pulls 00002..00004 itself.
exec "$LBIN" \
  -m "$MODEL" \
  -ngl 999 \
  -fa on \
  -fit off \
  --no-mmap \
  -b 4096 -ub 2048 \
  -np "${LNP:-1}" \
  -c "$LCTX" \
  --jinja \
  --alias deepseek-v4-flash \
  --host 127.0.0.1 --port "$LPORT"
