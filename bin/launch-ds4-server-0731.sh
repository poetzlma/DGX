#!/bin/bash
# ds4 lane — DeepSeek-V4-Flash-0731 (official 2026-07-31 release) TEST twin of
# launch-ds4-server.sh. Added 2026-08-01.
#
# WHY:  0731 supersedes the V4-Flash preview our lane has served since 05-12.
#       antirez published the matching ds4-layout GGUF on 2026-08-01 00:57 with
#       the IDENTICAL quant recipe and byte size as our preview file
#       (IQ2XXS gate/up, Q2_K down, AProj/SharedExp/Out Q8, chat-v2 imatrix,
#       86,720,111,488 B = 80.76 GiB), so this is a pure weights swap — same
#       binary, same flags, same tensor layout.
#
# HEADER-DIFF FINDINGS 2026-08-01 (bin/gguf-head.py, run against the partial
# download vs the live preview file's header pulled off codeserver). The GGUF
# metadata blocks are IDENTICAL except for imatrix provenance:
#     tensors            1328  ==  1328
#     metadata keys        62  ==  62
#     general.architecture      deepseek4  (both)  <- what this fork dispatches on
#     general.file_type         19         (both)
#     tokenizer.*               identical (gpt2/joyai-llm pre, 129280 tokens,
#                               bos 0 / eos 1)
#     tokenizer.chat_template   PRESENT in both, byte-identical, 4988 chars
#   Only deltas:
#     quantize.imatrix.file     ...-0731-chat-v2-routed-moe-ds4.dat  (NEW, 0731-specific)
#     quantize.imatrix.chunks_count  202100  vs  90042 on the preview
#
# So two worries are retired before the transfer even finished:
#   - No chat-template risk. 0731 upstream ships no Jinja template (a Python
#     encoding/ folder instead), but antirez's conversion carries the same
#     chat-v2 template the preview file has, byte for byte.
#   - The imatrix is NOT the preview one. The HF card text claiming a preview
#     imatrix describes the *schlaflos* re-upload, not antirez's; antirez
#     recalibrated on 0731 with 2.2x more chunks. Cross-release imatrix drift
#     is therefore a non-issue here.
#
# REMAINING UNKNOWN — the gating question this script exists to answer:
#   Does our PINNED ngc-shj q4 fork (~/ds4-q4, be43477 + 3 cherry-picks, ~95
#   commits behind antirez/main) actually LOAD it? Identical tensor count and
#   metadata schema make this very likely, but xangel82's "no loader changes
#   needed" report is about THEIR fork, not this binary. Metadata parity is
#   not proof: tensor NAMES and per-tensor quant types are not in the KV block.
#
# Everything below is copied verbatim from launch-ds4-server.sh except --model
# and --port. Do NOT "improve" flags here — one variable at a time, or the
# comparison against the preview lane's recorded 20.8 t/s decode / ~396 t/s
# prefill is worthless.
set -e

# See launch-ds4-server.sh for the full rationale on these two exports.
# cudaMemGetInfo's "free" on GB10 reads ~5 GiB even with 30 GiB+ headroom, so
# the default 5% reserve latches the q8 slow path on the first request.
export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=1024
# Opt-in Q4 lazy cache + dp4a decode matmul. Requires NO --mtp (every Q4
# dispatch gates on n_tok == 1).
export DS4_CUDA_Q4_DECODE=1

MODEL="${DS4_MODEL:-/home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-0731.gguf}"
PORT="${DS4_PORT:-9099}"
CTX="${DS4_CTX:-131072}"
# DS4_BIN: binary override for A/B runs. 2026-08-01 finding: the prod binary at
# ~/ds4-q4/ds4-server contains ONLY sm_75 cubins (Makefile `cuda-spark` target
# passes empty CUDA_ARCH; nvcc's default on this box is sm_75) — GB10 runs it
# via driver-JIT of Turing PTX. ~/ds4-q4-sm121/ds4-server is the same source
# rebuilt with `make cuda CUDA_ARCH=sm_121` (native Blackwell cubins).
BIN="${DS4_BIN:-/home/max/ds4-q4/ds4-server}"

exec "$BIN" \
  --cuda \
  --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL" \
  --ctx "$CTX" \
  --kv-disk-dir /home/max/ds4/kv-cache-0731 \
  --kv-disk-space-mb 32768 \
  --warm-weights
