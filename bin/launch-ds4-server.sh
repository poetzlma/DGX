#!/bin/bash
# DeepSeek-V4-Flash via antirez/ds4 (custom C/CUDA engine) — added 2026-05-12.
#
# WHY:    Alternate V4-Flash lane to the parked llama.cpp-fork slot. ds4 is
#         a from-scratch C/CUDA implementation by antirez with three things
#         the fork lacks:
#           1. Compressed-KV that scales to 256k+ ctx on the 119 GB Spark
#              (long-ctx CUDA fixes landed 2026-05-11; q8 fp16 cache memory
#              guard landed 2026-05-12 — both verified in our local build).
#           2. Persistent disk KV cache (cold/continued/evict/shutdown
#              triggers) — survives swap-out and restart. Matches the
#              ~100k:4k prefix-heavy coding/planning traffic shape.
#           3. Native speculative decoding via the dedicated MTP GGUF.
#
# ROLE:   Planner-only lane. Decode is ~13 t/s on Spark q2-imatrix (verified
#         via ds4-bench at 6k-8k ctx; matches antirez README's 13.75 t/s
#         Spark entry). NOT a default coding model — qwen3.6-27b-fp8 stays
#         as the coding workhorse. ds4 wins on long-ctx capability and the
#         disk KV cache, not raw throughput.
#
# NOTES:  - Build: ~/ds4/Makefile, CUDA_ARCH=sm_120 (GB10 reports sm_121
#           but sm_120 is forward-compatible and what antirez documents).
#         - Native binary, no docker — needs Spark host CUDA 13 toolchain.
#         - --warm-weights touches the 80.76 GiB tensor pages at startup
#           (~74s warm / ~104s cold) to avoid first-use stalls.
#         - --ctx 131072 chosen over 256k: at 256k ctx alloc the q8 fp16
#           cache budget guard fires and attention falls back to q8 kernels
#           (acceptable, but worth re-validating end-to-end before pushing).
#         - Port 9010 (avoids 9008 mtp / 9011 dsv4-llama.cpp / 9012 nemotron
#           / 9013 dflash / 9014 clean / 9015 sakamaki / 9016 qwen-fp8).
#         - Disk KV at /home/max/ds4/kv-cache, 32 GB budget. Persists
#           across restarts and llama-swap evictions.
set -e
# Q8->FP16 weight cache: override the default 5%-of-total reserve (6.08 GiB on
# Spark's 121 GiB unified pool). cudaMemGetInfo's "free" on GB10 unified memory
# reads ~5 GiB even when the unified pool has 30 GiB+ headroom, so the default
# reserve trips on the very first request and latches q8 fallback for the
# process lifetime, slowing prefill from ~340 t/s to ~95 t/s on real prompts.
# 1 GiB is plenty (model ~81 GB + ctx ~5 GB = ~86 GB used of 119 GB).
export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=1024
# 2026-05-18: switched to ngc-shj fork's q4-only build (cherry-picked onto
# antirez be43477 + local metrics patch). Opt-in Q4 lazy cache + dp4a decode
# matmul takes single-stream decode 13.22 -> 18.77 t/s on ds4-bench, and the
# bundled host-register-fallback takes prefill at 2048 ctx 96 -> 396 t/s
# (independent of Q4). See ~/.claude/projects/-home-max/memory/project_ds4_ngcshj_fork.md
# Rollback: launch-ds4-server.sh.bak.20260518-pre-q4 (binary at /home/max/ds4/ds4-server).
#
# 2026-05-18 follow-up: MTP disabled. Every Q4 dispatch in ds4_cuda.cu is
# gated on `n_tok == 1`; with --mtp-draft 2 the main forward processes 3
# tokens in parallel (1 verify + 2 draft predictions) so Q4 is silently
# bypassed. Five live curl queries with MTP+Q4 enabled steadied at ~13.8 t/s
# (~= old prod MTP+Q8). Dropping MTP frees Q4 and gets ~18 t/s decode.
export DS4_CUDA_Q4_DECODE=1
# 2026-06-27: ROLLED BACK from the rebase binary (~/ds4-rebase, commit 65f9552).
# The rebase gained ~3 t/s decode but lost the ngc-shj host-register prefill
# fallback and ignores DS4_CUDA_Q8_F16_CACHE_RESERVE_MB, latching the q8 slow
# path: ~97 t/s prefill vs this fork's ~371 t/s (verified live: 10.9k-tok prompt
# 111.6s -> 29.3s). On 80k-tok prompts that was ~14-min TTFT -> all reqs aborted.
# Do NOT repoint to ds4-rebase until that binary honors the Q8 reserve override.
exec /home/max/ds4-q4/ds4-server \
  --cuda \
  --host 127.0.0.1 --port 9010 \
  --model /home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --ctx 131072 \
  --kv-disk-dir /home/max/ds4/kv-cache \
  --kv-disk-space-mb 32768 \
  --warm-weights
