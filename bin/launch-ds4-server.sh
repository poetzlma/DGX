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
exec /home/max/ds4/ds4-server \
  --cuda \
  --host 127.0.0.1 --port 9010 \
  --model /home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp /home/max/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --ctx 131072 \
  --kv-disk-dir /home/max/ds4/kv-cache \
  --kv-disk-space-mb 32768 \
  --warm-weights
