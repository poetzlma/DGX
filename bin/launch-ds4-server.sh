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
# 2026-08-06: NOT SET ANY MORE. Mainline special-cases boxes with >= 112 GiB
# total (ds4_cuda.cu, cuda_q8_f16_cache_reserve_bytes) and defaults to a 512 MiB
# reserve, so the old 5%-of-total default that forced this override is gone.
# Verified: stock defaults reproduce upstream's published GB10 prefill numbers.
# export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=1024
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
# 2026-08-06: NOT SET ANY MORE — this env var does not exist upstream. It was
# ngc-shj's fork-only Q4 dp4a decode path, and dropping it is the whole cost of
# this cutover: decode 17.38 -> 14.18 t/s on a 34.6k coding context. Bought in
# exchange for prefill 368 -> 857 t/s (2.33x). See the header notes.
# export DS4_CUDA_Q4_DECODE=1
# 2026-06-27: ROLLED BACK from the rebase binary (~/ds4-rebase, commit 65f9552).
# The rebase gained ~3 t/s decode but lost the ngc-shj host-register prefill
# fallback and ignores DS4_CUDA_Q8_F16_CACHE_RESERVE_MB, latching the q8 slow
# path: ~97 t/s prefill vs this fork's ~371 t/s (verified live: 10.9k-tok prompt
# 111.6s -> 29.3s). On 80k-tok prompts that was ~14-min TTFT -> all reqs aborted.
# Do NOT repoint to ds4-rebase until that binary honors the Q8 reserve override.
# 2026-08-01: PROMOTED to the official DeepSeek-V4-Flash-0731 weights, and this
# lane became the DEFAULT coding model (replacing laguna-s-2.1, user's call).
# 0731 GGUF verified the same day: sha256 ca22ae2f..., loads on this exact
# binary unmodified, smoke 9/10 (code-exec 7/7, arithmetic exact, needle @30k),
# and benched at parity with the preview weights (404 t/s prefill @2k, 19-20
# t/s decode). Header diff vs preview: identical but for imatrix provenance —
# antirez recalibrated on 0731 (202,100 chunks vs 90,042).
#   Rollback to preview weights: launch-ds4-server.sh.bak.20260801-preview
#   — but NOTE (checked 2026-08-13) the preview GGUF is NO LONGER ON DISK; it
#   was evicted during the 2026-08-02 disk-reclaim pass, so that rollback now
#   costs an ~87 GB re-download, not a script swap.
# NOTE the binary stays ~/ds4-q4/ds4-server. It contains only sm_75 cubins and
# runs via driver-JIT; the sm_121 rebuild at ~/ds4-q4-sm121 was MEASURED WORSE
# (-3 to -6% prefill at 8k+, -2 to -4% decode). Do not "fix" the arch.
# =====================================================================
# 2026-08-06 CUTOVER: ngc-shj fork  ->  antirez/ds4 MAINLINE b030961.
#
# WHY: mainline's aligned-artifact repack + vendored mmq prefill tier removes
# the i-quant dequant wall we sat behind since May. Measured on THIS box, same
# IQ2XXS-0731 GGUF, same prompts (ds4-bench + ds4 CLI, --temp 0):
#     prefill @34.6k code ctx   368 -> 857 t/s   (2.33x)
#     prefill flat ~850 t/s from 2k to 32k; the fork DECAYED 401 -> 349
#     decode  @34.6k code ctx  17.38 -> 14.18 t/s  (-18%, the cost)
# Net win because this lane is prefill-bound: mainline wins whenever
#     prompt/output > ~8.4      (our traffic is ~100k:4k = 25:1)
#
# ALSO GAINED: the 2026-08-02 host-registration hard-hang is structurally gone
# (mainline leaves the model mmap unpinned, so there is no 80 GiB registration
# to be declined and no device-copy fallback). ALLOW_OOM_RISK no longer applies.
#
# REJECTED: DSpark speculative decode. Measured SLOWER, 23-24%, at both context
# lengths on coding prompts (13.84 / 10.93 t/s). Greedy-only and +6 GB besides.
# Support GGUF is kept at ds4-mainline-0805/gguf/ if upstream improves it.
#
# LOCAL DELTA: only the Prometheus /metrics endpoint (187 additive lines,
# ~/ds4-metrics-endpoint-b030961.patch), now under the `ds4:` namespace instead
# of `vllm:`. stack/engine.py parse_metrics() aliases ds4: -> vllm: so all the
# existing tooling keeps working. Re-apply that patch after any upstream pull.
#
# ROLLBACK (one line): point this exec back at /home/max/ds4-q4/ds4-server and
# re-enable the two exports above. That binary and its GGUF are untouched on
# disk. Full pre-cutover copy: launch-ds4-server.sh.bak.20260806-pre-mainline
#
# BEHAVIOR CHANGE VERIFIED LIVE 2026-08-06: mainline returns thinking in a
# separate `reasoning_content` field (fork emitted it inline in `content`).
# log-proxy already splits reasoning_content, so it is fine. BUT: on a
# small max_tokens the whole budget can go to thinking and `content` comes back
# EMPTY (measured: max_tokens=300 -> content 0 chars, finish=length;
# max_tokens=2000 -> finish=stop, 1422 chars of real code). Clients that read
# only `content` will see nothing on tight budgets. Our traffic is ~100k:4k so
# the budget is normally generous, but watch for empty-content complaints —
# the fix is a bigger client max_tokens, same as the Qwen3.6 thinking issue.
#
# STILL NOT PROVEN UNDER REAL TRAFFIC: disk-KV behavior at 131k ctx, and
# opencode tool-call parsing. Watch those first.
# =====================================================================
exec /home/max/ds4-mainline-0805/ds4-server \
  --cuda \
  --host 127.0.0.1 --port 9010 \
  --model /home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-0731.gguf \
  --ctx 131072 \
  --kv-disk-dir /home/max/ds4/kv-cache-0731-mainline \
  --kv-disk-space-mb 32768
# NOTE --warm-weights is GONE: mainline does not accept it (the lane will refuse
# to start if you re-add it). It is obsolete — mainline eagerly builds ~78.7 GiB
# of aligned CUDA artifacts at load (~22s), which prepares the weights anyway.
#
# NOTE separate --kv-disk-dir from the fork's kv-cache-0731. On-disk KV format
# compatibility between the two engines is UNVERIFIED, so mainline gets its own
# directory: the fork's warm cache stays intact for rollback and there is no
# cross-format risk. Cost is a cold disk-KV cache until this one fills.
