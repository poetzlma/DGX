#!/bin/bash
# DeepSeek-V4-Flash on the Entrpi/ds4 fork (v0.5.6.2) — EVAL LANE, added 2026-08-10.
#
# WHY THIS EXISTS: measured 2026-08-10 on this box against antirez mainline
# 84cc882, same GGUF (hard-linked, one inode), same prompts, server harness,
# decode measured first->last token so prefill is excluded:
#
#     LONG 34.6k coding ctx   14.10 -> 19.55 tok/s   (1.39x)
#     SHORT                   17.95 -> 20.14 tok/s   (1.12x)
#     TTFT @34.6k             40.2s -> 32.7s         (1.23x)
#
# The fork's plain decode also beats mainline WITH DSpark (15.72), so this lane
# does not need speculation to win. Full writeup:
# ~/.claude/projects/-home-max/memory/project_entrpi_ds4_eval_20260810.md
#
# --no-spec IS DELIBERATE. The fork's own DSpark drafter is a NET LOSS at long
# context here: 17.90 vs 19.55 plain (-8.4%, reproduced across both reps), while
# winning +9.9% on short prompts. Our traffic is ~100k:4k, so the long-context
# behavior is the one that matters. Counters were healthy while losing
# (accept_ratio 0.64-0.68, tok_per_step 2.5, quench_events 0) — it is a
# break-even/scheduling issue, not a broken drafter. Revisit with a re-bench on
# captured opencode traffic before turning speculation on.
#
# THIS IS THE LIVE DEFAULT LANE as of 2026-08-10 (user's call, knowing the
# engine is new here). It serves `deepseek-v4-flash-0731` — the same model name
# the gateway already routes to — so ALL ds4 traffic lands on the fork.
#
# ROLLBACK IS ONE LINE: in config/llama-swap.yaml point
#   deepseek-v4-flash-0731.cmd  ->  /home/max/llm-stack/bin/launch-ds4-server.sh
# That script and the mainline binary are untouched on disk. The config is
# live-watched, so the edit alone bounces the lane back (~10 min, no restart).
#
# QUALITY GATE: smoke-ds4-0731.py was run against this binary before cutover —
# see ~/entrpi-smoke-results.txt for the score. Same weights as prod, but a
# different ENGINE: kernels, sampling and tool-call handling are all fork code,
# so the model is the only thing that did not change.
set -e
# Weights: hard link to the same inode as the prod GGUF (verified identical to
# the HF content-length). Not a copy — costs no disk.
MODEL=/home/max/entrpi-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf

exec /home/max/entrpi-src/ds4-server \
  --cuda \
  --host 127.0.0.1 --port 9010 \
  -m "$MODEL" \
  --ctx 131072 \
  --no-spec \
  --kv-disk-dir /home/max/ds4/kv-cache-entrpi \
  --kv-disk-space-mb 32768 \
  --mem-floor-gb 8
# NOTE boot is ~90s, slower than mainline's ~27s: the fork builds its aligned
# fast-path weight artifacts in-process at startup (it logs "artifact_source").
# The installer also built a resident weight server (~/entrpi-src/ds4_weight_server)
# that can cut this to seconds by importing the model over IPC — not wired up
# here, worth doing if this lane is ever promoted.
#
# 2026-08-10 CTX 262144 -> 131072 (REVERTED, same day). The 256K bump caused a
# production outage ~15:53 and is NOT to be re-applied without the fix below.
#
# WHAT WENT WRONG. The reasoning for 256K was that KV slabs are DEMAND-MAPPED
# (boot log: "comp/index slabs demand-mapped, virtual 2238 MiB/bank, floor 97.8
# MiB/bank"), so the budget is virtual and backed only as contexts actually
# grow — "nearly free." The virtual budget IS free. The BACKING is not, and
# nothing caps its growth. Real traffic grew the slabs until the engine's
# 116.9 GiB allocation left MemAvailable at 1.7 GiB, against --mem-floor-gb 8.
# From that point every deep admission was unfundable:
#
#   ds4.c:35841  "cont admit rejected on memory floor (... the box cannot fund
#                 this admission; --mem-floor-gb and DS4_SERIAL_RESERVE_CTX
#                 govern)"
#
# and each rejected job bounced to the serial path, where the deep-serial guard
# (ds4_server.c:15708, DS4_SERVER_SERIAL_MAX_TOKENS, default 65536) refused it
# 503 because our coding prompts are ~65-100k tokens. Net effect: 33 refusals,
# ZERO completions, ~50 ms each — the lane was hard-down for all >64k traffic
# while looking healthy (process alive, cont_batch_failures_total 0,
# requests_inflight 0). It does not self-recover: the guard's comment assumes
# the memory dip is "transient (comp budget read during a memory dip)", but
# once the slabs have grown, MemAvailable never climbs back over the floor.
#
# WHY --mem-floor-gb 8 DID NOT SAVE US: it is enforced at ADMISSION, not by
# shrinking. It correctly refused to fund work — which is the 503s. It protects
# the box from the host-OOM hard-hang (see project_ds4_upstream_oom); it does
# not protect the lane from becoming unservable. Both things are true.
#
# BEFORE RETRYING DEEP CTX, you need a cap on backed slab growth (or a floor
# breach that evicts rather than refuses). Raising the ctx alone just moves the
# cliff. Concurrency is NOT the reason to want it: we MEASURED c=4 aggregate at
# 0.92x of c=1, fully serialized, tok_per_step 1.000. Upstream ships -c 262144
# as the ds4-serve default and documents 524288 as deepest tested — neither is
# on a 119 GiB unified-memory box with a 116 GiB resident model.
# Expect deep decode to be slow regardless: upstream's own data is ~146
# ms/token (~7 t/s) near 248k.
#
# NOTE the disk-KV dir is SEPARATE from the mainline lane's kv-cache-0731-mainline.
# On-disk KV format compatibility between the two engines is UNVERIFIED and
# mixing formats risks corruption, so the fork gets its own directory: the
# mainline warm cache stays intact for rollback. Cost is a cold cache until this
# one fills. Prefix caching is load-bearing for our ~100k:4k traffic, so this
# flag is not optional — dropping it would tank TTFT on repeated prefixes.
#
# NOTE --mem-floor-gb 8 keeps 8 GiB of headroom against live free memory before
# the cache is allowed to grow. This box HARD-HANGS on host OOM (needs a power
# cycle, see project_ds4_upstream_oom), so the floor is a deliberate guard, not
# a tuning knob. UNMEASURED on this lane — if TTFT looks bad on long prefixes,
# this is the first thing to loosen.
