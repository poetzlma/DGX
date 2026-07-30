#!/bin/bash
# poolside/Laguna-S-2.1-NVFP4 — V2 "spinquantless norot" weights (2026-07-23).
#
# Pinned to upstream commit 0761412 (re-upload ~2026-07-23): weights re-quantized
# WITHOUT SpinQuant rotations, resharded 14->15 files, config.json observer
# static_minmax->minmax. Apparent (unconfirmed by poolside) fix for the looping
# reports (HF discussions #4-#7, #10); #11 suggests the original rotate checkpoint
# was never runnable correctly on public runtimes. Early user reports: overthinking
# loops reduced but not gone on long creative gen.
#
# Drop-in replacement for launch-vllm-laguna-s21-nvfp4.sh (same port 9030, same
# container name, same served-model-name) — switch = point llama-swap cmd here.
# CAUTION: config/llama-swap.yaml is live-watched; editing it IS the cutover.
#
# A/B NOTES vs v1:
#   - Identical flags except --revision. Do not change two variables at once.
#   - DFlash drafter (unchanged upstream, single commit) was presumably trained
#     against the ORIGINAL target: watch acceptance rate / speedup after switch.
#     If decode tok/s tanks vs the ~33 tok/s @100k baseline, LAG_SPEC=0 to A/B.
#   - LAG_THINK=0 drops the server-side enable_thinking default (HF #6 reports
#     forcing it server-side triggers looping; clients then opt in per-request).
#     Keep =1 for the first A/B so only the weights change.
#
# Env knobs (same as v1 unless noted):
#   LAG_SPEC=0   -> detach DFlash drafter (default 1)
#   LAG_CTX      -> --max-model-len (default 262144)
#   LAG_UTIL     -> --gpu-memory-utilization (default 0.85)
#   LAG_SEQS     -> --max-num-seqs (default 8 since 2026-07-24, DGX issue #2;
#                   was 4 — multi-agent consumers were queue-starved while KV
#                   sat at ~45%. 8x48k avg = 384k of the 858k pool.)
#   LAG_IMAGE    -> container image (default upstream v0.25.1 aarch64)
#   LAG_MOE      -> --moe-backend override; LEAVE UNSET on GB10 (see v1 header)
#   LAG_SPEC_N   -> num_speculative_tokens (default 15)
#                   A/B RESOLVED 2026-07-30: KEEP 15. The NVIDIA-forum claim
#                   that draft positions 6-15 accept ~0% does NOT hold on our
#                   traffic — measured per-position acceptance at 100k decays
#                   smoothly 76.9% (pos 0) -> 2.5% (pos 14), and positions
#                   7-14 supply 17.3% of ALL accepted tokens. Tail positions
#                   only collapse to exactly 0.0 at 13k (start-of-session);
#                   the 60-100k bell curve is where n=15 earns its keep, and
#                   the 07-22 n=7 A/B already lost there (25.9 vs 33.3 tok/s).
#                   MiaAI-Lab pins n=7 — that is their config, not a fix.
#   LAG_THINK    -> NEW: 0 drops --default-chat-template-kwargs enable_thinking
set -e

# 2026-07-24 CO-RESIDENT CONFIG (later same day): seqs back to 4, DFlash back
# on, util 0.85 -> 0.63 to make room for the qwen 35B fast lane (see
# launch-vllm-35b-moe-nvfp4-colag.sh). History of the seqs-8 experiment:
#   - seqs 4->8 (DGX issue #2): DFlash n=15 + seqs 8 deadlocked the engine
#     core twice under real c=8 traffic (~06:51 and 07:40 UTC; token counter
#     frozen, /health still 200; logs in session scratchpad). Synthetic c=8
#     probe passed — trigger needs the real 48k-prompt mix.
#   - seqs 8 drafterless: stable but ~19 tok/s single-stream (vs 43-48).
#   - Community stable envelope is seqs<=4 WITH drafter (MiaAI-Lab stack;
#     DFlash crashes vLLM outright at the 256 default).
# 2026-07-24 (evening): BACK TO SOLO 0.85 / 262144 — qwen co-residency parked
# (its vLLM load transient trips the host-OOM floor next to resident laguna;
# see launch-vllm-35b-moe-nvfp4-colag.sh + memory project-coresident-split).
# seqs stays 4 + DFlash (seqs-8 is dead, see below). If the fast lane comes
# back, re-shrink BOTH: laguna 0.66/131072 max, qwen <=0.24 — measured math:
# DO NOT raise LAG_SEQS above 4 while LAG_SPEC=1. Co-resident memory math
# (measured 2026-07-24, four crash-loop attempts at util 0.63):
#   non-KV footprint (weights+drafter+runtime) = ~70.6 GB, KV/token = 38.4 KB.
#   vLLM refuses to start unless ONE max-model-len request fits in KV.
#   util 0.66 (~79 GB) -> KV ~8.4 GB ~= 219k tokens; ctx 131072 needs 4.8 ✓.
#   qwen lane takes 0.25 (~30 GB); sum ~109 GB, ~10 GB host headroom.
# GB10 hard-hangs on host OOM: do not raise either util unilaterally, and
# 256k ctx (LAG_CTX=262144) is SOLO-MODE ONLY (needs KV >= 9.6 GB + margin).
# DRAFTER REVISION IS PINNED (2026-07-30). It was NOT before, and that bit us
# silently: the target carries --revision but speculative_config named the
# drafter with no revision, so it resolved `main` on every start. poolside
# pushed drafter 4cdcc6e "Spinquant removal adaptation (#6)" on 2026-07-27 and
# the 07-28 11:44 UTC bounce swallowed it with no config change and no record
# (engine log: 73s re-download of the DFlash weights).
#
# That commit's ONLY substantive change is rope_theta 10000.0 -> 500000.0,
# matching the target (configuration_laguna.py defaults rope_theta=500000.0).
# So the drafter shipped a 50x-wrong RoPE base from initial release 07-21 until
# 07-27 — its positional encoding never matched the model it drafted for.
# MEASURED IMPACT (2026-07-30, bench-coding-realistic vs the 07-22 runs):
# smaller than the bug sounds. accept_len 13k 2.50->2.40, 60k 3.26->4.54,
# 100k 4.60->4.50; decode 100k 33.1->32.5 t/s. The 60k gain is inside the
# noise envelope of a single-fire bucket (5-8 samples; mean 23.6% vs median
# 30.0% acceptance). Verdict: keep the fixed drafter because the config is now
# internally consistent, but do NOT credit it with a throughput win.
# Bounded by design: the drafter is 6x sliding_attention/512, so it never
# attends past 512 tokens and the rope error was bounded by that window —
# which is why the damage was roughly context-independent, not long-ctx-only.
# LAG_DRAFT_REV overrides the pin (use to A/B a new upstream drafter).
DRAFT_REV="${LAG_DRAFT_REV:-4cdcc6e9b29105e8ff5790885cadccbeb4f33f54}"

# LAG_SPEC_KV: drafter KV dtype, vLLM >= 0.26.0 ONLY (PR #48787). Unset = the
# field is omitted, which defaults to None = INHERIT the target's --kv-cache-dtype
# (fp8 here) — i.e. exactly the pre-0.26.0 behaviour, so leaving it unset keeps
# an image swap a single variable. The BF16 drafter having its KV forced to fp8
# is the last untested acceptance lever; "auto" gives it bf16 KV instead.
# Costs KV pool: the drafter's cache stops being half-size. Valid values incl.
# auto|bfloat16|float16|fp8|fp8_e4m3. On vLLM 0.25.1 this key is REJECTED, so
# it must stay unset whenever LAG_IMAGE points back at v0.25.1.
# MEASURED 2026-07-30 — LAG_SPEC_KV=auto IS CATASTROPHIC. DO NOT SET IT.
# On vLLM 0.26.0 the drafter got bf16 KV while the target stayed fp8, and
# acceptance collapsed to 6 accepted tokens out of 51,030 drafted (0.01%,
# accept_len 1.002 vs 4.31 normal). DFlash needs its KV dtype to MATCH the
# target's; mismatch => every draft rejected and spec-decode becomes pure
# overhead (the bench blew past a 10-min timeout it normally clears in ~4).
# It also cost 22.7% of the KV pool (810k -> 626k) since the drafter cache
# stops being half-size. Not documented in PR #48787 or the vLLM recipe.
# Knob is kept ONLY so a future fp8-capable drafter can be A/B'd explicitly.
#
# KV-POOL CAVEAT for anyone comparing versions by pool size: the pool varies
# run-to-run on IDENTICAL config — measured 821,116 and 858,126 tokens on two
# consecutive v0.25.1 starts. v0.26.0's 810,622 is just below that spread, so
# pool size does NOT discriminate between the versions. Use accept_len/TTFT.
SPEC_KV_FIELD=""
if [ -n "${LAG_SPEC_KV:-}" ]; then
  SPEC_KV_FIELD=",\"kv_cache_dtype\":\"${LAG_SPEC_KV}\""
fi
SPEC_ARGS=()
if [ "${LAG_SPEC:-1}" = "1" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"poolside/Laguna-S-2.1-DFlash-NVFP4\",\"revision\":\"${DRAFT_REV}\",\"num_speculative_tokens\":${LAG_SPEC_N:-15}${SPEC_KV_FIELD}}")
fi

THINK_ARGS=()
if [ "${LAG_THINK:-1}" = "1" ]; then
  THINK_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
fi

docker rm -f vllm-laguna-s21 2>/dev/null || true
exec docker run --name vllm-laguna-s21 \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9030:9030 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/.cache/vllm-laguna:/root/.cache/vllm \
  -v /home/max/llm-stack/etc:/llm-stack-etc:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint vllm \
  "${LAG_IMAGE:-vllm/vllm-openai:v0.25.1-aarch64-ubuntu2404}" \
  serve poolside/Laguna-S-2.1-NVFP4 \
  --revision 07614121b31898586430f189d27a25a0be310843 \
  --served-model-name laguna-s-2.1 \
  --host 0.0.0.0 --port 9030 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  ${LAG_MOE:+--moe-backend ${LAG_MOE}} \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization ${LAG_UTIL:-0.85} \
  --max-model-len ${LAG_CTX:-262144} \
  --max-num-seqs ${LAG_SEQS:-4} \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --tool-call-parser poolside_v1 \
  --enable-auto-tool-choice \
  --reasoning-parser poolside_v1 \
  --override-generation-config '{"max_new_tokens": 131072}' \
  "${THINK_ARGS[@]}" \
  "${SPEC_ARGS[@]}"
