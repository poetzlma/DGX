#!/bin/bash
# ds4 UPSTREAM engine (antirez/ds4 @ 54b36ed, 2026-07-28) + the Q4K-hybrid 0731
# GGUF. Eval lane, port 9099. Added 2026-08-02.
#
# WHY A NEW BINARY: the pinned prod binary (~/ds4-q4, antirez be43477 + ngc-shj
# Q4 patches, tip da6f723 / 2026-05-12) can DECODE Q4_K experts but cannot
# PREFILL them — its only Q4_K kernels are
#   moe_gate_up_mid_decode_q4K_qwarp32_kernel
#   moe_down_q4K_sum6_qwarp32_kernel
# and the prefill path goes through the *_expert_tile* family, which has no q4K
# variant. Result: the engine loads the Q4K-hybrid fine (90.89 GiB mapped) then
# returns HTTP 500 `cuda prefill failed` on the first request.
#
# Upstream (276 commits ahead) adds the missing tile kernels —
#   moe_gate_up_mid_q4K_expert_tile8_rowspan_kernel
#   moe_down_q4K_expert_tile8_rowspan_kernel
#   + q4K tile8/tile16 MMA variants
# cuda_block_q4_K use goes 5 -> 50, and tensor_is_routed_expert_type() widens
# from {IQ2_XXS,Q2_K,Q4_K} to also accept Q8_0, Q5_K, Q6_K.
#
# CHANGES vs the prod launch script:
#   - no --warm-weights   (flag removed upstream)
#   - no DS4_CUDA_Q4_DECODE=1 (env removed; Q4 decode is native now)
#   - --prefill-chunk now defaults to 4096 upstream (prod pinned 2048)
# DS4_CUDA_Q8_F16_CACHE_RESERVE_MB still exists and is still needed: GB10's
# cudaMemGetInfo reports ~5 GiB free on a unified pool with far more headroom,
# so the default 5% reserve trips on the first request and latches the slow q8
# path for the process lifetime (~340 -> ~95 tok/s prefill).
#
# Built with: make cuda CUDA_ARCH=sm_120   (antirez's documented arch; GB10
# reports sm_121 but a prior sm_121 rebuild of the OLD code measured WORSE.
# Re-test the arch separately before changing it here.)
# ############################################################################
# !! DANGER — THIS SCRIPT OOM'd AND HARD-HUNG THE BOX ON 2026-08-02 21:55 !!
# It required a physical power cycle. Do not run it again without a memory
# guard and someone watching.
#
# WHAT HAPPENED (from logs/ds4-upstream-q4k.log):
#   ds4: CUDA (no-copy) host registration skipped: operation not supported
#   ds4: CUDA loading model tensors into device cache
#   ds4: ... 16 -> 80 GiB cached, while "prepared model tensor mappings" grew
#         in lockstep to 80.45 GiB
#   ds4: memory: ... resident model 90.88 GiB = 93.92 GiB planned
# The planned 93.92 GiB was fine. The problem is that upstream's no-copy host
# registration FAILS on this box ("operation not supported"), so it falls back
# to materialising a second, device-side copy of the weights on top of the host
# mapping — ~2 x 90.88 GiB against a 121.6 GiB unified pool. Kernel logged
# NVRM NV_ERR_NO_MEMORY repeatedly at 21:55:23-27, then a hung task at 22:02.
#
# The pinned prod fork does NOT have this problem: it logs
#   "CUDA registered 90.89 GiB model mapping for device access"
# i.e. genuine no-copy. The ngc-shj fork carries a host-register fallback patch
# (same one that took prefill 96 -> 396 tok/s) that upstream lacks.
#
# BEFORE RETRYING, in this order:
#   1. Find why cudaHostRegister returns "operation not supported" here
#      (ds4_cuda.cu:3274). Likely needs the ngc-shj fallback cherry-picked, or
#      GGML_CUDA_ENABLE_UNIFIED_MEMORY-style handling for GB10 unified memory.
#   2. Test with --simulate-used-memory or a smaller --ctx first.
#   3. Run under a watchdog that kills the process if used memory crosses
#      ~110 GiB, and never while prod is the only coding lane.
# Override intentionally with: ALLOW_OOM_RISK=1 <this script>
# ############################################################################
set -e
if [ "${ALLOW_OOM_RISK:-0}" != "1" ]; then
  echo "refusing to run: this configuration OOM'd and hard-hung the box on 2026-08-02." >&2
  echo "read the header, then re-run with ALLOW_OOM_RISK=1 if you really mean it." >&2
  exit 1
fi
export DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=1024
MODEL="${MODEL:-/home/max/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf}"
exec /home/max/ds4-upstream/ds4-server \
  --cuda \
  --host 127.0.0.1 --port "${DPORT:-9099}" \
  --model "$MODEL" \
  --ctx "${DCTX:-131072}" \
  --kv-disk-dir /home/max/ds4/kv-cache-upstream \
  --kv-disk-space-mb 32768
