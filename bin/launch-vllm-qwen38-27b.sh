#!/bin/bash
# Qwen3.8-27B — DAY-0 LANE, staged 2026-08-14 BEFORE the weights were public.
# Port 9030. Nothing here has been run against real weights yet: every value is
# either carried over from our working Qwen3.6-27B lanes or taken from the
# pre-release model card. Treat the first launch as an experiment, not a deploy.
#
# ── WHAT THE PRE-RELEASE CARD SAYS ───────────────────────────────────────
#   dense 27B, 64 layers, Gated DeltaNet + Gated Attention (hybrid linear/full
#   attention), native vision encoder, native MTP head, 262144 ctx natively and
#   "extensible up to 1,000,000", thinking ON by default with reasoning_effort
#   in {xhigh, medium, low}, BF16 weights at release (no official quant yet).
#   The card's long-context section uses --hf-overrides '{"text_config": ...}'
#   with YaRN, i.e. the config is a multimodal wrapper the same shape as
#   Qwen3.6-27B's Qwen3_5ForConditionalGeneration.
#
# ── WHY WE THINK THIS WILL JUST WORK ─────────────────────────────────────
#   Qwen3.6-27B is ALREADY a Gated DeltaNet hybrid — see
#   launch-vllm-27b-nvidia-nvfp4.sh: "48 linear-attention (Gated DeltaNet) + 16
#   full-attention layers". vLLM has shipped that path since PR #41025, and our
#   local v0.26.0 image registers the classes 3.6 loads under:
#       Qwen3_5ForConditionalGeneration, Qwen3_5MTP,
#       Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeMTP
#   Qwen3.6 reused Qwen3.5's architecture string. IF 3.8 does the same, this
#   image loads it on day 0 with no new engine. If instead config.json says
#   Qwen3_8ForConditionalGeneration, v0.26.0 will hard-fail with
#   "not supported" and you need the nightly (IMAGE= below). Released vLLM as
#   of 2026-08-14 is v0.27.1 and has NO Qwen3.8 entry, so nightly is the bet.
#   FIRST THING TO DO ON RELEASE:
#       hf download Qwen/Qwen3.8-27B config.json --local-dir /tmp/q38
#       grep -E 'architectures|model_type' /tmp/q38/config.json
#   That one grep decides which image you need. Do it before pulling 54 GB.
#
# ── SIZING ON THIS BOX (121.6 GB unified, GB10/sm_121) ───────────────────
#   BF16 27B ~= 54 GB of weights, and this is the first Qwen 27B generation we
#   can serve at full precision without waiting for a community quant. The
#   hybrid attention keeps KV small — only ~16 of 64 layers carry growing KV
#   (that is why 3.6 held 262144 ctx at c=4 in 0.85 util). Budget accordingly:
#   0.85 util is the solo, swap-exclusive setting. There is NO co-residency
#   with ds4 (which needs ~113 GB) — this is a straight replacement. GB10 hard
#   -hangs the whole box on host OOM, so never start this while ds4 is loaded;
#   run bin/park-ds4-for-eval.sh first.
#
# ── KNOBS AND WHY ────────────────────────────────────────────────────────
#   --mamba-cache-dtype float32 : the Gated DeltaNet recurrent state cache needs
#       fp32 for numerical stability (AEON recipe, carried from our NVFP4 lane).
#   KV_DTYPE unset by default : do NOT pin kv-cache-dtype for a native-MTP run.
#       A mismatch between drafter and target KV dtype collapsed DFlash
#       acceptance to 0.01% (6/51,030) once already. Our NVFP4 lane pins
#       bfloat16 only because DFlash specifically needs it.
#   SPEC : CONFIRMED 2026-08-14 — native MTP is the ONLY speculation available.
#       config.json text_config has mtp_num_hidden_layers=1 and the 15 mtp.*
#       tensors are present in BOTH the BF16 index (15/1199) and the unsloth
#       NVFP4 index (15/1968), so the head survives quantisation and vLLM's
#       Qwen3_5MTP can attach to either. There is NO DFlash/Eagle drafter for
#       3.8 on HF (searched DFlash/Eagle/draft; z-lab's newest anything is
#       2026-07-05). On Qwen3.6, native MTP n=1 measured ~2-3x SLOWER than
#       DFlash n=10, so expect MTP to be the modest option, not the fast one.
#       Always A/B against SPEC=none — speculation that is not accepted costs
#       throughput rather than adding it.
#   --reasoning-parser qwen3 : ONLY for smoke tests. It buffers the entire
#       <think> block before emitting a single token, which destroys interactive
#       TTFT. Drop it for the production lane — see
#       memory feedback_reasoning_parser_ttft.
#   Thinking is ON by default on this model and a small max_tokens will be eaten
#       entirely by the <think> block (that bit us on 3.6 at 1024-1536). Raise
#       client max_tokens rather than disabling thinking.
#
# ── EXPECTATION MANAGEMENT ───────────────────────────────────────────────
#   Incumbent is ds4 (Entrpi): 19.55 tok/s decode @34.6k, TTFT 32.7 s warm /
#   ~370 s cold, ctx 131072, and NO concurrency (it serialises prefills; at c=4
#   every stream saw 67 s TTFT on an 8k prompt).
#   For reference on THIS box: Qwen3.6-27B official FP8 + native MTP was 14.8
#   tok/s; the 26-27 tok/s figure needed Intel INT4 + DFlash n=4. So day-0
#   BF16+MTP landing near or below ds4 on raw decode would NOT be a surprise.
#   The case for switching is 262k ctx, real c=4 concurrency, native vision, and
#   no cold-start cliff — not tok/s. Bench that, not decode speed alone.
set -e

# ── 2026-08-14, WEIGHTS ARE OUT — CONFIRMED FROM THE REAL config.json ────
#   architectures: ['Qwen3_5ForConditionalGeneration'], model_type: qwen3_5,
#   text_config.model_type qwen3_5_text, 64 layers, full_attention_interval 4
#   (= 16 full / 48 linear, same split as 3.6), max_position_embeddings 262144,
#   vision_config present. So the guesses above held: BOTH our v0.26.0 image and
#   the AEON sm121a image already register this class. No new engine needed.
#
# ── HOW TO INVOKE, PER QUANT ─────────────────────────────────────────────
#   BF16 (Qwen/Qwen3.8-27B, 55.6 GB) — the quality reference:
#       ./launch-vllm-qwen38-27b.sh
#   FP8 official (Qwen/Qwen3.8-27B-FP8, 30.9 GB):
#       MODEL_ID=Qwen/Qwen3.8-27B-FP8 ./launch-vllm-qwen38-27b.sh
#   NVFP4 (unsloth/Qwen3.8-27B-NVFP4, 23.4 GB) — MUST use the AEON image:
#       MODEL_ID=unsloth/Qwen3.8-27B-NVFP4 \
#       IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
#       ./launch-vllm-qwen38-27b.sh
#     LEAVE QUANT EMPTY. Checked 2026-08-14: this quant is packaged as
#     quant_method=compressed-tensors, NOT modelopt — passing QUANT=modelopt
#     would be wrong. It is a MIXED quant: FP8 (num_bits 8, per-token dynamic)
#     on self_attn/linear_attn/lm_head and on layers 56-63 MLP, 4-bit on the
#     rest of the MLPs. Same shape as nvidia's 3.6 NVFP4 mixed-precision.
#     On a STOCK image NVFP4 can silently produce GARBAGE on sm_121 (Marlin FP4
#     is SM80-targeted -> wrong logits -> "!!!!"). VLLM_NVFP4_GEMM_BACKEND=
#     flashinfer-cutlass below is the fix and only the AEON build honours it.
#     VERIFY OUTPUT SANITY on the very first completion before trusting any
#     benchmark number — a Marlin-garbage lane still reports excellent tok/s.
# MODEL_ID may be an HF repo id OR a local directory. Local dirs are mounted at
# the SAME path inside the container (-v /home/max/models:/home/max/models:ro)
# because transformers only treats an argument as a path if that path actually
# resolves inside the container; otherwise it parses it as a repo id and dies
# with "Repo id must be in the form 'repo_name' or 'namespace/repo_name'".
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
# Image ladder, try in this order:
#   v0.26.0-aarch64  — known-good for the Qwen3_5* hybrid classes (our 3.6 lanes)
#   nightly-aarch64  — needed if 3.8 introduces a new architecture string
#   aeon-vllm-ultimate:latest — ONLY for NVFP4 weights on sm_121; stock images
#       route W4A16-NVFP4 to the SM80 Marlin kernel and emit garbage ("!!!!").
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-aarch64}"
PORT="${PORT:-9030}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-4}"
# 2026-08-15 HARD-HANG: UTIL=0.85 + MTP n=3 OOM-hung the box (NVRM logged
# NV_ERR_NO_MEMORY during warmup at 23:59, hang at ~00:08, power cycle needed).
# A concurrent 30 GB download shared the blame, but the driver was already
# failing allocations before the probe started. 0.80 default now; the 3.6
# concurrency work ran 0.70. NEVER run downloads/page-cache-heavy jobs while
# this engine is loading or benching. After launch, check:
#   journalctl -k --since "<launch time>" | grep NV_ERR_NO_MEMORY
# any hit = do not bench, lower UTIL.
UTIL="${UTIL:-0.80}"
SPEC="${SPEC:-mtp}"          # mtp | none
SPEC_N="${SPEC_N:-3}"        # MTP draft depth. See note below — 1 leaves speed on the table.
BATCH="${BATCH:-16384}"      # max-num-batched-tokens
KV_DTYPE="${KV_DTYPE:-}"     # leave EMPTY unless you know why
# NVFP4 GEMM backend. EMPTY = vLLM default, which on sm_121 is MARLIN.
#   2026-08-14: we originally forced flashinfer-cutlass here, inherited from the
#   Qwen3.6 NVFP4 lane where stock Marlin produced garbage for MODELOPT-packed
#   W4A16. That does NOT generalise: GB10/sm_121 has no native FP4 tensor cores
#   (no cvt.rn.satfinite.e2m1x2.f32 PTX), so every NVFP4 path dequantises FP4
#   ->BF16 in-kernel anyway, and on sm_121 FlashInfer falls back to a broken/
#   slow CUTLASS codepath while Marlin is the path the DGX-Spark community
#   converged on (~16% faster). Set to flashinfer-cutlass ONLY if output turns
#   to garbage without it — and verify text before trusting any tok/s number.
NVFP4_BACKEND="${NVFP4_BACKEND:-}"
QUANT="${QUANT:-}"           # empty for BF16; "modelopt" for NVFP4 weights
VISION="${VISION:-0}"        # 1 = drop --language-model-only, accept images
NAME="vllm-qwen38-27b"

EXTRA=()
[ -n "$KV_DTYPE" ] && EXTRA+=(--kv-cache-dtype "$KV_DTYPE")
[ -n "$QUANT" ]    && EXTRA+=(--quantization "$QUANT")
[ "$VISION" = "1" ] || EXTRA+=(--language-model-only)
case "$SPEC" in
  mtp)  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_N}}") ;;
  none) ;;
  *)    echo "unknown SPEC=$SPEC (want: mtp|none)" >&2; exit 2 ;;
esac

# Refuse to start on top of a loaded ds4 — host OOM hard-hangs this box.
# Use `pgrep -x` on the exact process name plus a listening-port check. Do NOT
# use `pgrep -f 'ds4-server .*--port 9010'`: -f matches full command lines, so
# any parent shell whose cmdline happens to contain the pattern text (including
# the shell running this very check) self-matches and the guard refuses for no
# reason. Cost us a false REFUSE on 2026-08-14.
ds4_running=0
pgrep -x ds4-server >/dev/null 2>&1 && ds4_running=1
ss -ltn 2>/dev/null | grep -q ':9010[[:space:]]' && ds4_running=1
if [ "$ds4_running" = "1" ]; then
  echo "REFUSING: ds4 is still resident on 9010 (~113 GB)." >&2
  echo "Starting a second engine beside it exceeds the 121.6 GB pool and GB10" >&2
  echo "hard-hangs the whole box on host OOM (power cycle required)." >&2
  echo "Run bin/park-ds4-for-eval.sh first." >&2
  exit 1
fi

# Second belt: no OTHER vLLM engine may be resident. Two engines on this box
# exceed the 121.6 GB unified pool and GB10 hard-hangs on host OOM (power cycle
# required). We remove our OWN container by name further down, so exclude it.
others=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^vllm-|vllm' | grep -v "^${NAME}$" || true)
if [ -n "$others" ]; then
  echo "REFUSING: another vLLM engine is already running:" >&2
  echo "$others" | sed 's/^/  /' >&2
  echo "Stop it first (docker rm -f <name>) — two engines OOM-hang this box." >&2
  exit 1
fi

# Third belt: require real headroom, whatever is or isn't running.
avail_gb=$(free -g | awk '/^Mem:/{print $7}')
need_gb="${NEED_GB:-60}"
if [ "$avail_gb" -lt "$need_gb" ]; then
  echo "REFUSING: only ${avail_gb} GB available, need >= ${need_gb} GB." >&2
  echo "Override with NEED_GB=<n> only if you know the model fits." >&2
  exit 1
fi

echo "launching $MODEL_ID on :$PORT  image=$IMAGE ctx=$CTX seqs=$SEQS spec=$SPEC"
docker rm -f "$NAME" 2>/dev/null || true
exec docker run --name "$NAME" \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:${PORT}:${PORT} \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/llm-stack/etc:/llm-stack-etc:ro \
  -v /home/max/models:/home/max/models:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL_ID" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$CTX" \
  --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "$BATCH" \
  --mamba-cache-dtype float32 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --attention-backend flash_attn \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --trust-remote-code \
  "${EXTRA[@]}"
