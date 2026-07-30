# Model reference

Every route the gateway serves: the full matrix, what each model is for, and the
load-bearing per-launcher configuration. Settings live in each launcher script
(`bin/<launcher>`) — this page explains *why* they're set that way. Operational
runbook: [operations.md](operations.md) · rationale history: [decisions.md](decisions.md).

## Models

Pick by `Route` (the `model` value). The **Tier** column says whether the model is always loaded (`resident`) or pulled from codeserver on demand (`copy-back`, +1–3 min first-load / ~13 min for ds4 — see [Weight offload](operations.md#weight-offload--codeserver-copy-back)). Speed/mem are single-stream figures from prior sweeps (`bin/bench-models.py`; raw in `logs/`). **Settings for each model live in its launcher** (`bin/<launcher>`) and are explained in [Per-launcher details](#per-launcher-details).

| Route (`model`) | Tier | Use it for | tok/s | Max ctx | Peak mem | Engine | Launcher (`bin/`) |
|---|---|---|---:|---:|---:|---|---|
| **`laguna-s-2.1`** ⟵ aliases `qwen3.6-27b`, `qwen3.6-35b-a3b`, `nemotron-3-puzzle-75b` | **resident** | **coding — default** | 43–54 | **256 k**⁶ | ~101 GB | vLLM | `launch-vllm-laguna-s21-nvfp4-v2.sh` |
| `nemotron-3-puzzle-75b` | parked | coding — **rollback** (was default 07-08 → 07-22) | 28 | 131 k⁵ | ~65 GB | vLLM | `launch-vllm-nemotron-puzzle-75b-mtp.sh` |
| `qwen3.6-35b-a3b-vision` | **dark** | **vision** (images) — no headroom beside laguna | 56 | 131 k | ~34 GB | vLLM⁴ | `launch-vllm-qwen-fast.sh` |
| `qwen3.6-27b-int4-dflash` | copy-back | coding — **rollback** target for the default | 29 | 120 k | 61 GB | vLLM | `launch-vllm-27b-int4-dflash.sh` |
| `qwen3.6-35b-a3b-nvfp4` | copy-back | coding, long-ctx throughput | 56 | 131 k | 53 GB | vLLM | `launch-vllm-35b-moe-nvfp4.sh` |
| `qwen3.6-27b-nvfp4` | copy-back | coding — NVIDIA NVFP4, **256 k ctx** | 31 | **256 k** | ~101 GB | vLLM | `launch-vllm-27b-nvidia-nvfp4.sh` |
| `qwen3.6-27b-nvfp4-vision` | copy-back | **vision** (images) at 256 k ctx² | 31 | **256 k** | ~101 GB | vLLM | `launch-vllm-27b-nvidia-nvfp4-vision.sh` |
| `ornith-1.0-35b` | copy-back | coding — agentic (thinking)² | 77 | 131 k¹ | 25 GB | llama.cpp | `launch-ornith.sh` |
| `qwopus3.6-27b-int4-dflash` | copy-back | coding — Opus-distilled | 39 | 131 k | 102 GB | vLLM | `launch-vllm-qwopus-int4-dflash.sh` |
| `qwen3.6-27b-fp8` | copy-back | coding — quality baseline | 21 | 131 k | 91 GB | vLLM | `launch-vllm-27b-qwen-fp8.sh` |
| `nemotron-3-nano-omni` | copy-back | multimodal (image/audio/video) | 56 | 131 k | 91 GB | vLLM | `launch-vllm-nemotron-omni.sh` |
| `cosmos3-nano-omni` | copy-back | image/video generation³ | ~6 s/img | — | ~30 GB | vLLM-Omni | `launch-vllm-cosmos3-nano-omni.sh` |
| `diffusiongemma-26b` | copy-back | speed / non-coding | 116 | 131 k | 50 GB | vLLM | `launch-vllm-diffusiongemma-nvfp4.sh` |
| `deepseek-v4-flash-ds4` | copy-back | long-context planner | 21 | 131 k | 22 GB | ds4 | `launch-ds4-server.sh` |

¹ llama.cpp lanes split total engine ctx across parallel slots, env-tunable: Ornith `ORNITH_CTX`/`ORNITH_PARALLEL` (3 slots × 131 k), 35B-vision `QWEN35B_CTX`/`QWEN35B_PARALLEL` (2 slots × 131 k). &nbsp; ² **Thinking model** — output goes to `reasoning_content`; give generous `max_tokens` or `content` returns empty. &nbsp; ³ **Generation model, not chat** — call `POST /v1/videos` · `/v1/videos/sync` · `/v1/images/generations` (multipart), not `/v1/chat/completions`; tok/s and token-ctx don't apply. ~6 s/512² image warm; first cold-load ~3–4 min (166 s weights + warmup). **A `/v1/chat/completions` request returns an *image*, not text** — only the diffusion stage is loaded, so it can't emit text. &nbsp; ⁴ **Vision resident** currently runs the vLLM `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` lane (NVFP4 + MTP n=2, `launch-vllm-qwen-fast.sh`). A llama.cpp GGUF variant (`launch-llamacpp-35b-moe-vision.sh`, UD-Q4_K_XL + mmproj, 69 tok/s) is the documented alternative in `llama-swap.full.bak` — swap the `cmd:` to switch. &nbsp; ⁵ **`nemotron-3-puzzle-75b`** is served at 131 k (native 262 k); it's a reasoning model — `--reasoning-parser nemotron_v3` splits `<think>` into `message.reasoning_content` (added 2026-07-08). Peak mem ~65 GB at the co-resident `NEMO_UTIL=0.55` split. &nbsp; ⁶ **`laguna-s-2.1`** serves the full native 256 k window solo (`util 0.85` → KV pool ~857 k tokens = 3.27× concurrency at full 256 k). Reasoning model — `--reasoning-parser poolside_v1` puts thinking in `reasoning_content`; give generous `max_tokens`. **`--max-num-seqs 4` is a hard ceiling with the DFlash drafter attached** — see [decision §34](decisions.md#34-laguna-concurrency-ceiling--seqs-8-deadlock-qwen-co-residency-parked-2026-07-24). &nbsp; tok/s / mem above are single-stream sweep figures; treat as ballpark.

> **Concurrency (2026-06-27 retune, applies to the copy-back lanes):** every model **except `int4-dflash` and `ds4`** is tuned to serve **c=1/2/3 at 131 k context** (`qwopus` is c=2 — its DFlash path forces heavier bf16 KV). `int4-dflash` stays single-stream (prefill-bound) and `ds4` stays single-stream (its decode doesn't scale with concurrency). See [decision §31](decisions.md#31-concurrency-retune--c123--131-k-for-the-swap-models-2026-06-27).

### What each model is for

- **`laguna-s-2.1`** 🟢 **(resident, coding default)** — poolside's **Laguna S-2.1** (118 B total / 8.5 B active MoE), NVFP4 weights + fp8 KV, with the matching **DFlash block-diffusion drafter** (`num_speculative_tokens=15`) — promoted to coding default 2026-07-22, moved to the re-uploaded "spinquantless" v2 weights 2026-07-23 (fixes the looping reports; revision-pinned in the launcher). 43–54 tok/s single-stream on code, full **native 256 k** context with a ~857 k-token KV pool (3.27× concurrency at max length). Serves `max-num-seqs 4` — a **hard ceiling** while the drafter is attached (seqs=8 deadlocks the engine under real traffic; decision §34). Reasoning model: thinking goes to `reasoning_content` via `--reasoning-parser poolside_v1`.
- **`nemotron-3-puzzle-75b`** **(parked — rollback for the default)** — NVIDIA's "Iterative Puzzle" compression of Nemotron-3-Super-120B down to **75.3 B total / 9.3 B active**, a hybrid **Mamba + MoE + Attention** stack (`NemotronHPuzzle`) with 256 k native context, NVFP4 weights, and MTP baked in — it fits one GB10 in ~50 GB of weights. Coding default 2026-07-08 → 2026-07-22, superseded by laguna. It wins at **long context** (beats the dense 27B past ~125 k) and **concurrency** (9× at 131 k), at the cost of lower single-stream speed than the Qwen lanes. Restore `config/llama-swap.yaml.bak.20260722-prelaguna` to bring back the 75B + vision resident pair.
- **`qwen3.6-27b-int4-dflash`** — Alibaba's **Qwen3.6-27B dense**, the previous coding default and now the **rollback target** for the aliases; on our own evals it beat the 35B-A3B MoE by ≥4 pts SWE-bench. Intel's AutoRound **INT4** keeps quality within noise of FP8 while halving the weight bandwidth that bottlenecks GB10, and the z-lab **DFlash** speculative drafter pushes it to ~41 tok/s. Reach for it (or roll the default back to it) for prefill-light everyday coding and agentic/tool-use work.
- **`qwen3.6-35b-a3b-nvfp4`** — Qwen's **sparse MoE** (35B total, ~3B active/token), so it sidesteps the bandwidth wall and stays cheap under load. RedHat's NVFP4 quant + native MTP give ~56 tok/s single-stream and the best concurrency/long-ctx throughput in the stack. Use it when you want speed and parallelism over the dense model's last few quality points.
- **`qwen3.6-35b-a3b-vision`** ⚫ **(dark since 2026-07-22)** — the former always-on **vision lane** for the 35B MoE; no headroom beside solo laguna, so requests to it currently fail. Previously ran unsloth's **`Qwen3.6-35B-A3B-NVFP4-Fast`** under vLLM (NVFP4 + MTP n=2, `qwen3_5_vision` encoder loaded, ~34 GB, 56 tok/s), so standard `image_url` content blocks work with vLLM's concurrency. A llama.cpp GGUF variant (`UD-Q4_K_XL` + `mmproj-F16`, 69 tok/s / ~27 GB — the fastest 35B c=1 measured) is the documented drop-in alternative; swap the launcher `cmd:` to switch. Use this lane to analyse images.
- **`qwen3.6-27b-nvfp4`** 🆕 — **NVIDIA's official ModelOpt NVFP4** build of the Qwen3.6-27B `qwen3_5` hybrid (48 Gated-DeltaNet + 16 full-attention layers, W4A16-NVFP4 MLPs). Its draw is **context**: `max_position_embeddings` is natively **262 144**, so it serves the **full 256 k window with no RoPE scaling and zero quality hit** — the longest-context coding lane in the stack. The hybrid layout means only 16/64 layers carry a growing KV cache, so 256 k is affordable on one GB10 (674 k-token KV pool). z-lab DFlash n=10 gives 31 tok/s single-stream. Needs the AEON `sm121a` engine + `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` — a **stock** vLLM image routes NVFP4 to a Marlin kernel that produces garbage on GB10/sm_121. Use it when you need a single request to span >130 k tokens.
- **`qwen3.6-27b-nvfp4-vision`** 🆕 — the `qwen3.6-27b-nvfp4` lane **with the `qwen3_5_vision` encoder loaded** (`--language-model-only` dropped, `--limit-mm-per-prompt image=4`): same weights, image, config, and full 256 k window, but `image_url` content blocks work. Perception is accurate on shapes/colour/spatial layout; OCR of stylised text is soft. Thinking model — the answer often lands in `reasoning_content`, so give generous `max_tokens`. Use it when a vision task needs the dense 27B's quality or >131 k context; for quick image work the 35B vision lane is faster.
- **`qwen3.6-27b-fp8`** — the **official Qwen FP8** build, "near-lossless" per the card with no community-quant noise. It's the ground-truth reference: when an INT4/NVFP4 result looks off, A/B against this. Use it for quality baselining and canonical Qwen3.6 behavior.
- **`qwopus3.6-27b-int4-dflash`** — Jackrong's **Qwopus**, a Qwen3.6-27B fine-tune **distilled on Claude-Opus reasoning traces**, so it keeps Qwen's coding base but reasons in a more Opus-like style. Same INT4+DFlash speed path as the default. Use it for coding/reasoning when you want Opus-flavored chains of thought.
- **`ornith-1.0-35b`** 🆕 — DeepReinforce's **Ornith-1.0** (MIT), a new agentic-coding MoE (35B/3B-active) that **writes its own RL training scaffold**; it scored **64.2 on Terminal-Bench 2.1, beating Qwen3.5-397B** (10× its size). Thinking model, and the fastest coding model here at 77 tok/s. The most interesting new model to pit against the Qwen incumbents on agentic/terminal tasks.
- **`diffusiongemma-26b`** — Google DeepMind's first **diffusion LLM**, which denoises 256-token blocks instead of decoding token-by-token, hitting ~142 tok/s (fastest in the stack). Google notes quality is below autoregressive Gemma 4, so it's a **speed lane, not a coding lane**. Use it for fast non-coding work — summaries, drafts, classification — where latency beats peak quality.
- **`nemotron-3-nano-omni`** — NVIDIA's **multimodal omni** model, a Mamba2-Transformer hybrid MoE with vision + audio encoders handling **text, image, audio, and video** in one model. It's the only multimodal option here — use it for anything the text-only coders can't see or hear.
- **`cosmos3-nano-omni`** 🆕 — NVIDIA's **Cosmos3-Nano** world model (~15B omni), the *generation* counterpart to the analysers above: it **produces** images and video from text/image prompts via vLLM-Omni. Unlike everything else in this table it is **not a chat model** — call `/v1/videos`, `/v1/videos/sync`, or `/v1/images/generations` (see footnote ³). The model *weights* are omni (text/audio capable), but only the diffusion generation stage is served here, so even a `/v1/chat/completions` call comes back as an **image, not text** — for text use the Qwen/Ornith/Nemotron lanes. The 64B Cosmos3-Super needs multi-GPU and stays standalone in `~/cosmos3`; Nano is the variant that fits one GB10. Use it for text→image/video generation.
- **`deepseek-v4-flash-ds4`** — DeepSeek's **V4-Flash**, whose multi-head-latent / compressed-KV attention scales to **256 k+ context cheaply**, run on the from-scratch antirez/ds4 C/CUDA engine with persistent disk-KV. Decode is slow (~21 tok/s) and doesn't parallelize, so it's a **planner, not a chat workhorse**. Use it for long-context planning/reasoning over whole codebases or long documents.

## Per-launcher details

> The first subsection is the **resident** (always loaded); the next few are the former residents and highest-traffic **copy-back** lanes. Lanes not detailed here (`qwopus`, `qwen3.6-27b-fp8`, `cosmos3-nano-omni`, `diffusiongemma-26b`, `ornith-1.0-35b`, `deepseek-v4-flash-ds4`) are covered in the [Decision log](decisions.md) (§29–§32) and their launcher scripts. Every copy-back lane pulls its weights from codeserver on first load (see [Weight offload](operations.md#weight-offload--codeserver-copy-back)); the configs below otherwise apply unchanged.

### Resident coding default: `laguna-s-2.1` (launcher: `bin/launch-vllm-laguna-s21-nvfp4-v2.sh`)

Coding default since 2026-07-22. **`poolside/Laguna-S-2.1-NVFP4`** (118 B total / 8.5 B active MoE, NVFP4 weights ~64 GB) + the matching **`poolside/Laguna-S-2.1-DFlash-NVFP4`** block-diffusion drafter at `num_speculative_tokens=15`. Port 9030, container `vllm-laguna-s21`, stock `vllm/vllm-openai:v0.25.1-aarch64-ubuntu2404` image (no AEON build needed — NVFP4 routes via `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`).

- **Weights are revision-pinned — target *and* drafter.** The v2 launcher pins the 2026-07-23 **"spinquantless norot" re-upload** (`0761412`) — poolside re-quantized without SpinQuant rotations as the apparent fix for the looping reports (HF discussions #4–#7, #10). The v1 launcher stays pinned to the original rotate weights (`b482b5d`) as the rollback: swap the `cmd:` in `llama-swap.yaml` line 25 to roll back. Since 2026-07-30 the **DFlash drafter is pinned too** (`4cdcc6e`, the rope_theta-matched re-upload) — before that, `speculative_config` resolved `main` on every start, and poolside's 07-27 drafter push reached prod silently on a routine bounce ([decision §35](decisions.md#35-vllm-0260-trial--reverted-drafter-revision-pinned-2026-07-30)). `LAG_DRAFT_REV` overrides the pin for A/Bs.
- **Looping status:** verbatim repetition-looping was smoke-verified gone on v2 (and DFlash decode got *faster* — 43–48 tok/s on code/math vs ~33 before). An **intermittent** runaway-reasoning mode (whole token budget burned inside `<think>`, near-empty visible content) still reproduces occasionally; it's an upstream open issue (HF #16) independent of the drafter, and the community's `repetition_penalty=1.15` mitigation measured *worse* here — see [decision §35](decisions.md#35-vllm-0260-trial--reverted-drafter-revision-pinned-2026-07-30).
- **Config (live)**: `util 0.85 / ctx 262144 / max-num-seqs 4 / fp8 KV / chunked prefill + prefix caching / max-num-batched-tokens 8192`. KV pool **856,686 tokens** = 3.27× concurrency at full 256 k. Env knobs: `LAG_UTIL / LAG_CTX / LAG_SEQS / LAG_SPEC(_N) / LAG_THINK` — headers in the launcher document each.
- **`--max-num-seqs 4` is a hard ceiling while the drafter is attached.** seqs=8 + DFlash n=15 **deadlocks the vLLM engine core under real traffic** (2× reproduced 2026-07-24; token counter freezes while `/health` stays 200, so llama-swap never recovers it). Drafterless seqs=8 is stable but ~19 tok/s single-stream — not worth it. Full findings in [decision §34](decisions.md#34-laguna-concurrency-ceiling--seqs-8-deadlock-qwen-co-residency-parked-2026-07-24).
- **`--reasoning-parser poolside_v1` + `--tool-call-parser poolside_v1`** — thinking goes to `reasoning_content`; give generous `max_tokens`. A long thinking answer at 20–26 tok/s legitimately takes 2–4 min — clients should not treat that as a hang.
- Drafter n=7 vs n=15 A/B: **resolved 2026-07-30, keep n=15.** The forum claim that draft positions 6–15 accept ~0 % does not hold on code traffic — measured per-position acceptance at 100 k decays smoothly 77 % → 2.5 %, and positions 7–14 supply 17.3 % of all accepted tokens. n=7 already lost the 07-22 A/B at 100 k (25.9 vs 33.3 tok/s). Details at `LAG_SPEC_N` in the launcher; [decision §35](decisions.md#35-vllm-0260-trial--reverted-drafter-revision-pinned-2026-07-30).

Measured: **54.2 tok/s** single-stream smoke (short), **43–48 tok/s** on code/math, ~33 tok/s @100 k. Cold start ~10–12 min. Deep notes: `memory project_laguna_s21_prod`, `project_laguna_v2_spinquantless`, `project_coresident_split_20260724`.

### Parked rollback: `nemotron-3-puzzle-75b` (launcher: `bin/launch-vllm-nemotron-puzzle-75b-mtp.sh`)

Coding default 2026-07-08 → 2026-07-22 (superseded by laguna; restore `llama-swap.yaml.bak.20260722-prelaguna` to reactivate with the vision pair). **`nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4`** — NVIDIA's "Iterative Puzzle" compression of Nemotron-3-Super-120B into a **75.3 B-total / 9.3 B-active** hybrid `NemotronHPuzzle` stack (interleaved Mamba-2 + MoE + full-attention layers), NVFP4 weights (~50 GB), 256 k native context, MTP head baked in. Held the `qwen3.6-27b` / `qwen3.6-35b-a3b` default aliases until 2026-07-22 (they now resolve to laguna at the gateway). Port 9027.

- **Image**: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (AEON sm_121a / GB10 build) — same image as the vision resident.
- **`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'` (MTP n=4) + CUDA graphs** (the `-mtp.sh` launcher). Both work on sm_121 via the AEON image: the vLLM #37431 Mamba "eager-only" tax is **not** binding here because graph capture is piecewise. MTP acceptance ~81 %.
- **Co-residence env** (set in `llama-swap.yaml`): `NEMO_UTIL=0.55` (~65 GB, leaves room for the vision lane) and `NEMO_SEQS=4`. Launcher solo defaults are `NEMO_UTIL=0.85 / NEMO_SEQS=8 / NEMO_CTX=131072`. (`llama-swap.full.bak` sets `0.60` — reconcile if you restore it.)
- **`--tool-call-parser qwen3_coder` + `--reasoning-parser nemotron_v3`** (parser added 2026-07-08) — splits `<think>…</think>` into `message.reasoning_content`, so clients read clean `content`. Still a reasoning model: give it generous `max_tokens`.

Measured (agg tok/s): short c1/c4 = **28 / 81**; 125 k c1/c4 = **20 / 12**. It **beats the dense 27B past ~125 k context** and scales ~9× under concurrency at 131 k, but is slower single-stream than the Qwen lanes and the 35B MoE (eval finding). An n=2 single-stream variant (~33 tok/s c=1) sits at `scratchpad/launch-mtp-n2.sh` if you want peak single-stream over concurrency. Deep writeup: `memory project_nemotron_puzzle_75b`.

### Dark vision lane: `qwen3.6-35b-a3b-vision` (launcher: `bin/launch-vllm-qwen-fast.sh`)

The former always-on **vision** lane (images), co-resident with the 75B until 2026-07-22 — **dark** while laguna holds the box solo. Ran **`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`** under vLLM (NVFP4 + MTP n=2, `qwen3_5_vision` encoder loaded), port 9026, `QWENFAST_UTIL=0.28` (~34 GB), `QWENFAST_CTX=262144`, `QWENFAST_SEQS=2`, `--reasoning-parser qwen3`, same AEON image as the 75B. Accepts standard `image_url` content blocks. 56 tok/s single-stream.

> **Alternate implementation.** `llama-swap.full.bak` defines this same route via **llama.cpp** instead — `bin/launch-llamacpp-35b-moe-vision.sh` serving unsloth's `UD-Q4_K_XL` GGUF + `mmproj-F16` (69 tok/s, ~27 GB, the fastest 35B c=1 measured). The two are interchangeable; swap the `cmd:` to switch engines. The llama.cpp variant's full write-up is preserved below under [MoE vision](#moe-vision-qwen36-35b-a3b-vision-launcher-binlaunch-llamacpp-35b-moe-visionsh).

### Dense: `qwen3.6-27b-int4-dflash` (launcher: `bin/launch-vllm-27b-int4-dflash.sh`)

Promoted 2026-05-17. **Intel/Qwen3.6-27B-int4-AutoRound** target + **z-lab/Qwen3.6-27B-DFlash** drafter at `num_speculative_tokens=4`. 29 tok/s single-stream — +80 % over the demoted Qwen official FP8 + native MTP n=3 prod (14.8 tok/s).

Why INT4 + DFlash (and why not 50 tok/s):
- The 27B is bandwidth-bound on Spark. INT4 weights (~14 GB) move half the bytes per forward vs FP8 (~28 GB) → that's the source of the +80 %.
- DFlash drafter was distilled against the BF16/NVFP4 base, not INT4. Per-position acceptance drops past pos 4 (0.74 / 0.50 / 0.31 / 0.22), so `n=4` ties `n=8` on throughput — anything higher is just wasted drafter compute.
- AEON-7's published 37.6 tok/s on Spark uses their proprietary abliterated NVFP4 weights + FlashInfer NVFP4 GEMM autotune — not portable to clean INT4.

**Critical config (load-bearing):**
- **`--attention-backend flash_attn`** — required by DFlash.
- **`--kv-cache-dtype auto` (bf16)** — FLASH_ATTN doesn't support fp8 KV in this image, so per-token KV is ~290 KB. This is the cost driver that caps c=1 at the co-resident config.
- **`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`** — load-bearing. FULL graph capture with INT4-GPTQ + DFlash drafter hangs for 30+ min on this image. PIECEWISE-only boots in ~6 min.
- **No `--quantization` flag** — vLLM auto-detects the auto-round packing as GPTQ-Marlin.
- **`--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":4}'`**.
- **Co-residence overrides 2026-05-17**: `--max-model-len 120000 --max-num-seqs 1 --gpu-memory-utilization 0.50`. KV pool 147 k tokens, max-concurrency 1.23× at 120 k. Solo-mode reverts to 200 k ctx / max-num-seqs 16 / gpu-mem-util 0.85.
- **Image**: AEON v4 (`ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4`) — bundles the interleaved-sliding-window-attention patch DFlash needs. Vanilla v0.20.1 doesn't have it. v4 over v3 was +7 % single-stream.

Cold start ~6 min warm-cache (torch.compile + FlashInfer caches hit on second boot).

### MoE: `qwen3.6-35b-a3b-nvfp4` (launcher: `bin/launch-vllm-35b-moe-nvfp4.sh`)

Re-added 2026-05-17 after Scargall's published Spark recipe (55.9 tok/s single-user / 433 tok/s c=32). **RedHatAI/Qwen3.6-35B-A3B-NVFP4** + native MTP n=1. ~17 GB weights, ~3 B active params per token — MoE bypasses the bandwidth wall the dense path hits.

Throughput: 56 tok/s c=1 / 101 c=2 / 169 c=4 (short input). Long context: 48 tok/s c=1 at 125 k input / 38 s TTFT. MTP n=1 accept consistently 80–87 %.

**Critical config:**
- **`--quantization compressed-tensors`** — required for Red Hat NVFP4 packaging.
- **`--moe-backend flashinfer_cutlass`** — sm_121a-verified MoE routing. The default backend mis-routes on Spark.
- **`--kv-cache-dtype fp8_e4m3`** — halves KV per token vs bf16; per-token KV ~145 KB.
- **`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`** — Qwen3.6 MTP is single-layer; `>1` re-runs the same layer with diminishing returns (Scargall's recipe documents this).
- **Co-residence overrides 2026-05-17**: `--max-model-len 80000 --max-num-seqs 2 --gpu-memory-utilization 0.40`. KV pool 574 k tokens, max-concurrency 23× at 80 k. Solo-mode reverts to 131 072 ctx / max-num-seqs 32 / gpu-mem-util 0.87.
- **Image**: `vllm/vllm-openai:cu130-nightly` — Scargall's recommended image. Required for Blackwell NVFP4 + MoE routing.

Known issue: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` had a silent correctness bug with `--enable-prefix-caching + compressed-tensors` on older vLLM ([vllm#40252](https://github.com/vllm-project/vllm/issues/40252)). cu130-nightly as of 2026-04 is reportedly patched — verify on real traffic if it goes live for coding workloads. **The 2026-05-11 MoE-vs-dense quality A/B is the open question** — MoE lost coding by ≥4 pts SWE-bench / -15.5 SkillsBench. Today's re-eval was throughput-only.

### MoE vision: `qwen3.6-35b-a3b-vision` (launcher: `bin/launch-llamacpp-35b-moe-vision.sh`)

Added 2026-07-07. **unsloth/Qwen3.6-35B-A3B-GGUF `UD-Q4_K_XL`** (dynamic 4-bit, 22.4 GB) **+ `mmproj-F16.gguf`** (0.9 GB vision encoder) at `~/models/qwen3.6-35b-a3b-gguf/`, served by the same mainline llama.cpp CUDA build as ornith (`~/llama.cpp/build/bin/llama-server`). Port 9026.

Why it exists: the vLLM `qwen3.6-35b-a3b-nvfp4` lane is text-only — llama.cpp's `--mmproj` path is what turns the 35B MoE's declared vision tower into working `image_url` support.

- **`--mmproj mmproj-F16.gguf`** — loads the multimodal projector (~1.1 GB worst-case per the mtmd estimate). Standard OpenAI `image_url` content blocks (data-URL base64 verified).
- **`--ctx-size 262144 --parallel 2`** — llama.cpp splits total ctx across slots → **c=2 @ 131 k per request** (`QWEN35B_CTX`/`QWEN35B_PARALLEL` env to retune). `context_window: 131072` in `deployed.yaml` — downstream harnesses must read that, not the engine total.
- **`-fa on -b 2048 -ub 512 --no-mmap --jinja --reasoning-budget -1`** — same recipe as the ornith lane.

Measured (2026-07-07, solo): **69.5 tok/s** text decode / **66.9 tok/s** on vision requests, image prefill ~670 tok/s — **faster single-stream than the vLLM NVFP4 lane (56)**, llama.cpp Q4_K_XL beats vLLM NVFP4 c=1 on GB10. Cold load ~4.5 min (22 GB weights + 262 k KV alloc). ~27 GB peak → second-lightest lane in the stack. Thinking model: pass `chat_template_kwargs: {"enable_thinking": false}` for terse answers. Quality caveat: the 2026-05-11 MoE-vs-dense coding rejection still applies — this is a **vision/speed lane, not a coding lane**.

### NVFP4 long-ctx: `qwen3.6-27b-nvfp4` (launcher: `bin/launch-vllm-27b-nvidia-nvfp4.sh`)

Added 2026-06-30; bumped to the full 256 k window 2026-07-01. **NVIDIA's official `nvidia/Qwen3.6-27B-NVFP4`** (ModelOpt, W4A16-NVFP4 MLPs + FP8 attention) on the `qwen3_5` hybrid architecture — 48 Gated-DeltaNet linear-attention layers + 16 full-attention layers (`full_attention_interval=4`), vision tower served text-only. Drafted by **z-lab/Qwen3.6-27B-DFlash** at `num_speculative_tokens=10`.

Why it exists: it's the **longest-context lane in the stack**. `max_position_embeddings` is natively **262 144**, so it serves the full **256 k window with no RoPE scaling and no quality penalty**.

- **The image is the whole story.** A stock vLLM image hardcodes the **Marlin** FP4 kernel for W4A16-NVFP4; Marlin's FP4 path is SM80-targeted and computes wrong logits on GB10/sm_121 → pure garbage output (`!!!!` / `d d d`), independent of KV dtype or graph mode. This is the documented DGX-Spark bug (vLLM [#37030](https://github.com/vllm-project/vllm/issues/37030), [#43906](https://github.com/vllm-project/vllm/issues/43906); fix [#38126](https://github.com/vllm-project/vllm/pull/38126)). The fix is `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (vLLM 0.23.0 built for sm_121a **with** CUTLASS NVFP4 kernels) + **`VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`**, which routes W4A16 to the CUTLASS path that *is* supported here. Weights are NVIDIA's, unmodified.
- **`--mamba-cache-dtype float32`** — the GDN recurrent-state cache must be fp32 for numerical stability (AEON recipe).
- **`--kv-cache-dtype bfloat16`** — mandatory for the DFlash drafter path; this checkpoint ships no fp8 KV scales, so fp8 would be lossy anyway. KV is cheap here (only 16/64 layers grow with context), so bf16 is free.
- **`--max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.85`** — KV pool **674 k tokens** = **2.57× max-concurrency at full 256 k**. c=4 runs fine up to ~150 k each; you cannot run 4 simultaneous requests all at full 256 k (that needs ~1.05 M tokens of KV), but real traffic never does.
- **`--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":10}'`** — DFlash n=10 measured ~2–3× faster single-stream than native MTP n=1.

Measured (2026-07-01, solo): decode **31 tok/s** short · 26.8 @118 k · **15.1 @254 k** (falls off attending over a longer KV); c=2 56 / c=4 103 tok/s aggregate (short). Cold prefill ~144 s @118 k, **~438 s (~7 min) @254 k** — a 256 k cold prefill is slow; warm decode is unaffected by the cap.

**Vision twin** (`qwen3.6-27b-nvfp4-vision`, `bin/launch-vllm-27b-nvidia-nvfp4-vision.sh`, port 9025, added 2026-07-05): identical weights/image/config, but `--language-model-only` is dropped and `--limit-mm-per-prompt '{"image":4,"video":0}'` added, so the `qwen3_5_vision` encoder loads and `image_url` blocks work at the full 256 k window. Verified on GB10: shape/colour/spatial perception accurate; stylised-text OCR soft.

### Co-residence contention behavior (single GPU caveat)

> Measured on the *retired* dense-27B + 35B-MoE co-resident pair, but the lesson applies unchanged to any future co-resident pair — any two engines sharing GB10 contend for SM cycles. (Since 2026-07-22 laguna runs solo, so this is currently moot — and note the harder 2026-07-24 finding in §34: next to a fully-resident laguna, a second vLLM engine can't even *load*.)

| Scenario | Dense | MoE |
|---|---:|---:|
| Dense fires alone (MoE resident-idle) | 31.2 tok/s | (idle) |
| MoE fires alone (dense resident-idle) | (idle) | 101.3 tok/s |
| Both decode simultaneously | 17.3 | 39.4 |

Memory is partitioned cleanly. Compute isn't — GB10 is one physical GPU, and two engines competing for SM cycles halves each one's throughput. Co-residence wins for **bursty / sequential** mixed traffic (one busy at a time); it loses for **sustained concurrent** dual-engine load. For the current pair this is by design: the vision lane answers short handwriting-OCR turns while the 75B does occasional deep reasoning — rarely both flat-out at once.

## Adding a model

1. **Check the cache first:** `ls ~/.cache/huggingface/hub/ | grep -i <name>`.
2. **Download if missing:** `~/llm-stack/venv/bin/hf download <org>/<repo>`. Requires `max:max` ownership on `~/.cache/huggingface/hub`.
3. **Add a `models:` block** in `config/llama-swap.yaml`:
   - safetensors (BF16, FP8, NVFP4, INT4-AutoRound): use a vLLM launcher pattern (see `bin/launch-vllm-27b-int4-dflash.sh` as template).
   - GGUF: use a llama.cpp launcher pattern (see `bin/launch-vllm-qwen.sh` historical pattern).
4. **Pick a group** (`config/llama-swap.full.bak`): add the key to `experiments` (swap-exclusive, one-at-a-time) for an eval/rollback model, or to `resident` (persistent, co-resident — mind the GPU-memory split) for an always-on model.
5. **(Optional) offload the weights** to codeserver so they don't sit on local NVMe. Move them to `codeserver:~/llm-weights-archive/<area>/`, add the local path to `etc/copyback-models.txt`, and wrap the slot's `cmd:` with `bin/copyback-launch.sh <local_path> <remote_relpath> <real_launcher>`. Do **not** offload a `resident` model. See [Weight offload](operations.md#weight-offload--codeserver-copy-back).
6. **Validate**: `python3 -c "import yaml; yaml.safe_load(open('config/llama-swap.yaml'))"`.
7. **Save** — `-watch-config` reloads within ~1 s (see decision log §27). Verify via `curl -s http://localhost:8080/v1/models`.
8. **Smoke test** by sending a 5-token completion — first request is the cold load (plus the copy-back pull, if offloaded).

Fallback for binary / unit changes: `pkill -9 llama-swap` (SIGKILL → systemd respawn; SIGTERM does not — `Restart=on-failure`).

## MTP patch (AlphaOxO Qwen3.6-27B-NVFP4 only)

> Applies **only** to the `qwen3.6-27b-mtp` rollback slot. The production INT4+DFlash path has no MTP. The FP8 rollback and sakamaki rollback both ship MTP weights natively / grafted and need no patch. Apply this only if you reactivate the AlphaOxO launcher.

The AlphaOxO NVFP4 repo ships `model_mtp.safetensors` but strips `num_nextn_predict_layers` from `config.json`. Without that field, vLLM silently loads with MTP disabled — `--speculative-config` becomes a no-op and you lose 1.6–2× decode throughput.

```bash
python3 <<'PYEOF'
import json, os
SNAP = os.path.expanduser('~/.cache/huggingface/hub/models--AlphaOxO--Qwen3.6-27B-NVFP4/snapshots')
snap = os.path.join(SNAP, os.listdir(SNAP)[0])
p = os.path.join(snap, 'config.json')
c = json.load(open(os.path.realpath(p)))
if c.get('num_nextn_predict_layers') == 1:
    print('already patched'); raise SystemExit
c['num_nextn_predict_layers'] = 1
if os.path.islink(p):
    os.unlink(p)          # break the symlink so we don't corrupt HF blob hashes
json.dump(c, open(p, 'w'), indent=2)
print('patched: num_nextn_predict_layers=1')
PYEOF
```

`hf download --force` will revert the patch — re-apply if weights are re-fetched. Verify by grepping vLLM startup logs for `Detected MTP model. Sharing target model embedding weights with the draft model.` — present = wired; absent = silently off.
