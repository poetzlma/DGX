# Spark LLM Stack

OpenAI-compatible LLM gateway on one **DGX Spark (GB10, 119 GB unified)**. Call it at **`http://192.168.1.12:8079/v1`** and route by the `model` field.

The stack runs in **two tiers**:

- **Resident tier** — since 2026-07-22 a **single** always-loaded model: **`laguna-s-2.1`** (poolside Laguna S-2.1, 118B-A8.5B MoE, NVFP4 + DFlash speculative decoding — the coding default; the legacy `qwen3.6-27b` / `qwen3.6-35b-a3b` / `nemotron-3-puzzle-75b` names all resolve here at the gateway). It holds the box solo at `util 0.85 / ctx 262144 / max-num-seqs 4`; the former nemotron-75B + vision resident pair is the documented rollback (`config/llama-swap.yaml.bak.20260722-prelaguna`), and the vision lane is **dark** until topology changes.
- **Dormant tier** — ~11 eval/rollback models. Their weights live **off-box on codeserver** (`192.168.1.16`) to keep the Spark's NVMe from filling; requesting one **rsyncs its weights back on demand** (1–3 min for most, ~13 min for ds4's 85 GB GGUF), then serves it swap-exclusively. See [Weight offload](#weight-offload--codeserver-copy-back). One dormant model is loaded at a time. (The live config is in LOCKED MODE — laguna only — so dormant slots need the full roster restored first; see [Groups](#groups-resident--experiments).)

> **Request path:** client → **LiteLLM** (authed edge, `192.168.1.7`) → **log-proxy** (`192.168.1.12:8079`, logs every request) → **llama-swap** (`:8080`, routes by `model`, hot-swaps backends) → **engine** (`:90xx`). On the LAN you can skip LiteLLM and hit `:8079` directly.

## Models

Pick by `Route` (the `model` value). The **Tier** column says whether the model is always loaded (`resident`) or pulled from codeserver on demand (`copy-back`, +1–3 min first-load / ~13 min for ds4 — see [Weight offload](#weight-offload--codeserver-copy-back)). Speed/mem are single-stream figures from prior sweeps (`bin/bench-models.py`; raw in `logs/`). **Settings for each model live in its launcher** (`bin/<launcher>`) and are explained in [Per-launcher details](#per-launcher-details).

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

¹ llama.cpp lanes split total engine ctx across parallel slots, env-tunable: Ornith `ORNITH_CTX`/`ORNITH_PARALLEL` (3 slots × 131 k), 35B-vision `QWEN35B_CTX`/`QWEN35B_PARALLEL` (2 slots × 131 k). &nbsp; ² **Thinking model** — output goes to `reasoning_content`; give generous `max_tokens` or `content` returns empty. &nbsp; ³ **Generation model, not chat** — call `POST /v1/videos` · `/v1/videos/sync` · `/v1/images/generations` (multipart), not `/v1/chat/completions`; tok/s and token-ctx don't apply. ~6 s/512² image warm; first cold-load ~3–4 min (166 s weights + warmup). **A `/v1/chat/completions` request returns an *image*, not text** — only the diffusion stage is loaded, so it can't emit text. &nbsp; ⁴ **Vision resident** currently runs the vLLM `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` lane (NVFP4 + MTP n=2, `launch-vllm-qwen-fast.sh`). A llama.cpp GGUF variant (`launch-llamacpp-35b-moe-vision.sh`, UD-Q4_K_XL + mmproj, 69 tok/s) is the documented alternative in `llama-swap.full.bak` — swap the `cmd:` to switch. &nbsp; ⁵ **`nemotron-3-puzzle-75b`** is served at 131 k (native 262 k); it's a reasoning model — `--reasoning-parser nemotron_v3` splits `<think>` into `message.reasoning_content` (added 2026-07-08). Peak mem ~65 GB at the co-resident `NEMO_UTIL=0.55` split. &nbsp; ⁶ **`laguna-s-2.1`** serves the full native 256 k window solo (`util 0.85` → KV pool ~857 k tokens = 3.27× concurrency at full 256 k). Reasoning model — `--reasoning-parser poolside_v1` puts thinking in `reasoning_content`; give generous `max_tokens`. **`--max-num-seqs 4` is a hard ceiling with the DFlash drafter attached** — see [decision §34](#34-laguna-concurrency-ceiling--seqs-8-deadlock-qwen-co-residency-parked-2026-07-24). &nbsp; tok/s / mem above are single-stream sweep figures; treat as ballpark.

> **Concurrency (2026-06-27 retune, applies to the copy-back lanes):** every model **except `int4-dflash` and `ds4`** is tuned to serve **c=1/2/3 at 131 k context** (`qwopus` is c=2 — its DFlash path forces heavier bf16 KV). `int4-dflash` stays single-stream (prefill-bound) and `ds4` stays single-stream (its decode doesn't scale with concurrency). See [decision §31](#31-concurrency-retune--c123--131-k-for-the-swap-models-2026-06-27).

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

### Call it

```sh
curl http://192.168.1.12:8079/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "hi"}]
}'
# "qwen3.6-27b" / "qwen3.6-35b-a3b" / "nemotron-3-puzzle-75b" are gateway aliases → the resident laguna-s-2.1 default.
# Resident models answer immediately. A copy-back (dormant) model first rsyncs its weights
# from codeserver, so set client timeout ≥ 1200s (matches llama-swap healthCheckTimeout).
# :8079 is the logged path (log-proxy → llama-swap); :8080 is the direct gateway.
```

### Quick commands

(full runbook — logs, rollback, copy-back inspection — under [Operations](#operations).)

```sh
curl -s http://192.168.1.12:8080/v1/models | jq -r '.data[].id'   # list routes
curl -s http://192.168.1.12:8080/running                          # which model is hot
sudo systemctl status llama-swap                                  # service health
curl -X POST http://192.168.1.12:8080/unload                      # unload the current dormant model (residents unaffected)
docker ps --filter name=vllm- --filter name=llama-                # running engine container
```

Re-run the full characterisation sweep (speed + cold-load + peak mem for every model) with `python3 bin/bench-models.py`.

---

## Hardware

- **NVIDIA DGX Spark (GB10)** — Grace/Blackwell SoC, compute capability 12.1 (SM 12.1 — GB10 chiplet, often mis-reported as SM120 generic Blackwell).
- **119 GB unified memory.** No separate VRAM — CPU and GPU read/write the same physical DRAM over NVLink-C2C. `--gpu-memory-utilization 0.85` ≈ 101 GB.
- **FP4 / FP8 hardware support** — native matmul kernels for both. We lean on these (production dense is INT4-Marlin, MoE is NVFP4-CUTLASS).
- **NVMe** — single nvme0n1, multi-GB/s sequential reads. Default llama.cpp mmap paths cap ~200 MB/s; `--no-mmap` recovers throughput (see [decision log §6](#decision-log)).

---

## Gateway

| Field | Value |
|---|---|
| Public URL | `http://192.168.1.12:8079/v1` (log-proxy → llama-swap) |
| Direct (no log-proxy) | `http://192.168.1.12:8080/v1` |
| Gateway software | [llama-swap](https://github.com/mostlygeek/llama-swap) (`~/bin/llama-swap`) |
| Auth | none (trusted LAN) |
| Model list | `GET /v1/models` |
| Currently hot | `GET /running` |
| Web UI | `http://192.168.1.12:8080/ui` |
| Hot config reload | `-watch-config` enabled — yaml edits apply in ~1 s, no restart |

### Groups: `resident` + `experiments`

Two llama-swap groups (`config/llama-swap.full.bak`):

- **`resident`** (`swap: false, exclusive: false, persistent: true`) — since 2026-07-22 a **single** always-on model, `laguna-s-2.1`, holding the box at `util 0.85` (~101 GB). `ttl: 0` — no idle unload, plus an `on_startup` preload hook so it loads at service start. (The previous era's two-member pair — `nemotron-3-puzzle-75b` + `qwen3.6-35b-a3b-vision` split `0.55`/`0.28` — is preserved in `llama-swap.yaml.bak.20260722-prelaguna` and `llama-swap.full.bak`.)
- **`experiments`** (`swap: true, exclusive: true`) — the dormant pool. **At most one** member loads at a time; requesting a different one unloads the current and cold-loads the next. Each is a [copy-back model](#weight-offload--codeserver-copy-back): its weights rsync from codeserver first (+1–3 min, ~13 min for ds4), then the engine cold-loads (1–10 min vLLM, ~30–105 s for llama.cpp/ds4 binaries). `ttl: 3600`.

> **⚠️ Swap-exclusive vs. itself, not vs. residents.** The `experiments` group only unloads *other experiments members* — it never evicts the resident. Solo laguna at `util 0.85` (~101 GB) leaves **no headroom for any co-loaded model**: even the 23 GB qwen 35B fast lane trips the host-OOM floor during its vLLM load transient (decision §34), and **the GB10 hard-hangs on host OOM** with no remote recovery. Free the resident first (`docker rm -f vllm-laguna-s21`) or re-shrink it (`LAG_UTIL=0.66 LAG_CTX=131072`) before loading anything beside it. See [Troubleshooting](#troubleshooting).

> **⚠️ Active config is in LOCKED MODE.** `config/llama-swap.yaml` (the live config) currently exposes **only `laguna-s-2.1`** — the dormant pool is not loadable from it. The full roster lives in `config/llama-swap.full.bak`; restore it (`cp config/llama-swap.full.bak config/llama-swap.yaml`, `-watch-config` picks it up) to re-enable the copy-back models. **Editing the live yaml bounces the resident** (~10–12 min reload) — the preload hook + cron watchdog auto-recover, but treat any yaml write as a production restart.

---

## Weight offload — codeserver copy-back

The Spark's 916 GB NVMe was filling (95%). Dormant model weights now live **off-box on codeserver** (`192.168.1.16`, a 1.9 TB LAN host reached over 2.5 GbE) under `~/llm-weights-archive/` (~409 GB), and are **pulled back on demand** the moment llama-swap starts that slot — dropping the Spark to **~48% used**. Only the resident's weights (laguna + its DFlash drafter) plus the keep-local set stay permanently on NVMe.

**How a dormant model loads.** Each `experiments` slot's `cmd:` in `llama-swap.full.bak` is wrapped by [`bin/copyback-launch.sh`](bin/copyback-launch.sh):

```
copyback-launch.sh <local_path> <remote_relpath> <real_launch_script>
```

On start it (1) **evicts every other managed model** listed in `etc/copyback-models.txt` — enforcing *one dormant model on NVMe at a time* and self-healing if a prior slot was SIGKILL'd before it could clean up; (2) **rsyncs** the weights from `codeserver:~/llm-weights-archive/<remote_relpath>` if they aren't already local (`--partial` resumes an interrupted pull; a failed pull is removed, not left corrupt); (3) runs the real launcher with `TERM`/`INT` forwarded; (4) on stop, **evicts** the weights again (evict-immediately-after-use — leanest disk, re-pulls each cold start).

**The manifest** `etc/copyback-models.txt` is the eviction guard: one absolute weight path per line. **Keep-local models are deliberately absent** so they can never be evicted — the `laguna-s-2.1` weights + its DFlash drafter, `nemotron-3-puzzle-75b`, the `qwen3.6-35b-a3b-vision` weights, and the z-lab DFlash drafter.

**Archive layout** (`codeserver:~/llm-weights-archive/`):

```
hub/models--<org>--<name>   # HF-cache models → restore to ~/.cache/huggingface/hub/
qwopus/Qwopus3.6-27B-v2-int4-AutoRound
ornith/ornith-1.0-35b
ds4/gguf                    # ~85 GB — the longest pull (~13 min)
cosmos3-models/Cosmos3-Nano
```

**Operational notes:**

- `healthCheckTimeout: 1200` (20 min) in `llama-swap.yaml` **must exceed** the pull+load time, or llama-swap kills the slot mid-download.
- **New dependency:** dormant models require **codeserver online** to load — a failed pull exits non-zero and the slot won't start. The resident has no such dependency. Weights exist *only* on codeserver now (local copies were deleted after verification).
- The pull path uses SSH key auth to a bare host (no credentials in-repo). The codeserver address defaults to the LAN IP but is overridable via the **`COPYBACK_REMOTE`** env var (and `COPYBACK_ARCHIVE_ROOT` for the archive path) in `bin/copyback-launch.sh` — set it in the systemd env or launcher if you'd rather not hardcode internal topology.

---

## Per-launcher details

> The first subsection is the **resident** (always loaded); the next few are the former residents and highest-traffic **copy-back** lanes. Lanes not detailed here (`qwopus`, `qwen3.6-27b-fp8`, `cosmos3-nano-omni`, `diffusiongemma-26b`, `ornith-1.0-35b`, `deepseek-v4-flash-ds4`) are covered in the [Decision log](#decision-log) (§29–§32) and their launcher scripts. Every copy-back lane pulls its weights from codeserver on first load (see [Weight offload](#weight-offload--codeserver-copy-back)); the configs below otherwise apply unchanged.

### Resident coding default: `laguna-s-2.1` (launcher: `bin/launch-vllm-laguna-s21-nvfp4-v2.sh`)

Coding default since 2026-07-22. **`poolside/Laguna-S-2.1-NVFP4`** (118 B total / 8.5 B active MoE, NVFP4 weights ~64 GB) + the matching **`poolside/Laguna-S-2.1-DFlash-NVFP4`** block-diffusion drafter at `num_speculative_tokens=15`. Port 9030, container `vllm-laguna-s21`, stock `vllm/vllm-openai:v0.25.1-aarch64-ubuntu2404` image (no AEON build needed — NVFP4 routes via `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`).

- **Weights are revision-pinned.** The v2 launcher pins the 2026-07-23 **"spinquantless norot" re-upload** (`0761412`) — poolside re-quantized without SpinQuant rotations as the apparent fix for the looping reports (HF discussions #4–#7, #10). The v1 launcher stays pinned to the original rotate weights (`b482b5d`) as the rollback: swap the `cmd:` in `llama-swap.yaml` line 25 to roll back. Looping was smoke-verified gone on v2, and DFlash decode got *faster* (43–48 tok/s on code/math vs ~33 before).
- **Config (live)**: `util 0.85 / ctx 262144 / max-num-seqs 4 / fp8 KV / chunked prefill + prefix caching / max-num-batched-tokens 8192`. KV pool **856,686 tokens** = 3.27× concurrency at full 256 k. Env knobs: `LAG_UTIL / LAG_CTX / LAG_SEQS / LAG_SPEC(_N) / LAG_THINK` — headers in the launcher document each.
- **`--max-num-seqs 4` is a hard ceiling while the drafter is attached.** seqs=8 + DFlash n=15 **deadlocks the vLLM engine core under real traffic** (2× reproduced 2026-07-24; token counter freezes while `/health` stays 200, so llama-swap never recovers it). Drafterless seqs=8 is stable but ~19 tok/s single-stream — not worth it. Full findings in [decision §34](#34-laguna-concurrency-ceiling--seqs-8-deadlock-qwen-co-residency-parked-2026-07-24).
- **`--reasoning-parser poolside_v1` + `--tool-call-parser poolside_v1`** — thinking goes to `reasoning_content`; give generous `max_tokens`. A long thinking answer at 20–26 tok/s legitimately takes 2–4 min — clients should not treat that as a hang.
- Pending A/B (documented at `LAG_SPEC_N` in the launcher): drafter n=7 vs n=15 — forum-measured acceptance at draft positions 6–15 is ~0 % on natural text, but verify is ~free on bandwidth-bound GB10 and acceptance is higher on code. Costs one ~12-min bounce to test.

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

---

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Gateway logs
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Resident container logs (always loaded)
docker logs -f vllm-laguna-s21             # laguna-s-2.1 (coding default)
# NOTE: llama-swap wipes the container on relaunch — to capture a crash loop,
# stream to a file while it happens: docker logs -f vllm-laguna-s21 > /tmp/lag.log 2>&1 &

# Dormant (copy-back) engines — a container exists only while that model is loaded.
# Names follow each launcher's --name (vllm-qwen-27b-int4-dflash, vllm-qwen-27b-fp8,
# vllm-nemotron-omni, ds4-server, …). List whatever is live with `docker ps`.

# State inspection
curl http://192.168.1.12:8080/running
curl http://192.168.1.12:8080/v1/models
docker ps --filter name=vllm- --filter name=llama- --filter name=ds4-

# Force-unload the current experiments-group model (residents are unaffected)
curl -X POST http://192.168.1.12:8080/unload

# Copy-back inspection: archive contents + which dormant weights are staged locally now
ssh 192.168.1.16 'du -sh ~/llm-weights-archive/*'
cat etc/copyback-models.txt   # the managed paths (at most one exists locally at a time)
```

### Roll the coding default off laguna

```sh
# LAGUNA WEIGHTS ROLLBACK (looping or quality regression on the v2 weights):
#   edit config/llama-swap.yaml line 25 — point cmd: back to
#   bin/launch-vllm-laguna-s21-nvfp4.sh (v1, pinned to the original b482b5d
#   rotate weights). The yaml edit IS the cutover (live-watched, ~10 min bounce).
#
# FULL MODEL ROLLBACK (leave laguna entirely):
#   restore config/llama-swap.yaml.bak.20260722-prelaguna to bring back the
#   nemotron-3-puzzle-75b + vision resident pair, then repoint the gateway names.
#
# qwen3.6-27b / qwen3.6-35b-a3b / nemotron-3-puzzle-75b currently resolve to
# laguna-s-2.1. To fall back further to the dense Qwen 27B, repoint the aliases
# to qwen3.6-27b-int4-dflash:
#     aliases:
#       - qwen3.6-27b
#       - qwen3.6-35b-a3b
#
# WHERE to change it: llama-swap v201 drops yaml aliases on -watch-config reload,
# so the authoritative alias→model map lives at the LiteLLM gateway (deployed.yaml
# on cockroach / 192.168.1.7), NOT only in config/llama-swap.yaml. Update it there.
#
# int4-dflash is a copy-back model — its first load after rollback pulls ~18 GB from
# codeserver (~2 min) before it serves. Pre-stage by hitting it once to warm the pull.
```

---

## Benchmarking

| Script | Purpose | Usage |
|---|---|---|
| `bin/bench-parallel.py` | Single-engine concurrent decode, fixes reasoning-parser stream counting | `BENCH_URL=http://.../v1/chat/completions ./bench-parallel.py MODEL CONCURRENCY PROMPT_TOKENS MAX_TOKENS` |
| `bin/bench-mixed.py` | Multi-engine parallel (fires N to dense + M to MoE simultaneously) | `./bench-mixed.py` |
| `bin/bench-bigctx-concurrency.py` | Long-context concurrency sweep, honors `BENCH_OUT_TOKENS` env | `BENCH_OUT_TOKENS=1024 ./bench-bigctx-concurrency.py MODEL TARGET_INPUT LEVELS…` |
| `bin/bench-coding-realistic.py` | 13 k / 60 k / 100 k real-traffic-shape A/B with spec-decode acceptance scrape | `./bench-coding-realistic.py MODEL CONTAINER [LABEL]` |
| `bin/bench-deep.py` | TTFT, decode tok/s, multi-tier, concurrency 3/5, 3-pass mean±stdev | `python3 bin/bench-deep.py [model…]` |
| `bin/bench-models.py` | Quick cold/warm + tok/s sweep across all models | `python3 bin/bench-models.py` |
| `bin/bench-concurrency-sweep.py` | Ad-hoc concurrency sweep on a warm model | inline |
| `bin/bench-context-sweep.py` | Context-length sweep (TTFT, prefill, decode by ctx) | inline |

Results go to `logs/bench-*.json` and `logs/bench-*.log` with timestamped names. `logs/bench-deep-latest.json` is the symlink to the most recent deep run.

---

## Adding a model

1. **Check the cache first:** `ls ~/.cache/huggingface/hub/ | grep -i <name>`.
2. **Download if missing:** `~/llm-stack/venv/bin/hf download <org>/<repo>`. Requires `max:max` ownership on `~/.cache/huggingface/hub`.
3. **Add a `models:` block** in `config/llama-swap.yaml`:
   - safetensors (BF16, FP8, NVFP4, INT4-AutoRound): use a vLLM launcher pattern (see `bin/launch-vllm-27b-int4-dflash.sh` as template).
   - GGUF: use a llama.cpp launcher pattern (see `bin/launch-vllm-qwen.sh` historical pattern).
4. **Pick a group** (`config/llama-swap.full.bak`): add the key to `experiments` (swap-exclusive, one-at-a-time) for an eval/rollback model, or to `resident` (persistent, co-resident — mind the GPU-memory split) for an always-on model.
5. **(Optional) offload the weights** to codeserver so they don't sit on local NVMe. Move them to `codeserver:~/llm-weights-archive/<area>/`, add the local path to `etc/copyback-models.txt`, and wrap the slot's `cmd:` with `bin/copyback-launch.sh <local_path> <remote_relpath> <real_launcher>`. Do **not** offload a `resident` model. See [Weight offload](#weight-offload--codeserver-copy-back).
6. **Validate**: `python3 -c "import yaml; yaml.safe_load(open('config/llama-swap.yaml'))"`.
7. **Save** — `-watch-config` reloads within ~1 s (see decision log §27). Verify via `curl -s http://localhost:8080/v1/models`.
8. **Smoke test** by sending a 5-token completion — first request is the cold load (plus the copy-back pull, if offloaded).

Fallback for binary / unit changes: `pkill -9 llama-swap` (SIGKILL → systemd respawn; SIGTERM does not — `Restart=on-failure`).

---

## LiteLLM integration

Points at the log-proxy on `:8079` so every request lands in `~/llm-stack/logs/proxy/{date}/{model}/`. `deployed.yaml` in this repo is a drop-in `model_list` for LiteLLM. Minimum per-entry:

```yaml
model_list:
  - model_name: qwen3.6-27b                      # what clients call (legacy alias)
    litellm_params:
      model: openai/nemotron-3-puzzle-75b        # resolve the alias to the real key HERE
      api_base: http://192.168.1.12:8079/v1
      api_key: none
      timeout: 1200                              # ≥ cold-load + copy-back pull
litellm_settings:
  request_timeout: 1200
```

The `openai/<key>` string in `litellm_params.model` must exactly match the key under `models:` in the active llama-swap config — that's how llama-swap routes and decides which backend to swap in. **Resolve aliases at this gateway layer** (map `qwen3.6-27b` → the real model name): llama-swap v201 drops yaml `aliases:` on `-watch-config` reload, so the LiteLLM `model_list` is the reliable place to pin them.

---

## Files

```
~/llm-stack/
├── config/
│   ├── llama-swap.yaml             # ACTIVE config — LOCKED MODE (2 residents only)
│   └── llama-swap.full.bak         # full roster (resident + experiments groups) — restore to unlock
├── deployed.yaml                   # LiteLLM model_list for downstream (two-tier framing, updated 2026-07-13)
├── bin/
│   ├── copyback-launch.sh              # copy-back/dormant-tier wrapper: rsync weights from codeserver, run, evict
│   ├── launch-vllm-laguna-s21-nvfp4-v2.sh # RESIDENT — coding default (Laguna S-2.1 v2 spinquantless + DFlash n=15)
│   ├── launch-vllm-laguna-s21-nvfp4.sh    # rollback — Laguna v1 (original rotate weights, pinned b482b5d)
│   ├── launch-vllm-35b-moe-nvfp4-colag.sh # PARKED — qwen 35B fast lane sized to co-reside with laguna (see §34)
│   ├── smoke-laguna-v2.py                 # laguna smoke test (looping check + tok/s)
│   ├── launch-vllm-nemotron-puzzle-75b-mtp.sh # parked rollback — coding default 07-08→07-22 (NVFP4 75B hybrid, MTP n=4)
│   ├── launch-vllm-qwen-fast.sh        # dark — 35B vision (unsloth NVFP4-Fast, MTP n=2, vLLM)
│   ├── launch-vllm-27b-int4-dflash.sh   # copy-back — rollback default (Intel INT4 + DFlash n=4)
│   ├── launch-vllm-35b-moe-nvfp4.sh     # copy-back — 35B MoE throughput (RedHatAI NVFP4 + MTP n=1)
│   ├── launch-llamacpp-35b-moe-vision.sh # alt vision impl (unsloth GGUF + mmproj, llama.cpp)
│   ├── launch-vllm-27b-nvidia-nvfp4.sh  # copy-back — NVIDIA NVFP4 256k lane (DFlash n=10)
│   ├── launch-vllm-27b-nvidia-nvfp4-vision.sh # copy-back — 256k lane VISION twin (encoder loaded)
│   ├── launch-vllm-27b-qwen-fp8.sh      # rollback — was prod 2026-05-08 → 2026-05-17
│   ├── launch-vllm-qwopus-int4-dflash.sh # eval — Opus-distilled 27B (INT4 + DFlash)
│   ├── launch-ornith.sh                 # eval — Ornith-1.0-35B MoE (llama.cpp)
│   ├── launch-vllm-diffusiongemma-nvfp4.sh # eval — Google diffusion LLM
│   ├── launch-vllm-cosmos3-nano-omni.sh # eval — Cosmos3-Nano image/video generation
│   ├── launch-vllm-27b-int4-autoround.sh # retired — Intel INT4 + native MTP (no DFlash)
│   ├── launch-vllm-27b-sakamaki-mtp.sh  # retired — NVFP4 + MTP graft
│   ├── launch-vllm-27b-dflash.sh        # retired — aeon NVFP4 + DFlash k=4
│   ├── launch-vllm-27b-nvfp4.sh         # retired — AlphaOxO NVFP4 + MTP-3
│   ├── launch-vllm-27b-clean-dflash.sh  # retired — eval, non-prod
│   ├── launch-vllm-qwen.sh              # retired — real 35B-A3B MoE (alias-only)
│   ├── launch-vllm-nemotron-omni.sh     # active — multimodal omni
│   ├── launch-ds4-server.sh             # active — antirez ds4 planner lane
│   ├── bench-parallel.py                # single-engine concurrent decode bench
│   ├── bench-mixed.py                   # multi-engine parallel bench
│   ├── bench-bigctx-concurrency.py      # long-context concurrency sweep (BENCH_OUT_TOKENS env)
│   ├── bench-coding-realistic.py        # 13k/60k/100k real-traffic A/B
│   ├── bench-deep.py                    # deep bench (TTFT, decode, concurrency)
│   ├── bench-models.py                  # quick cold/warm sweep
│   ├── bench-concurrency-sweep.py       # ad-hoc concurrency sweep
│   ├── bench-context-sweep.py           # context-length sweep
│   ├── bench-multimodal-{smoke,large}.py # multimodal benches
│   ├── bench-compare-deep.py            # before/after comparison
│   ├── log-proxy.py                     # systemd-managed; on :8079 in-path of LiteLLM → llama-swap
│   ├── stack-{api,status,tui}           # cockpit / metrics
│   └── traffic-tui                      # rich TUI for proxy traffic
├── systemd/
│   ├── llama-swap.service               # main unit (ExecStartPre/Post cleanup hooks)
│   ├── llama-swap.service.d/
│   │   └── watch-config.conf            # drop-in: enables -watch-config
│   ├── log-proxy.service                # systemd-managed log-proxy
│   └── stack-api.service                # systemd-managed stack-api
├── etc/
│   ├── qwen3.6-chat-template-froggeric.jinja  # chat template used by Qwen3.6 entries
│   └── copyback-models.txt                    # copy-back eviction manifest (dormant weight paths)
├── docs/
│   ├── qwen3.6-27b-dflash.md            # deep DFlash writeup
│   └── deepseek-v4-flash.md             # ds4 deep writeup
├── venv/                                # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.{log,err}             # gateway std{out,err}
│   ├── proxy/{date}/{model}/            # per-request triples from log-proxy
│   ├── bench-deep-latest.json           # symlink to latest deep bench
│   └── bench-*.json                     # timestamped runs
└── README.md

~/bin/llama-swap                                            # gateway binary
/etc/systemd/system/llama-swap.service                      # main unit
/etc/systemd/system/llama-swap.service.d/watch-config.conf  # hot-reload drop-in
~/.cache/huggingface/hub/                                   # model weights
~/.cache/huggingface/token                                  # HF auth, chmod 600
```

---

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

---

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| Gateway down after reboot | Service not enabled | `sudo systemctl enable llama-swap` |
| Engine "up" but tokens frozen (`/health` still 200, requests hang) | vLLM engine-core deadlock — seen with DFlash drafter + `max-num-seqs 8` under real traffic (§34) | `docker rm -f vllm-laguna-s21` (llama-swap relaunches via preload/cron). Detect with a stall watchdog: `generation_tokens_total` unchanged for minutes while `num_requests_running > 0`. Keep `LAG_SEQS ≤ 4` with the drafter |
| Co-resident engines won't both load | Insufficient GPU memory | Lower `--gpu-memory-utilization` on one engine; verify with `nvidia-smi --query-compute-apps` |
| A heavy dormant model OOMs on cold start | Resident pair holding ~99 GB (75B + vision) | The `experiments` group is swap-exclusive vs itself but **not** vs residents — a >20 GB dormant model may not fit beside them. `docker rm -f vllm-nemotron-puzzle-75b vllm-qwen-fast` to free the residents, or lower `NEMO_UTIL` |
| Dormant model won't load, `pull failed` in logs | codeserver (`192.168.1.16`) unreachable, or SSH key not loaded | `ssh 192.168.1.16 true` to check; dormant models require codeserver online. Residents are unaffected |
| Dormant model load times out (>20 min) | Pull + engine load exceeded `healthCheckTimeout` | Raise `healthCheckTimeout` in `llama-swap.yaml`; ds4's 85 GB pull alone is ~13 min |
| Local disk fills despite offload | A SIGKILL'd dormant model left stale weights | Next managed launch self-evicts them; or `rm -rf` the stale path from `etc/copyback-models.txt` manually |
| Both engines decode at half-speed | Single-GPU SM contention from concurrent compute | Expected — serialize the workload if sustained dual-engine load matters |
| `kv_cache_dtype not supported` on FLASH_ATTN | Backend doesn't support fp8 KV in this image | Use `--kv-cache-dtype auto` (bf16) — or switch to FLASHINFER backend |
| INT4+DFlash hangs at CUDA-graph capture | FULL graph mode + INT4-GPTQ + DFlash | Force `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` |
| 502 with `upstream command exited prematurely` | Orphan container holds port / VRAM | `docker rm -f vllm-<name>`; systemd unit handles via `ExecStartPre` |
| SIGKILL leaves orphan container | llama-swap's `healthCheckTimeout` kill doesn't trigger service cleanup | `docker rm -f <name>` manually; bump `healthCheckTimeout` |
| CUDA OOM on backend start | Previous container still resident | same as above; verify with `docker ps` |
| 122B NVFP4 fails with `Free memory ... less than desired GPU memory utilization` | Residual memory from prior models on unified memory | Lower `--gpu-memory-utilization` to 0.80 |
| Cold load 5× slower than expected | Default mmap path on Spark | ensure `--no-mmap` (llama.cpp) or `--load-format fastsafetensors` (vLLM) |
| 502 on first call, fine after | Model still loading | wait; watch `docker logs` for startup complete |
| `AssertionError: In Mamba cache align mode` | Qwen3.5 / Nemotron-Omni + prefix caching needs larger batch | `--max-num-batched-tokens 8192` |
| `PlaceholderModule should not be used` (audio path) | vLLM cached the missing-module placeholder before `av`/`soundfile` were installed | Rebuild the omni image with deps baked in — hot-install + restart isn't enough |
| `TimeoutError: VLLM_ENGINE_READY_TIMEOUT_S` | 600 s default too short for large models | `-e VLLM_ENGINE_READY_TIMEOUT_S=1800` |
| `ValueError: Tokenizer class TokenizersBackend` | Repo exported on transformers v5 | patch cached `tokenizer_config.json` to `"Qwen2TokenizerFast"` (see decision log §19) |
| Vision / audio missing on omni | vLLM's media path needs `av` / `soundfile` at import time | Use the custom `cu130-nightly-omni` image (decision log §29) |
| Flood of `Skipping tactic` / `Failed to initialize cutlass` | FlashInfer MoE autotuner on GB10 | expected, non-fatal — use `--moe-backend flashinfer_cutlass` to skip |

---

## Decision log

Reasons for non-obvious config choices, in roughly the order they were made. Numbering preserved for cross-references in launchers and yaml comments.

### 1. Solo 27B vs. the retired qwen36 co-resident pair *(superseded 2026-05-17)*

The 27B and 35B-A3B were a co-resident pair (qwen36 group, `swap: false`) until 2026-04-24, then solo 27B was production until 2026-05-17, when the pair config returned in a different form (Intel INT4 dense + RedHatAI NVFP4 MoE). The original 2026-04 split was 27B-dense for deep reasoning + 35B-A3B-MoE for sub-agent fanout; the solo era proved that NVFP4 27B at MTP-3 hits ~149 tok/s aggregate at c=10 alone. The 2026-05-17 return to co-residence trades single-stream peak (29 vs 31 tok/s solo) for zero-swap-latency between the two engines on mixed-model traffic.

### 2. KV cache quantization is mandatory, not optional

Weights + KV share the unified 119 GB. At long context, KV eats the budget.
- vLLM entries: `--kv-cache-dtype fp8` (~lossless on Blackwell, halves KV) where the attention backend supports it. The current dense prod is forced to bf16 KV because FLASH_ATTN doesn't support fp8 KV in the AEON image — that's the trade-off for keeping the DFlash drafter.
- llama.cpp entries: `--cache-type-k q8_0 --cache-type-v q8_0` or `q4_0` for tighter budgets. Q8 is indistinguishable from fp16; q4_0 used on SuperGemma, MiniMax, and agent-mode 122B.

### 3. Qwen3.5 is 262 k native — no YaRN

Earlier configs applied `--rope-scaling '{"rope_type":"yarn",…}'`. Those are Qwen3 params, not Qwen3.5. Qwen3.5 ships 262 k native in `config.json`. Static YaRN scales every request and degrades short-context quality. We removed all `--rope-scaling` for Qwen3.5/3.6 entries.

### 4. Jackrong distill is capped at 32 k

HF model card: SFT at 8 192 tokens. Architecture inherits 262 k positional embeddings, but distilled behavior was only trained ≤8 k. Cap at 32 k (4× SFT, benign extrapolation); recommend base `qwen3.5-35b-a3b` for genuinely long.

### 5. MiniMax M2.7 is full-attention, not linear

Earlier MiniMax-Text-01 used hybrid lightning/linear attention. **M2 series reverted to full softmax attention** (MiniMax's own engineering post). KV cost is now linear in context. With q4_0 KV + flash attention we run 64 k safely.

### 6. `--no-mmap` for llama.cpp on Spark

Default `mmap = true` copies tensors to CUDA via synchronous `cudaMemcpy` per page fault — NVMe hits ~200 MB/s at ~17 % utilization. `--no-mmap` uses `pread()` streaming, ~10× faster on Spark's 6.17 kernel.

### 7. `--load-format fastsafetensors` — considered, not applied

`fastsafetensors` cuts cold load times by 10×+, but the `cu130-nightly` image may or may not ship the package on any given day, and the flag hard-fails on missing. Warm-cache loads complete in ~80 s without it. Revisit if cold-start times become the bottleneck.

### 8. `-fa` flash attention for llama.cpp

Standard best practice on any CUDA-capable llama.cpp. Always on.

### 9. `VLLM_FLASHINFER_MOE_BACKEND=latency` env

Default FlashInfer MoE backend (`throughput`) emits SM120-generic kernels that misbehave on SM12.1 (GB10 chiplet). Force latency backend. Set on every vLLM entry except `qwen3.5-122b-nvfp4` (uses `--moe-backend flashinfer_cutlass` CLI flag instead — §17).

### 10. `--enable-prefix-caching` for vLLM

Single-user workloads reuse the same system prompts repeatedly. TTFT drops on repeated conversations. Applied to all vLLM entries (production dense has a known caveat with INT4+DFlash; verify acceptance metrics if prefix-cache hit rate is suspiciously high).

### 11. Container cleanup in systemd

llama-swap uses `docker run --rm` for clean exits, but an unclean shutdown leaves orphan containers holding GPU memory. `ExecStartPre`/`ExecStopPost` hooks run `docker ps -aq --filter name=llama- | xargs -r docker rm -f` to catch this.

### 12. Service enabled at boot

`systemctl enable llama-swap` — was `disabled` at install. Now auto-starts.

### 13. vLLM image must be `cu130-nightly` for Qwen3.5 MoE / RedHatAI NVFP4 MoE

`vllm/vllm-openai:gemma4-cu130` ships 0.14-branch which doesn't register `Qwen3_5MoeForConditionalGeneration`. The class landed in 0.16+; pull `cu130-nightly`.

### 14. `--max-num-batched-tokens ≥ block_size` for Qwen3.5 / Qwen3.6 hybrid Mamba

Hybrid attention + SSM architectures. With `--enable-prefix-caching`, vLLM enforces `block_size ≤ max_num_batched_tokens`. Default `max_num_batched_tokens=2048` but vLLM resolves `block_size=2096`. Fix: `--max-num-batched-tokens 8192` (or 32768 for the current prod launchers).

### 15. First-run cold load can exceed `VLLM_ENGINE_READY_TIMEOUT_S`

Default 600 s deadline. First load with uncached weights exceeds it. Every vLLM entry sets `-e VLLM_ENGINE_READY_TIMEOUT_S=1800`.

### 16. `--language-model-only` for vision-capable models

Qwen3.5/3.6 repos declare a vision tower in `config.json`; vLLM loads those weights by default. `--language-model-only` skips it. Applied to all text-only Qwen entries.

### 17. `--moe-backend flashinfer_cutlass` for NVFP4 MoE on GB10

FlashInfer's default MoE backend attempts SM 8.0/Ampere tactics that GB10 doesn't support. Applied to `qwen3.5-122b-nvfp4` historically, and to the current `qwen3.6-35b-a3b-nvfp4` MoE.

### 18. `--reasoning-parser qwen3` for Qwen3.5/3.6 chat completions

Qwen3.5/3.6 injects `<think>` blocks. Without a parser, OpenAI clients see raw `</think>` in `message.content`. The parser routes thinking to `message.reasoning_content`. **TTFT cost**: parser buffers the entire `<think>` block before emitting any delta — +3–7 s at short context, negligible at the 60 k+ coding traffic this stack serves. Applied selectively; some entries (DFlash, sakamaki rollback) leave it off because the client parses inline.

### 19. Tokenizer-class patch: `"TokenizersBackend"` → `"Qwen2TokenizerFast"`

Two Qwen3.5 repos ship `tokenizer_class: "TokenizersBackend"` (transformers v5.x). vLLM is pinned <5 and raises `ValueError`. In-place patch of cached `tokenizer_config.json`:

```bash
for model in \
  "Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled" \
  "RedHatAI--Qwen3.5-122B-A10B-NVFP4"; do
  f=$(find ~/.cache/huggingface/hub/models--${model} -name tokenizer_config.json -path "*/snapshots/*" | head -1)
  [ -f "${f}.bak" ] || cp "$f" "${f}.bak"
  python3 -c "import json; d=json.load(open('$f')); d['tokenizer_class']='Qwen2TokenizerFast'; json.dump(d, open('$f','w'), indent=2, ensure_ascii=False)"
done
```

### 30. ds4 best-config A/B/C — our custom-Q4 decode wins; rebased onto mainline *(2026-06-27)*

Searched for the fastest ds4 config for DeepSeek-V4-Flash on GB10. Ran an A/B/C concurrency matrix (`bin/bench-ds4-matrix.py`, c=1/2/4, 3-run median, idle 90 s, GB10 allocator confounder):

| Config | c=1 | c=2 agg | c=4 agg | peak mem |
|---|---:|---:|---:|---:|
| **A — fork q2 + custom Q4 decode** | **21.3** | 21.3 | 21.3 | ~105 GB |
| B — mainline q2 + MTP draft=4 | 13.8 | 13.8 | 13.8 | ~109 GB |
| C — mainline q2-q4 GGUF (native Q4_K experts) + MTP | 13.5 | 13.5 | 13.5 | 120/121 GB |

Findings: (1) **The two "Q4"s are different mechanisms** — our fork re-quantizes *dense* Q8/F16 weights → Q4_0 *at decode time* (`DS4_CUDA_Q4_DECODE`, the source of 21.3); mainline's Q4 is *routed-MoE experts stored Q4_K in the GGUF*, which never fires on our 2-bit (IQ2_XXS) GGUF. (2) **MTP is a dud on GB10** — B and C ≈ the bare q2 baseline (~13.75); speculation can't help a memory-bandwidth-bound decode. (3) **Native Q4_K experts didn't move decode** (C = B) — expert compute on 6 layers isn't the bottleneck; dense-weight bandwidth is. (4) **ds4 decode does NOT scale with concurrency** — aggregate is flat; c=2/4 just split the fixed throughput. Run ds4 one request at a time. (5) The q2-q4 GGUF barely fits one GB10 (0 headroom) → would OOM on real ~100 k-ctx traffic.

**Action: rebased our 3 custom-Q4 commits onto mainline** (`80ebbc3`, branch `q4-rebase` in worktree `~/ds4-rebase`). Conflicts were additive (upstream's SSD-streaming expert-cache structs/globals + host-register env knob alongside our Q4 cache). Rebuilt (`make cuda-spark`), verified **21.2 tok/s + coherent output**. So the live `deepseek-v4-flash-ds4` lane now runs **our custom-Q4 speed *and* the 225 upstream fixes** (tool-call recovery inside unclosed `<think>`, deterministic batched-prefill attention, SSD expert streaming, `--mtp-draft>2` verify fix). Launcher repointed to `~/ds4-rebase/ds4-server`; rollback = `~/ds4-q4/ds4-server` + `launch-ds4-server.sh.bak.20260627-pre-rebase`. Mainline's own metrics (`metrics_record_complete`) superseded our custom Prometheus patch (dropped; saved at `~/ds4-metrics-endpoint.patch`). **DFlash** (z-lab/ByteDance block-diffusion drafter) is *not* usable on ds4 yet — no V4-Flash drafter weights, no C/CUDA backend; watch `z-lab/dflash`.

`hf download --force` reverts the patch — re-apply if weights are re-fetched.

### 20. MTP speculative decoding for Qwen3.5 / Qwen3.6

Both ship native MTP weights. MTP-1 predicts 1 additional token per step with high acceptance. Added `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (MoE) or `qwen3_next_mtp`/`qwen3_5_mtp` (dense variants). **Requires `--attention-backend FLASHINFER` on the older NVFP4 paths** — MTP silently disables without it. The current dense prod uses DFlash (different drafter), not MTP — see launcher details above. The 122B NVFP4 cannot use MTP — RedHatAI stripped the head during quantization.

### 21. `--attention-backend FLASHINFER` for vLLM on Spark

vLLM may default to FLASH_ATTN on SM121. FlashInfer has optimized kernels for the Blackwell memory hierarchy (+16 % per albond benchmarks). Applied to NVFP4 entries. **The current INT4+DFlash dense prod uses `flash_attn` instead** because DFlash's drafter requires it — that's the load-bearing reason it can't use fp8 KV (FLASH_ATTN doesn't support fp8 KV in this image).

### 22. Gemma 4 switched from vLLM to llama.cpp

Two compounding vLLM bugs:
1. **vLLM #39407/#39049**: `--quantization fp8` produces garbage on all Gemma 4 variants. FP8 checkpoints have activation scales pre-absorbed but vLLM applies dynamic per-token quant on top → logit saturation.
2. **vLLM #38887**: Gemma 4's heterogeneous head dims (256 SWA, 512 global) force TRITON_ATTN fallback, 10–15× slowdown.

llama.cpp has no SM121 kernel compatibility issues. Gemma E4B: 18.5 → 37.9 tok/s (+105 %); Gemma 26B: 37.7 → 58.3 tok/s (+55 %), with 75–83 % less memory.

### 23. `--parallel N` splits context across slots

llama.cpp's `--parallel N` creates N concurrent slots but **divides total `-c` evenly**. So `-c 131072 --parallel 4` = 32 K per slot, not 131 K. Weights are batched across slots (real batching), but each slot gets `total_ctx / N`.

For agent mode: `-c 262144 --parallel 4` = 64 K per slot (reasoning); `-c 65536 --parallel 4` = 16 K per slot (worker).

### 24. Vision via mmproj on Gemma E4B

Gemma 4 E4B supports multimodal via `--mmproj` in llama.cpp. mmproj file adds ~200–500 MB overhead. Standard OpenAI `image_url` content type. Qwen3.5-122B does **not** support vision via llama.cpp — CLIP graph uses unsupported operators (llama.cpp #21268).

### 25. `qwen3.6-35b-a3b` launcher script — drop `--rm` for crash traceability

MoE has been observed silently crashing under sustained AsyncOpenAI workload: vLLM exits status 0, llama-swap logs `<qwen3.6-35b-a3b> process exited but not StateStopping`. With `docker run --rm`, the dead container is wiped before `docker logs` can be read. Two fixes:
1. Drop `--rm` so logs survive the crash.
2. Clean up the named container before respawning (`docker rm -f X 2>/dev/null; exec docker run --name X …`).

llama-swap parses `cmd:` with shellwords (no shell), so compound commands can't live inline. Move to a script file and point yaml at it:

```yaml
qwen3.6-27b-int4-dflash:
  cmd: /home/max/llm-stack/bin/launch-vllm-27b-int4-dflash.sh
```

Apply the pattern to every vLLM container where post-mortem logs matter.

### 26–26d. Pre-2026-05-17 27B production lineage

The 27B has been through five production configs since 2026-04-24:

| Date | Config | Speed | Demoted because |
|---|---|---:|---|
| 2026-04-24 → 2026-04-30 | AlphaOxO NVFP4 + MTP-3 (§26b) | ~19 tok/s c=1 / 149 c=10 | DFlash wins on short-context bench |
| 2026-04-30 → 2026-05-08 AM | AEON-7 NVFP4 + DFlash k=15→k=4 (§26) | 41 tok/s c=1 / 207 c=10 | Drafter style-mismatch on long-context coding (acceptance 6 % @ 100 k) |
| 2026-05-08 AM | Sakamaki NVFP4 + MTP graft n=3 (§26d) | (won A/B; -57 % wall-clock @ 100 k) | `</think>C` truncation in real opencode traffic same day |
| 2026-05-08 PM → 2026-05-17 | Qwen official FP8 + native MTP n=3 (§26c) | 14.8 tok/s c=1 | Bandwidth-bound; INT4 path is +80 % |
| 2026-05-17 → now | Intel/Qwen3.6-27B-int4-AutoRound + z-lab DFlash n=4 (above) | 29 tok/s c=1 | (live) |

The launchers for all five are still on disk. Detailed gotchas for each are inline in the launcher scripts under `bin/launch-vllm-27b-*.sh`.

### 27. llama-swap `-watch-config` hot-reload (2026-04-26)

Runs with `-watch-config` via `/etc/systemd/system/llama-swap.service.d/watch-config.conf` (mirrored in repo at `systemd/llama-swap.service.d/watch-config.conf`). Yaml edits apply in ~1 s **without restarting the proxy** and **without touching running model containers** — no cold-start tax on config changes.

Drop-in contents:

```ini
[Service]
ExecStart=
ExecStart=/home/max/bin/llama-swap -config /home/max/llm-stack/config/llama-swap.yaml -watch-config -listen 0.0.0.0:8080
```

The empty `ExecStart=` line is required to override the unit's original ExecStart. Apply with `sudo install -m 0644 systemd/llama-swap.service.d/watch-config.conf /etc/systemd/system/llama-swap.service.d/watch-config.conf && sudo systemctl daemon-reload && sudo systemctl restart llama-swap`.

Fallback for non-yaml changes: `pkill -9 llama-swap`. SIGTERM (plain `pkill`) is a clean exit so systemd does NOT respawn (`Restart=on-failure`); only SIGKILL or `sudo systemctl restart` works.

### 28. MoE `qwen3.6-35b-a3b` route history *(superseded 2026-05-17)*

From 2026-04-27 → 2026-05-17, the `qwen3.6-35b-a3b` name was an alias on the active 27B entry. The real MoE never cold-started during that window — requests to that name routed to the 27B FP8.

**Since 2026-05-17, the real MoE is back** under the explicit name `qwen3.6-35b-a3b-nvfp4` (co-resident with the dense entry). The legacy `qwen3.6-35b-a3b` alias still points at the **dense** route (preserved for backwards compatibility with old clients). New clients should call `qwen3.6-35b-a3b-nvfp4` explicitly when they want the MoE path.

### 29. `nemotron-3-nano-omni` multimodal slot (2026-04-29)

NVIDIA's Mamba2-Transformer hybrid MoE 30B/3B-active with CRADIO-v4-H vision and Parakeet audio encoders. Swap-exclusive member of `experiments` like the rest.

**Custom image required.** vLLM's audio path needs `av` + `soundfile` + `librosa` at process import time; base `cu130-nightly` ships without them. Hot-installing into a running container leaves a cached `PlaceholderModule` and audio fails. Build a derived image once:

```bash
docker exec vllm-nemotron-omni pip install av soundfile librosa
docker commit vllm-nemotron-omni vllm/vllm-openai:cu130-nightly-omni
```

Other notable flags (full set in the launcher):
- `--max-num-batched-tokens 8192` (Mamba block-size assertion — see §14).
- `--max-num-seqs 8` (concurrency peak at c=8 → 383 tok/s agg; sharp cliff at c=9).
- `--gpu-memory-utilization 0.65` (≈77 GB; multimodal scratch peaks +1.8 GB on 1080p video).
- `--max-model-len 131072` (model native 262 k — conservative; multimodal KV grows fast).
- `--reasoning-parser nemotron_v3` — buffers `<think>` block, +18 s TTFT on reasoning prompts. Drop if interactive UX matters more than parsed `message.reasoning_content`.
- `--kv-cache-dtype fp8`, `--tool-call-parser qwen3_coder`, `--video-pruning-rate 0.5`.

Cold start ~150 s steady state (warm HF cache).

### 31. Concurrency retune — c=1/2/3 @ 131 k for the swap models *(2026-06-27)*

The launcher contexts/concurrency were co-resident-era fossils (`gpu-util 0.40–0.70`, reduced `--max-model-len`, throughput-tuned `--max-num-seqs`). Solo + swap-exclusive, that left 30–70 GB unused on most models. Retuned every model **except `int4-dflash` and `ds4`** to serve **c=1/2/3 at 131 k**, sized by KV math against ~113 GB usable (weights + `seqs × 131 072 × KV/token`):

| Model | Before (len/seqs/util) | After | KV/tok | Max c @ 131 k |
|---|---|---|---|:---:|
| `qwen3.6-35b-a3b-nvfp4` | 80 k / 2 / 0.40 | 131 k / 3 / 0.70 | 145 KB fp8 | c=3 |
| `qwen3.6-27b-fp8` | 200 k / 2 / 0.70 | 131 k / 3 / 0.80 | 145 KB fp8 | c=3 |
| `qwopus3.6-27b-int4-dflash` | 128 k / 8 / 0.85 | 131 k / **2** / 0.85 | 290 KB bf16 | **c=2** |
| `nemotron-3-nano-omni` | 131 k / 8 / 0.65 | 131 k / 8 / 0.72 | 145 KB fp8 | c≥3 (util bump for headroom) |
| `ornith-1.0-35b` | 32 k / 1 slot | `--ctx 393216 --parallel 3` | 125 KB f16 | c=3 |
| `diffusiongemma-26b` | unchanged (131 k / 4 / 0.30) | — | diffusion | c=4 (seqs=4 card-mandatory) |

Excluded by design: **`int4-dflash`** stays single-stream — it's the prefill-bound coding default, and a 131 k prompt already costs ~150 s TTFT, so concurrency would compound that. **`ds4`** stays single-stream — its decode is bandwidth-bound and does **not** scale with concurrency (measured: flat aggregate at c=1/2/4, see §30). **`qwopus` is c=2 only**: its DFlash path forces FLASH_ATTN, which can't do fp8 KV, so each stream's bf16 KV is ~2× heavier and c=3 wouldn't fit. Values are KV-math estimates (±10 % vs vLLM's real block accounting) — verify each loads at the target before trusting c=3 under load.

### 32. Two-tier residents + codeserver copy-back offload *(2026-07-13)*

The Spark's 916 GB NVMe hit 95 %. Rather than delete any of the ~11 dormant eval/rollback models, their weights were moved to **codeserver** (`192.168.1.16`, 1.9 TB, 2.5 GbE) and are pulled back on demand — dropping local usage to ~48 % (**~408 GB freed, nothing deleted**). This also formalized the **two-tier** split implicit since the 2026-07-08 resident-pair revival:

- **Resident tier** (`resident` group, `persistent: true`, co-resident): `nemotron-3-puzzle-75b` + `qwen3.6-35b-a3b-vision` — always local, always loaded. Supersedes the "every model loads solo with the full 119 GB" framing of §1/§31 (which still holds for the dormant pool).
- **Dormant tier** (`experiments` group): weights on codeserver, each slot's `cmd:` wrapped by `bin/copyback-launch.sh` (pull-on-start, evict-on-stop, one-at-a-time via `etc/copyback-models.txt`).

Design choices worth recording: **evict-immediately-after-use** (leanest disk; re-pulls each cold start) was chosen over LRU retention — dormant models are rarely hit, so paying the pull each time beats holding tens of GB resident. **`healthCheckTimeout` raised to 1200 s** so a pull can't trip the health check mid-download (ds4's 85 GB GGUF is a ~13 min pull). **HF-cache blobs are root-owned** (docker populates the cache as root), so `rm` as `max` can't remove them — delete via a throwaway `docker run --rm -v ~/.cache/huggingface:/hf alpine rm -rf /hf/hub/models--…` (no sudo; `max` is in the `docker` group). **New dependency:** dormant models require codeserver online to load; the residents do not. Full writeup: `memory project_weights_offload_codeserver`.

### 33. Laguna S-2.1 is the coding default; v2 "spinquantless" weights *(2026-07-22 / 07-23)*

**poolside/Laguna-S-2.1-NVFP4** (118 B / 8.5 B-active MoE) + its **DFlash NVFP4 drafter** replaced `nemotron-3-puzzle-75b` as coding default on 2026-07-22 — 33 tok/s @100 k beat the 75B's 20, with 3.3× KV concurrency at the full native 256 k window. The old default names (`qwen3.6-27b`, `qwen3.6-35b-a3b`, `nemotron-3-puzzle-75b`) were remapped to it **at the LiteLLM gateway** (llama-swap v201 drops yaml aliases — §*LiteLLM integration*). The vision lane went **dark** — no headroom beside solo laguna.

On 2026-07-23 the lane moved to poolside's re-uploaded **"spinquantless norot" weights** (revision `0761412`, launcher `-v2.sh`): re-quantized without SpinQuant rotations, the apparent fix for the looping reports (HF discussions #4–#7, #10 — #11 suggests the original rotate checkpoint never ran correctly on public runtimes). Smoke-verified: looping gone, and DFlash decode *improved* (43–48 tok/s code/math). **Both launchers are revision-pinned** so an upstream re-upload can't silently change prod; v1 (`b482b5d`, original weights) is the one-line rollback. Lesson worth keeping: **never benchmark speculative decoding with a random-text harness** — acceptance collapses on noise and the numbers say nothing about real code traffic.

### 34. Laguna concurrency ceiling — seqs-8 deadlock; qwen co-residency parked *(2026-07-24)*

Dev-box asked for `max-num-seqs` 4 → 8 (DGX issue #2) to serve a multi-agent workload. The day produced four durable findings:

1. **DFlash n=15 + `max-num-seqs 8` deadlocks the vLLM engine core under real traffic** — twice reproduced (~48 k-token prompts, chunked prefill; token counter freezes while `/health` stays 200, so llama-swap never restarts it). A synthetic c=8 probe passed; only the real traffic mix triggers it. Drafterless seqs=8 is stable but ~19 tok/s single-stream vs 43–54 with the drafter — the drafter is worth more than the extra slots. **Community corroboration:** DFlash crashes vLLM outright at the default seqs=256; the one published stable config (MiaAI-Lab) pins seqs=4 / n=7. → `LAG_SEQS=4` is a hard ceiling while `LAG_SPEC=1`. Detection: stall watchdog on `generation_tokens_total` frozen while `num_requests_running > 0`.
2. **Co-residency math for GB10 (measured via 4 crash-loop attempts):** laguna's non-KV footprint is **~70.6 GB** (weights + drafter + runtime), KV costs **38.4 KB/token** fp8, and vLLM refuses to start unless one full `max-model-len` request fits in the KV pool. `util 0.66 / ctx 131072` is the smallest proven laguna shape (leaves ~30 GB for a neighbor).
3. **A second vLLM engine cannot load next to resident laguna.** The qwen 35B-A3B NVFP4 fast lane (`launch-vllm-35b-moe-nvfp4-colag.sh`) ran healthy once at 70 tok/s c=1, but vLLM's **weight-load transient spikes ~5–6 GB above its steady reservation** — MemAvailable hit 1 GB and the OOM guard killed it, twice. Since the **GB10 hard-hangs on host OOM** (no remote recovery), the lane is **parked**: yaml entry removed so nothing can launch it by name. Revival preconditions (in the launcher header): re-shrink laguna to 0.66/131 k *first*, then either `sudo drop_caches` pre-launch or switch the lane to llama.cpp/GGUF (mmap pages are reclaimable, unlike vLLM's anonymous allocation).
4. **Observability gap:** LiteLLM now sends **`/v1/responses`** (Responses API) for laguna traffic; `log-proxy` only parses `/v1/chat/completions`, so per-request logs and `traffic-tui` are currently **blind to prod traffic** (zero files written all morning despite healthy 200s). Until the proxy is extended, judge engine health from vLLM `/metrics` + llama-swap logs, not the proxy tree. A related client-side note: a long thinking generation at 20–26 tok/s legitimately runs 2–4 min — downstream harnesses need timeout budgets that don't misread it as a hang.

Config restored same evening to the proven solo shape: `util 0.85 / ctx 262144 / seqs 4 / DFlash n=15` (KV 856,686 tokens, 54.2 tok/s smoke, 2 h watchdog clean). Full audit trail on DGX issue #2; measured math in the launcher headers and `memory project_coresident_split_20260724`.

---

## Archive: historical benchmarks

Kept for reference. Configs reflect the model lineup of each era.

### Pre-Qwen3.6 lineup (2026-04-16, swap mode)

| Model | Cold start | Memory | tok/s (e2e) | TTFT | Decode tok/s |
|---|---:|---:|---:|---:|---:|
| `qwen3.5-35b-a3b` | 238 s | 102 GB | 62.2 | 8.2 s * | ~instant (MTP buffered) |
| `gemma-4-26b-a4b` | 79 s | 21 GB | 58.3 | 5.3 s * | 231 |
| `supergemma-4-26b` | 67 s | 32 GB | 44.0 | 124 ms | 46 |
| `qwen3.5-35b-distill` | 209 s | 107 GB | 42.4 | 8.8 s * | 156 (MTP) |
| `gemma-4-e4b` | 47 s | 13 GB | 37.9 | 70 ms | 39 |
| `minimax-m2.7` | 73 s | 108 GB | 23.6 | 9.9 s * | 87 |
| `qwen3.5-122b-nvfp4` | 613 s | 110 GB | 15.3 | 33 s * | unstable |

\* TTFT includes hidden thinking time (`--reasoning-parser qwen3`).

### qwen3.6 pair spot-check (2026-04-23, llama.cpp GGUF era)

| Concurrency | Aggregate tok/s | TTFT p50 | TTFT p95 | Per-req tok/s |
|---|---:|---:|---:|---:|
| 1 | 11.07 | 0.32 s | 0.32 s | 11.07 |
| 4 | 29.51 | 0.65 s | 0.65 s | 7.47 |
| 10 | 25.34 | **55.55 s** | **109.90 s** | 7.47 |

Throughput regressed c=4 → c=10 because llama.cpp's slot count couldn't keep up. Motivated the 2026-04-24 switch to vLLM NVFP4+MTP.

### Solo 27B NVFP4 + MTP (2026-04-24 → 04-30 production)

| n | c=1 decode | c=10 peak agg | MTP accept |
|---|---:|---:|---|
| 1 | ~13 tok/s | 120 tok/s | 1/1 (94 %) |
| 2 | ~16 tok/s | 136 tok/s | mean 1.74/2 (87 %, 75 %) |
| **3** | **~19 tok/s** | **149 tok/s** | mean 3.0/4 (85 %, 63 %, 51 %) |

Through llama-swap adds ~3 % vs direct-to-:9008. Net vs GGUF-pair-era at c=10: 25 → 149 tok/s (~6×).

### Solo 27B NVFP4 + DFlash k=15 (2026-04-30 → 05-08 AM production)

| c | DFlash agg | MTP agg | Δ |
|---|---:|---:|---:|
| 1 | **41.0** | 20.3 | +102 % (2.0×) |
| 5 | **139.0** | 92.0 | +51 % |
| 10 | **207.1** | 169.1 | +22 % |

Wins on short-context but acceptance collapsed on long-context coding (37 % @ 13 k → 6 % @ 100 k). Demoted. Deep writeup: [`docs/qwen3.6-27b-dflash.md`](docs/qwen3.6-27b-dflash.md).

### Real-traffic-shaped A/B (2026-05-08, drove the AM rollover)

| Bucket | AEON DFlash k=4 | sakamaki MTP n=3 | Δ wall-clock |
|---|---:|---:|---:|
| 13 k / 1024 out | 75.0 s | 50.6 s | **−32 %** |
| 60 k / 1500 out | 274.4 s | 132.8 s | **−52 %** |
| 100 k / 2048 out | 534.6 s | 230.2 s | **−57 %** |

Decode tok/s @100 k: 4.9 (DFlash) → 18.5 (MTP), +278 %. Acceptance @100 k: 6.0 % vs 66.2 %. Full report: `logs/bench-coding-AB-report-20260508.md`.

### 2026-05-17 quant comparison (current era)

Configurations tested in a 4-hour optimization pass, single-stream prompt~200 / 1024 out:

| Config | tok/s | mean accept | per-pos rates | Notes |
|---|---:|---:|---|---|
| Dense INT4 base only (no spec) | 13.4 | — | — | GPTQ-Marlin baseline |
| Dense INT4 + MTP n=1 | 19.4 | 1.86 | 0.86 | small spec window |
| Dense INT4 + MTP n=2 | 22.0 | 2.26 | 0.75/0.51 | Intel card recommended |
| Dense INT4 + DFlash n=8 | 26.7 | 2.86 | drops past pos 4 | drafter-distill mismatch |
| **Dense INT4 + DFlash n=4 (prod)** | **27 → 29 (v4 image)** | 2.96 | 0.78/0.58/0.40/0.20 | winner |
| AlphaOxO NVFP4 + DFlash n=15 | 19.2 | 3.0 | accept near-zero past pos 6 | n=15 drafter overhead dominates |
| Dense FP8 + native MTP n=3 (demoted prod) | 14.8 | 2.85 | 0.74/0.55/0.42 | bandwidth-bound |
| **MoE NVFP4 + MTP n=1 (prod)** | **56.6** | 1.83 | 0.83 | 3.79× over FP8 prod |

Long-context (125 k input, 1024 out, c=1): MoE 48.5 tok/s / 38 s TTFT; dense 18 tok/s / 151 s TTFT.

---
