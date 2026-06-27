# Spark LLM Stack

OpenAI-compatible LLM gateway on one **DGX Spark (GB10, 119 GB unified)**. Call it at **`http://192.168.1.12:8079/v1`** and route by the `model` field. **One model is resident at a time** — asking for a different `model` unloads the current one and cold-loads the next (1–10 min for vLLM engines, ~30–105 s for the ds4 / llama.cpp binaries).

> **Phase (2026-06-27): experimentation — no fixed production model.** Every model is a swap-exclusive member of the single `experiments` group, so each loads alone with the full 119 GB. Pick the one you want from the table. The `qwen3.6-27b` / `qwen3.6-35b-a3b` aliases still resolve for legacy clients.

## Models

Pick by `Route` (the `model` value). Speed/mem are from the 2026-06-27 sweep (`bin/bench-models.py`, single-stream, each cold-loaded alone; raw in `logs/bench-results.json`). **Settings for each model live in its launcher** (`bin/<launcher>`) and are explained in [Per-launcher details](#per-launcher-details).

| Route (`model`) | Use it for | tok/s | Max ctx | Peak mem | Engine | Launcher (`bin/`) |
|---|---|---:|---:|---:|---|---|
| **`qwen3.6-27b-int4-dflash`** | **coding — default** | 41 | 120 k | 61 GB | vLLM | `launch-vllm-27b-int4-dflash.sh` |
| `qwen3.6-35b-a3b-nvfp4` | coding, long-ctx throughput | 61 | 131 k | 53 GB | vLLM | `launch-vllm-35b-moe-nvfp4.sh` |
| `ornith-1.0-35b` 🆕 | coding — agentic (thinking)² | 77 | 131 k¹ | 25 GB | llama.cpp | `launch-ornith.sh` |
| `qwopus3.6-27b-int4-dflash` | coding — Opus-distilled | 39 | 131 k | 102 GB | vLLM | `launch-vllm-qwopus-int4-dflash.sh` |
| `qwen3.6-27b-fp8` | coding — quality baseline | 21 | 131 k | 91 GB | vLLM | `launch-vllm-27b-qwen-fp8.sh` |
| `nemotron-3-nano-omni` | multimodal (image/audio/video) | 56 | 131 k | 91 GB | vLLM | `launch-vllm-nemotron-omni.sh` |
| `cosmos3-nano-omni` 🆕 | image/video generation³ | — | — | ~30 GB | vLLM-Omni | `launch-vllm-cosmos3-nano-omni.sh` |
| `diffusiongemma-26b` | speed / non-coding | 142 | 131 k | 50 GB | vLLM | `launch-vllm-diffusiongemma-nvfp4.sh` |
| `deepseek-v4-flash-ds4` | long-context planner | 21 | 131 k | 22 GB | ds4 | `launch-ds4-server.sh` |

¹ Ornith ctx is `ORNITH_CTX`/`ORNITH_PARALLEL`-tunable (now 3 slots × 131 k). &nbsp; ² **Thinking model** — output goes to `reasoning_content`; give generous `max_tokens` or `content` returns empty. &nbsp; ³ **Generation model, not chat** — call `POST /v1/videos` · `/v1/videos/sync` · `/v1/images/generations` (multipart), not `/v1/chat/completions`; tok/s and token-ctx don't apply. &nbsp; tok/s / mem above are the 2026-06-27 single-stream sweep (pre-retune).

> **Concurrency (2026-06-27 retune):** every model **except `int4-dflash` and `ds4`** is now tuned to serve **c=1/2/3 at 131 k context** (`qwopus` is c=2 — its DFlash path forces heavier bf16 KV). `int4-dflash` stays single-stream (it's the prefill-bound coding default) and `ds4` stays single-stream (its decode doesn't scale with concurrency). See [decision #21](#decision-log).

### What each model is for

- **`qwen3.6-27b-int4-dflash`** — Alibaba's **Qwen3.6-27B dense**, the coding workhorse; on our own evals it beat the 35B-A3B MoE by ≥4 pts SWE-bench, so it's the default. Intel's AutoRound **INT4** keeps quality within noise of FP8 while halving the weight bandwidth that bottlenecks GB10, and the z-lab **DFlash** speculative drafter pushes it to ~41 tok/s. Reach for it for everyday coding and agentic/tool-use work.
- **`qwen3.6-35b-a3b-nvfp4`** — Qwen's **sparse MoE** (35B total, ~3B active/token), so it sidesteps the bandwidth wall and stays cheap under load. RedHat's NVFP4 quant + native MTP give 61 tok/s single-stream and the best concurrency/long-ctx throughput in the stack. Use it when you want speed and parallelism over the dense model's last few quality points.
- **`qwen3.6-27b-fp8`** — the **official Qwen FP8** build, "near-lossless" per the card with no community-quant noise. It's the ground-truth reference: when an INT4/NVFP4 result looks off, A/B against this. Use it for quality baselining and canonical Qwen3.6 behavior.
- **`qwopus3.6-27b-int4-dflash`** — Jackrong's **Qwopus**, a Qwen3.6-27B fine-tune **distilled on Claude-Opus reasoning traces**, so it keeps Qwen's coding base but reasons in a more Opus-like style. Same INT4+DFlash speed path as the default. Use it for coding/reasoning when you want Opus-flavored chains of thought.
- **`ornith-1.0-35b`** 🆕 — DeepReinforce's **Ornith-1.0** (MIT), a new agentic-coding MoE (35B/3B-active) that **writes its own RL training scaffold**; it scored **64.2 on Terminal-Bench 2.1, beating Qwen3.5-397B** (10× its size). Thinking model, and the fastest coding model here at 77 tok/s. The most interesting new model to pit against the Qwen incumbents on agentic/terminal tasks.
- **`diffusiongemma-26b`** — Google DeepMind's first **diffusion LLM**, which denoises 256-token blocks instead of decoding token-by-token, hitting ~142 tok/s (fastest in the stack). Google notes quality is below autoregressive Gemma 4, so it's a **speed lane, not a coding lane**. Use it for fast non-coding work — summaries, drafts, classification — where latency beats peak quality.
- **`nemotron-3-nano-omni`** — NVIDIA's **multimodal omni** model, a Mamba2-Transformer hybrid MoE with vision + audio encoders handling **text, image, audio, and video** in one model. It's the only multimodal option here — use it for anything the text-only coders can't see or hear.
- **`cosmos3-nano-omni`** 🆕 — NVIDIA's **Cosmos3-Nano** world model (~15B omni), the *generation* counterpart to the analysers above: it **produces** images and video from text/image prompts via vLLM-Omni. Unlike everything else in this table it is **not a chat model** — call `/v1/videos`, `/v1/videos/sync`, or `/v1/images/generations` (see footnote ³). The 64B Cosmos3-Super needs multi-GPU and stays standalone in `~/cosmos3`; Nano is the variant that fits one GB10. Use it for text→image/video generation.
- **`deepseek-v4-flash-ds4`** — DeepSeek's **V4-Flash**, whose multi-head-latent / compressed-KV attention scales to **256 k+ context cheaply**, run on the from-scratch antirez/ds4 C/CUDA engine with persistent disk-KV. Decode is slow (~21 tok/s) and doesn't parallelize, so it's a **planner, not a chat workhorse**. Use it for long-context planning/reasoning over whole codebases or long documents.

### Call it

```sh
curl http://192.168.1.12:8079/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-27b-int4-dflash",
  "messages": [{"role": "user", "content": "hi"}]
}'
# Set client timeout ≥ 900s — a cold swap can take up to 10 min.
# :8079 is the logged path (log-proxy → llama-swap); :8080 is the direct gateway.
```

### Operate

```sh
curl -s http://192.168.1.12:8080/v1/models | jq -r '.data[].id'   # list routes
curl -s http://192.168.1.12:8080/running                          # which model is hot
sudo systemctl status llama-swap                                  # service health
curl -X POST http://192.168.1.12:8080/unload                      # free the resident model
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

### Group semantics

- **`experiments`** (`swap: true, exclusive: true`) — the only group. At most one model resident at a time; requesting a different model unloads the current one and cold-loads the next. Cold-swap latency 1–10 min for vLLM engines, ~30–105 s for the llama.cpp (Ornith) and ds4 native binaries. Each loads alone with the full 119 GB available.
- *(historical: the `qwen-coresident` + `main` split was retired 2026-06-27 when we dropped the dedicated-prod model. See [decision log](#decision-log).)*

---

## Per-launcher details

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

### Co-residence contention behavior (single GPU caveat)

| Scenario | Dense | MoE |
|---|---:|---:|
| Dense fires alone (MoE resident-idle) | 31.2 tok/s | (idle) |
| MoE fires alone (dense resident-idle) | (idle) | 101.3 tok/s |
| Both decode simultaneously | 17.3 | 39.4 |

Memory is partitioned cleanly. Compute isn't — GB10 is one physical GPU, and two vLLM containers competing for SM cycles halves each engine's throughput. Co-residence wins for **bursty / sequential** mixed traffic (one busy at a time); it loses for **sustained concurrent** dual-engine load.

---

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Gateway logs
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Active container logs (production)
docker logs -f vllm-qwen-27b-int4-dflash   # qwen3.6-27b (dense, prod)
docker logs -f vllm-qwen-35b-moe-nvfp4     # qwen3.6-35b-a3b-nvfp4 (MoE)

# Rollback / specialty containers
docker logs -f vllm-qwen-27b-fp8           # qwen3.6-27b-fp8 (demoted prod 2026-05-17)
docker logs -f vllm-qwen-27b-int4-autoround # qwen3.6-27b-int4-autoround
docker logs -f vllm-qwen-27b-sakamaki-mtp  # sakamaki rollback
docker logs -f vllm-qwen-27b-dflash        # aeon-dflash rollback
docker logs -f vllm-qwen-27b               # mtp rollback (oldest)
docker logs -f vllm-qwen-27b-clean-dflash  # eval slot
docker logs -f vllm-nemotron-omni          # multimodal omni
docker logs -f ds4-server                  # deepseek-v4-flash-ds4 (planner)

# State inspection
curl http://192.168.1.12:8080/running
curl http://192.168.1.12:8080/v1/models
docker ps --filter name=vllm- --filter name=llama-

# Force an unload
curl -X POST http://192.168.1.12:8080/unload

# Kill a co-resident engine to free memory for a swap-exclusive heavy
docker rm -f vllm-qwen-35b-moe-nvfp4   # frees ~49 GB
docker rm -f vllm-qwen-27b-int4-dflash # frees ~55 GB
```

### Rollback to FP8 prod

```sh
# Edit config/llama-swap.yaml — move these aliases:
#   from qwen3.6-27b-int4-dflash entry
#   to   qwen3.6-27b-fp8 entry
#     aliases:
#       - qwen3.6-27b
#       - qwen3.6-35b-a3b
# Save. -watch-config picks it up in ~1 s. Next request triggers swap-in of fp8.
# Optional: docker rm -f vllm-qwen-27b-int4-dflash to free its 55 GB now.
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
4. **Append the new key** to the `experiments` group members list.
5. **Validate**: `python3 -c "import yaml; yaml.safe_load(open('config/llama-swap.yaml'))"`.
6. **Save** — `-watch-config` reloads within ~1 s (see decision log §27). Verify via `curl -s http://localhost:8080/v1/models`.
7. **Smoke test** by sending a 5-token completion — first request is the cold load.

Fallback for binary / unit changes: `pkill -9 llama-swap` (SIGKILL → systemd respawn; SIGTERM does not — `Restart=on-failure`).

---

## LiteLLM integration

Points at the log-proxy on `:8079` so every request lands in `~/llm-stack/logs/proxy/{date}/{model}/`. `deployed.yaml` in this repo is a drop-in `model_list` for LiteLLM. Minimum per-entry:

```yaml
model_list:
  - model_name: qwen3.6-27b              # what clients call
    litellm_params:
      model: openai/qwen3.6-27b          # must match the llama-swap map key
      api_base: http://192.168.1.12:8079/v1
      api_key: none
      timeout: 900                       # ≥ cold-load time
litellm_settings:
  request_timeout: 900
```

The `openai/<key>` string in `litellm_params.model` must exactly match the key under `models:` in the active llama-swap config — that's how llama-swap routes and decides which backend to swap in.

---

## Files

```
~/llm-stack/
├── config/
│   └── llama-swap.yaml             # active config (single `experiments` swap group)
├── deployed.yaml                   # source-of-truth manifest for downstream (LiteLLM)
├── bin/
│   ├── launch-vllm-27b-int4-dflash.sh   # ACTIVE — dense prod (Intel INT4 + DFlash n=4, v4 image)
│   ├── launch-vllm-35b-moe-nvfp4.sh     # ACTIVE — MoE prod (RedHatAI NVFP4 + MTP n=1)
│   ├── launch-vllm-27b-qwen-fp8.sh      # rollback — was prod 2026-05-08 → 2026-05-17
│   ├── launch-vllm-27b-int4-autoround.sh # eval — Intel INT4 + native MTP (no DFlash)
│   ├── launch-vllm-27b-sakamaki-mtp.sh  # rollback — NVFP4 + MTP graft
│   ├── launch-vllm-27b-dflash.sh        # rollback — aeon NVFP4 + DFlash k=4
│   ├── launch-vllm-27b-nvfp4.sh         # rollback — AlphaOxO NVFP4 + MTP-3
│   ├── launch-vllm-27b-clean-dflash.sh  # eval — non-prod
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
│   └── qwen3.6-chat-template-froggeric.jinja  # chat template used by Qwen3.6 entries
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
| Co-resident engines won't both load | Insufficient GPU memory | Lower `--gpu-memory-utilization` on one engine; verify with `nvidia-smi --query-compute-apps` |
| `nemotron` or `ds4` OOMs on cold start | Qwen co-resident pair holding ~104 GB | `docker rm -f vllm-qwen-{27b-int4-dflash,35b-moe-nvfp4}` first |
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
