# Operations

Gateway topology, the two-tier resident/dormant mechanics, weight offload,
day-2 runbook, rollback procedures, and troubleshooting. Model-level detail:
[models.md](models.md) · why things are configured this way: [decisions.md](decisions.md).

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

**How a dormant model loads.** Each `experiments` slot's `cmd:` in `llama-swap.full.bak` is wrapped by [`bin/copyback-launch.sh`](../bin/copyback-launch.sh):

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


## Runbook

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
│   ├── models.md                        # full model matrix + per-launcher details
│   ├── operations.md                    # this file
│   ├── decisions.md                     # decision log (§1–§35)
│   ├── benchmarks.md                    # bench tooling + historical archive
│   ├── qwen3.6-27b-dflash.md            # deep DFlash writeup
│   └── deepseek-v4-flash.md             # ds4 deep writeup
├── venv/                                # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.{log,err}             # gateway std{out,err}
│   ├── proxy/{date}/{model}/            # per-request triples from log-proxy
│   ├── bench-deep-latest.json           # symlink to latest deep bench
│   └── bench-*.json                     # timestamped runs
└── README.md                            # front door — architecture, highlights, quickstart

~/bin/llama-swap                                            # gateway binary
/etc/systemd/system/llama-swap.service                      # main unit
/etc/systemd/system/llama-swap.service.d/watch-config.conf  # hot-reload drop-in
~/.cache/huggingface/hub/                                   # model weights
~/.cache/huggingface/token                                  # HF auth, chmod 600
```

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
