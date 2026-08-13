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

- **`resident`** (`swap: false, exclusive: false, persistent: true`) — since 2026-08-01 a **single** always-on model, `deepseek-v4-flash-0731` (~86 GB of the 121 GB pool, a native binary rather than a container). `ttl: 0` — no idle unload, plus an `on_startup` preload hook so it loads at service start. `healthCheckTimeout: 1800` covers its cold start. (Previous eras are preserved as config backups: `llama-swap.yaml.bak.20260801-prelaguna-swap` (solo laguna), `…20260722-prelaguna` (the `nemotron-3-puzzle-75b` + `qwen3.6-35b-a3b-vision` pair split `0.55`/`0.28`), and `llama-swap.full.bak` (full roster).)
- **`experiments`** (`swap: true, exclusive: true`) — the dormant pool. **At most one** member loads at a time; requesting a different one unloads the current and cold-loads the next. Each is a [copy-back model](#weight-offload--codeserver-copy-back): its weights rsync from codeserver first (+1–3 min, ~13 min for ds4), then the engine cold-loads (1–10 min vLLM, ~30–105 s for llama.cpp/ds4 binaries). `ttl: 3600`.

> **⚠️ Swap-exclusive vs. itself, not vs. residents.** The `experiments` group only unloads *other experiments members* — it never evicts the resident. The ds4 resident holds ~86 GB of 121 GB, so a >30 GB neighbour will not fit, and **the GB10 hard-hangs on host OOM** with no remote recovery (it has, twice). Do not co-load by hand: **park the resident** with `bin/park-prod-ds4.sh` (points the live `cmd:` at `bin/eval-window-blocked.sh` *and* clears the `on_startup` preload, so neither a stray request nor a config reload can spawn a second engine), run the eval, then `bin/restore-prod-ds4.sh`. See [decision §37](decisions.md#37-ds4-quant-eval-the-upstream-host-registration-oom-hard-hang-2026-08-02) and [Troubleshooting](#troubleshooting).

> **⚠️ Active config is in LOCKED MODE.** `config/llama-swap.yaml` (the live config) currently exposes **only `deepseek-v4-flash-0731`** — the dormant pool is not loadable from it. The full roster lives in `config/llama-swap.full.bak`; restore it (`cp config/llama-swap.full.bak config/llama-swap.yaml`, `-watch-config` picks it up) to re-enable the copy-back models — but note it still declares the **pre-laguna** resident pair (`qwen3.6-35b-a3b-vision` + `nemotron-3-puzzle-75b`), and the 75B's weights were deleted 2026-08-02, so restoring it wholesale would put a re-download in the startup path. **Editing the live yaml bounces the resident** (~10 min reload) — the preload hook + cron watchdogs auto-recover, but treat any yaml write as a production restart.

---

## Weight offload — codeserver copy-back

The Spark's 916 GB NVMe was filling (95%). Dormant model weights now live **off-box on codeserver** (`192.168.1.16`, a 1.9 TB LAN host reached over 2.5 GbE) under `~/llm-weights-archive/` (~409 GB), and are **pulled back on demand** the moment llama-swap starts that slot — dropping the Spark to **~48% used** at the time (85% today, after the 2026-08 ds4 quant evals staged ~170 GB of GGUFs locally). Only the resident's weights plus the keep-local set stay permanently on NVMe.

**How a dormant model loads.** Each `experiments` slot's `cmd:` in `llama-swap.full.bak` is wrapped by [`bin/copyback-launch.sh`](../bin/copyback-launch.sh):

```
copyback-launch.sh <local_path> <remote_relpath> <real_launch_script>
```

On start it (1) **evicts every other managed model** listed in `etc/copyback-models.txt` — enforcing *one dormant model on NVMe at a time* and self-healing if a prior slot was SIGKILL'd before it could clean up; (2) **rsyncs** the weights from `codeserver:~/llm-weights-archive/<remote_relpath>` if they aren't already local (`--partial` resumes an interrupted pull; a failed pull is removed, not left corrupt); (3) runs the real launcher with `TERM`/`INT` forwarded; (4) on stop, **evicts** the weights again (evict-immediately-after-use — leanest disk, re-pulls each cold start).

**The manifest** `etc/copyback-models.txt` is the eviction guard: one absolute weight path per line. **Keep-local paths are deliberately absent** so they can never be evicted — the resident's GGUFs under `~/ds4/gguf` and `~/entrpi-gguf` (removed from the manifest 2026-08-13, when it became clear a copy-back launch would have evicted the engine-rollback weights), the `qwen3.6-35b-a3b-vision` weights, and the z-lab DFlash drafter. **Anything promoted to resident must be taken off this list in the same change.**

**Archive layout** (`codeserver:~/llm-weights-archive/`):

```
hub/models--<org>--<name>   # HF-cache models → restore to ~/.cache/huggingface/hub/
qwopus/Qwopus3.6-27B-v2-int4-AutoRound
ornith/ornith-1.0-35b
ds4/gguf                    # ~85 GB — kept in the archive, but no longer
                            # copy-back-managed: ds4 is the resident since
                            # 2026-08-01 and its weights stay local
cosmos3-models/Cosmos3-Nano
```

**Operational notes:**

- `healthCheckTimeout` **must exceed** the pull+load time, or llama-swap kills the slot mid-download. It is `1800` (30 min) in the live config — raised from 1200 for the ds4 resident's cold start; `llama-swap.full.bak` still says 1200, which is enough for every copy-back lane.
- **New dependency:** dormant models require **codeserver online** to load — a failed pull exits non-zero and the slot won't start. The resident has no such dependency. Weights exist *only* on codeserver now (local copies were deleted after verification).
- The pull path uses SSH key auth to a bare host (no credentials in-repo). The codeserver address defaults to the LAN IP but is overridable via the **`COPYBACK_REMOTE`** env var (and `COPYBACK_ARCHIVE_ROOT` for the archive path) in `bin/copyback-launch.sh` — set it in the systemd env or launcher if you'd rather not hardcode internal topology.


## Runbook

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Gateway logs
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Resident engine (always loaded) — a NATIVE BINARY, not a container, so there
# are no `docker logs` for it: its stdout/stderr land in the gateway log above.
ps -eo pid,etime,cmd | grep '[d]s4-server --cuda'
curl -s http://127.0.0.1:9010/v1/models          # readiness (ds4 has no /health)
curl -s http://127.0.0.1:9010/metrics | grep -E 'ds4_requests_total|ds4_requests_inflight'

# Watchdog state (both cron-installed; see Troubleshooting)
tail -f ~/llm-stack/logs/ds4-keepalive.log           # engine GONE          (*/5)
tail -f ~/llm-stack/logs/ds4-degraded-watchdog.log   # engine UP but REFUSING (*/2)

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

### Roll the coding default back

```sh
# TIER 1 — ENGINE ROLLBACK, one line, no download (Entrpi fork -> antirez mainline):
#   in config/llama-swap.yaml point the deepseek-v4-flash-0731 cmd: at
#   bin/launch-ds4-server.sh. Same weights (hard-linked GGUF), different binary.
#   Costs ~28% decode, gains 2.33x prefill and ~27s boot. The yaml edit IS the
#   cutover (live-watched, ~10 min bounce). See decisions §39/§40.
#   Each engine keeps its OWN --kv-disk-dir, so the prefix cache goes cold on
#   the switch and refills; that is expected, not a fault.
#
# TIER 2 — WEIGHTS ROLLBACK: bin/launch-ds4-server.sh.bak.20260801-preview
#   restores the pre-0731 preview quant, BUT that GGUF was evicted on 2026-08-02
#   — it is an ~87 GB re-download now, not a script swap.
#
# TIER 3 — LEAVE ds4 ENTIRELY (back to laguna): restore
#   config/llama-swap.yaml.bak.20260801-prelaguna-swap, swap the keepalive cron
#   back to bin/laguna-keepalive.sh, AND re-download poolside/Laguna-S-2.1-NVFP4
#   (67 GiB, revision 07614121 — deleted 2026-08-02). Budget ~1 h. The DFlash
#   drafter (4.2 GiB) is still cached. Know why you are doing it: laguna was
#   pulled for planning without committing to an action, not for speed (§36).
#
# TIER 4 — further back (dense Qwen 27B): repoint the gateway names at
#   qwen3.6-27b-int4-dflash. It is a copy-back model, so its first load pulls
#   ~18 GB from codeserver (~2 min) — pre-stage by hitting it once.
#
# WHERE to change route names: llama-swap v201 drops yaml aliases on
# -watch-config reload, so the authoritative alias->model map lives at the
# LiteLLM gateway (deployed.yaml on cockroach / 192.168.1.7), NOT in
# config/llama-swap.yaml. All six ds4-backed names are mapped there, each with
# its own price block — a route repointed at a different engine keeps the OLD
# price until you fix it (§38).
```

## LiteLLM integration

Points at the log-proxy on `:8079` so every request lands in `~/llm-stack/logs/proxy/{date}/{model}/`. `deployed.yaml` in this repo is a drop-in `model_list` for LiteLLM. Minimum per-entry:

```yaml
model_list:
  - model_name: qwen3.6-27b                       # what clients call (legacy alias)
    litellm_params:
      model: openai/deepseek-v4-flash-0731        # resolve the alias to the real key HERE
      api_base: http://192.168.1.12:8079/v1
      api_key: none
      timeout: 1200                               # ≥ cold-load + copy-back pull
      input_cost_per_token: 0.0000001             # price EVERY alias, not just the real key
      output_cost_per_token: 0.0000004            #   (§38 — an unpriced route reports $0)
litellm_settings:
  request_timeout: 1200
```

**Key scopes are part of the contract — and `/v1/models` lies about the roster.** LiteLLM filters `GET /v1/models` by the calling key's `models` allowlist, so a client (or an agent debugging one) sees *its own scope*, not what the gateway serves. A key scoped to two routes reports two models at `<gateway-host>` while the gateway is exposing 17 — which reads exactly like "the canonical route isn't published," and isn't. Probe with an unscoped key before concluding a route is missing.

Consequently **every entry in a key's allowlist must be a route that can actually be served.** In locked mode that is only the six ds4-backed names (`deepseek-v4-flash-0731`, `deepseek-v4-flash-ds4`, `laguna-s-2.1`, `nemotron-3-puzzle-75b`, `qwen3.6-27b`, `qwen3.6-35b-a3b`) — anything else 404s at llama-swap even though the gateway accepts it, so a stale allowlist is a menu of failures. Audited and cleaned 2026-08-13: six keys carried dead routes (`qwen3.6-35b-a3b-vision`, dark since 07-22; `deepseek-v4-flash` retired 06-27; `deepseek-v4-pro`, which never existed; and the whole Qwen3.5/gemma/minimax generation), and none carried the canonical resident name. Audit with:

```sh
# on cockroach (.7) — flags allowlist entries that are not servable
curl -s -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  'http://127.0.0.1:4000/key/list?return_full_object=true&size=100' | jq -r \
  '.keys[] | select((.models // []) - ["deepseek-v4-flash-0731","deepseek-v4-flash-ds4","laguna-s-2.1","nemotron-3-puzzle-75b","qwen3.6-27b","qwen3.6-35b-a3b","all-team-models"] | length > 0) | "\(.key_alias): \(.models)"'
```

Back up before editing (`/key/list` → `infra/backups/`, `chmod 600`); `POST /key/update` accepts the hashed token from that listing, and `models` is **replaced**, not merged. Verify with a throwaway scoped key (`/key/generate` with `duration`, then `/key/delete`) rather than by asking a paying client to retry.

**Billing check after any gateway change:** query for rows with `spend = 0 AND prompt_tokens > 0`. Pricing silently vanished for two months (2026-06-01 → 08-03, backfilled +$60.04), and because six route names resolve to one engine, pricing only the real key reports $0 for every alias clients actually call ([decision §38](decisions.md#38-gateway-pricing-for-the-ds4-lane-the-silent-zero-spend-window-2026-08-03)). Never `docker restart litellm` on cockroach — its baked `--apply-only` command crash-loops; use `docker compose up -d --force-recreate litellm`.

The `openai/<key>` string in `litellm_params.model` must exactly match the key under `models:` in the active llama-swap config — that's how llama-swap routes and decides which backend to swap in. **Resolve aliases at this gateway layer** (map `qwen3.6-27b` → the real model name): llama-swap v201 drops yaml `aliases:` on `-watch-config` reload, so the LiteLLM `model_list` is the reliable place to pin them.


## Files

```
~/llm-stack/
├── config/
│   ├── llama-swap.yaml             # ACTIVE config — LOCKED MODE (resident only)
│   └── llama-swap.full.bak         # full roster (resident + experiments groups) — restore to unlock
├── deployed.yaml                   # LiteLLM model_list for downstream (ds4 routes + pricing, updated 2026-08-03)
├── bin/
│   ├── copyback-launch.sh              # copy-back/dormant-tier wrapper: rsync weights from codeserver, run, evict
│   ├── launch-ds4-entrpi.sh            # RESIDENT — coding default (Entrpi/ds4 v0.5.6.2, 0731 IQ2_XXS, --no-spec)
│   ├── launch-ds4-server.sh            # engine rollback — antirez mainline b030961, same weights (one-line swap)
│   ├── ds4-keepalive.sh                # cron */5 — relaunch the resident if the engine is GONE
│   ├── ds4-degraded-watchdog.sh        # cron */2 — reload it when UP but REFUSING (refusals up, completions flat)
│   ├── mem-watchdog.sh                 # SIGKILL a runaway engine before it hard-hangs the box (manual/eval use)
│   ├── park-prod-ds4.sh                # park the resident for an eval window (cmd -> blocked, preload cleared, cron off)
│   ├── restore-prod-ds4.sh             # undo park-prod-ds4.sh and warm the lane back up
│   ├── eval-window-blocked.sh          # placeholder cmd: a stray request cannot spawn a second ~90 GB engine
│   ├── smoke-ds4-0731.py               # correctness smoke for the ds4 lane (10 checks; gate for engine cutovers)
│   ├── bench-ds4-0731.py               # ds4 prefill+decode bench, streaming, context-swept
│   ├── bench-concurrency.py            # concurrency harness that does NOT assume parallel prefill
│   ├── probe-commits-to-action.py      # does the lane commit to an action, or plan forever? (why laguna was pulled)
│   ├── gguf-head.py                    # dump GGUF metadata from a PARTIAL download (chat template, tokenizer, recipe)
│   ├── pardl.py                        # resumable parallel-range downloader (~9-16 MB/s single -> ~33 MB/s)
│   ├── dl-{antirez-ds4-q4k-hybrid,unsloth-ds4-iq3xxs}.sh  # quant candidate fetchers
│   ├── dl-ds4-quants-chain.sh          # serialised fetch — egress is a hard ~32 MB/s ceiling, parallel just starves
│   ├── launch-ds4-server-0731.sh       # eval — 0731 weights on the pinned ngc-shj fork (the 08-01 test twin)
│   ├── launch-ds4-server-q4k-hybrid.sh # eval — Q4K-hybrid 0731 (97.6 GB), port 9099
│   ├── launch-ds4-upstream-q4k.sh      # eval — antirez upstream 54b36ed; GUARDED (ALLOW_OOM_RISK=1) after the OOM
│   ├── launch-llamacpp-ds4-unsloth-iq3xxs.sh # eval — unsloth UD-IQ3_XXS on llama.cpp (cannot load on ds4)
│   ├── ds4-test-window.sh              # open/close a ds4 test window by parking the (then) laguna resident
│   ├── run-ds4-0731-window.sh          # unattended window driver: load -> smoke -> bench -> stop
│   ├── launch-vllm-laguna-s21-nvfp4-v2.sh # retired default — Laguna S-2.1 v2 (weights deleted 2026-08-02)
│   ├── launch-vllm-laguna-s21-nvfp4.sh    # retired — Laguna v1 (original rotate weights, pinned b482b5d)
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
│   ├── decisions.md                     # decision log (§1–§41)
│   ├── benchmarks.md                    # bench tooling + historical archive
│   ├── qwen3.6-27b-dflash.md            # deep DFlash writeup
│   └── deepseek-v4-flash.md             # ds4 deep writeup
├── venv/                                # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.{log,err}             # gateway std{out,err} (resident engine's stdout lands here too)
│   ├── ds4-keepalive.log                # engine-gone watchdog
│   ├── ds4-degraded-watchdog.{log,state} # up-but-refusing watchdog (state = last counter snapshot)
│   ├── ds4-0731-window-<stamp>/         # unattended eval-window output (smoke + bench)
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
| **Resident answers `/v1/models` but every real request 503s in ~50 ms** | ds4 memory floor latched: demand-mapped KV slabs grew until `MemAvailable` fell under `--mem-floor-gb`, so deep admissions are unfundable and bounce to the 64 k deep-serial guard (§41) | `bin/ds4-degraded-watchdog.sh` (cron `*/2`) reloads the lane automatically — ~20 s, prefix cache survives. Confirm with `ds4_requests_total{outcome="refused_deep_serial"}` rising while `{outcome="completed"}` stays flat. **Never "fix" this by raising `--ctx`** |
| Resident vanished (crash, power cycle) | Engine died; the `on_startup` preload only covers service start | `bin/ds4-keepalive.sh` (cron `*/5`) pokes llama-swap to relaunch. Cold start ~10 min — the keepalive request carries a 1500 s timeout so a real customer doesn't pay for it |
| Box hard-hung, needs a power cycle | Host OOM — an engine's *real* footprint exceeded its planned one (§37: upstream ds4's host-registration fallback made a device-side copy and doubled a planned 93.9 GiB) | No remote recovery exists — prevention only. Keep `--mem-floor-gb`, run `bin/mem-watchdog.sh` during evals, and **park prod** (`bin/park-prod-ds4.sh`) before loading anything large: a cleared `on_startup` preload is why the box came back cleanly last time |
| Resident has no `docker logs` | It's a native binary (ds4), not a container | Its stdout/stderr go to `logs/llama-swap.log`; state via `curl :9010/v1/models` and `:9010/metrics` |
| Engine "up" but tokens frozen (`/health` still 200, requests hang) | vLLM engine-core deadlock — seen with DFlash drafter + `max-num-seqs 8` under real traffic (§34) | `docker rm -f vllm-laguna-s21` (llama-swap relaunches via preload/cron). Detect with a stall watchdog: `generation_tokens_total` unchanged for minutes while `num_requests_running > 0`. Keep `LAG_SEQS ≤ 4` with the drafter |
| Co-resident engines won't both load | Insufficient GPU memory | Lower `--gpu-memory-utilization` on one engine; verify with `nvidia-smi --query-compute-apps` |
| A heavy dormant model OOMs on cold start | The resident is holding ~86 GB of 121 GB | The `experiments` group is swap-exclusive vs itself but **not** vs the resident — anything over ~30 GB will not fit beside it. Park the resident (`bin/park-prod-ds4.sh`) rather than racing it; do **not** rely on the eval model's own guard |
| Dormant model won't load, `pull failed` in logs | codeserver (`192.168.1.16`) unreachable, or SSH key not loaded | `ssh 192.168.1.16 true` to check; dormant models require codeserver online. Residents are unaffected |
| Dormant model load times out | Pull + engine load exceeded `healthCheckTimeout` | Raise `healthCheckTimeout` in `llama-swap.yaml` (live config is 1800 s); the biggest copy-back pull is ~20 GB / ~3 min, but a cold vLLM load can add 10 min |
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
