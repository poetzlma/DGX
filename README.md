# Spark LLM Stack

Single OpenAI-compatible endpoint serving models on a DGX Spark (ThinkStation PGX, 192.168.1.12). The active config defines **two groups, simultaneously available** through the same gateway:

- **`main` group** (5 models, `swap: true, exclusive: true`) — heavy models hot-swapped one at a time: gemma-4-26b-a4b, minimax-m2.7, supergemma-4-26b, qwen3.5-35b-distill, qwen3.5-122b-nvfp4.
- **`qwen36` group** (2 models, `swap: false`) — co-resident pair that stays warm together: `qwen3.6-35b-a3b` (vLLM NVFP4, MoE 35B/3B-active) + `qwen3.6-27b` (llama.cpp, dense UD-Q4_K_XL). Workflow split in practice: the **27B dense is used for deep/reasoning work** (param density helps quality-per-token), and the **35B-A3B MoE handles parallel work on short contexts** (3B active params → high concurrent tok/s when many sub-agents hit it at once). This replaces the older Qwen-122B + Gemma-E4B "agent mode" pair — same two-models-always-warm pattern, different split.

## Hardware

- **NVIDIA DGX Spark (GB10)** — Grace/Blackwell SoC, compute capability 12.1 (SM 12.1 = GB10 chiplet, often misreported as SM120/Blackwell generic).
- **119 GB unified memory.** There is no separate VRAM. CPU and GPU read/write the same physical DRAM over the coherent NVLink-C2C fabric. When vLLM reports `CUDA0 buffer 96 GB`, that's 96 GB of the 119 GB pool — not extra memory. Every memory calculation here must treat "system RAM", "page cache", and "VRAM" as one shared budget.
- **FP4/FP8 hardware support** — native matmul kernels exist for NVFP4 and FP8 on Blackwell. We lean on these aggressively.
- **NVMe local storage** — single nvme0n1, capable of multi-GB/s sequential reads, but heavily bottlenecked by default mmap paths in llama.cpp (see below).

## Gateway

| Field | Value |
|---|---|
| Public URL | `http://192.168.1.12:8080/v1` |
| Gateway software | [llama-swap](https://github.com/mostlygeek/llama-swap) |
| Auth | none (trusted LAN) |
| Model-list endpoint | `GET /v1/models` |
| Currently-loaded | `GET /running` |
| Web UI | `http://192.168.1.12:8080/ui` |

### `main` group (swap-exclusive)

`swap: true, exclusive: true`, at most one model resident. Idle timeout `ttl: 3600` (1 h). Cold-swap latency 1–10 min depending on model. Requesting any member evicts the others.

### `qwen36` group (co-resident pair)

`swap: false`, both models stay loaded simultaneously. `ttl: 3600` (each member). Activating a `main`-group model will evict this pair because `main.exclusive: true`; next request to `qwen36` brings both back up.

| Model | Arch | Backend | Quant | GPU memory (nominal) | Primary use | Notes |
|---|---|---|---|---|---|---|
| `qwen3.6-27b` | Dense 27B | llama.cpp | UD-Q4_K_XL (GGUF) | ~18 GB weights + KV | **Thinking / deep reasoning** — higher param density per token, quality-over-throughput. Lower tok/s (~11 @c=1) but ~0.3s TTFT because llama.cpp streams `<think>` as it's generated. | 131k ctx (native 262k), `-fa on`, `--no-mmap`, q8 KV. |
| `qwen3.6-35b-a3b` | MoE 35B/3B-active | vLLM | NVFP4 | `--gpu-memory-utilization 0.55` (~65 GB) | **Parallel workers on short contexts** — MoE's low active-param count gives high aggregate tok/s when many sub-agents hit it concurrently. TTFT is higher (~8s) because `--reasoning-parser qwen3` buffers the `<think>` block before emitting. | MTP-1 speculative + FLASHINFER. Launcher at `bin/launch-vllm-qwen.sh` (see §25). |

Both route through the single `http://192.168.1.12:8080/v1` endpoint. Pick by model key.

### Switching between modes

You don't — they co-exist in one config. Request the model you want by key; llama-swap handles group eviction. The `config/llama-swap-swap.yaml` and `config/llama-swap-agent.yaml` files are **legacy templates from the pre-qwen3.6 era** and are not used by the running service. They're kept in-tree as reference snapshots of the old swap-only / 122B+E4B setups.

## Models

All entries are OpenAI-compatible, reached through the gateway at the same URL. Pick a model by setting `"model": "<key>"` in the request body. Group membership determines whether requesting one evicts others.

| Key | Group | Backend | Quant | Weights (GB) | Native ctx | Served ctx | Notes |
|---|---|---|---|---|---|---|---|
| `qwen3.6-35b-a3b` | qwen36 | vLLM | NVFP4 (on-disk) | ~23 | 262 k | 131 072 | MoE 35B/3B active. Used for parallel sub-agents on short contexts. MTP-1 + FLASHINFER. Launcher at `bin/launch-vllm-qwen.sh` (see §25). |
| `qwen3.6-27b` | qwen36 | llama.cpp | UD-Q4_K_XL (GGUF) | ~18 | 262 k | 131 072 | Dense. Used for thinking / deep reasoning. `-fa on`, `--no-mmap`, q8 KV. |
| `gemma-4-26b-a4b` | main | llama.cpp | Q4_K_M (GGUF) | ~15 | 256 k | 131 072 | MoE 26B/4B active. Switched from vLLM due to FP8 + attention bugs. |
| `qwen3.5-35b-distill` | main | vLLM | BF16 (on-disk) | ~72 | 262 k (arch) | 8 192 | Claude Opus distilled. MTP + FLASHINFER. Ctx capped at 8 k = SFT ceiling. |
| `qwen3.5-122b-nvfp4` | main | vLLM | NVFP4 (on-disk) | ~75 | 262 k | 65 536 | Largest model via vLLM. Unstable under sustained load at 0.80 util. |
| `minimax-m2.7` | main | llama.cpp | UD-IQ4_XS (unsloth) | ~60 | ~200 k | 65 536 | 230B/10B MoE. Changed from Q3_K_XL; KV cache q4_0. |
| `supergemma-4-26b` | main | llama.cpp | Q8_0 (multimodal) | ~26 | 256 k | 65 536 | Gemma 4 26B abliterated + mmproj vision. Context reduced from 131K; KV q4_0. |
| `gemma-4-e4b` | main | llama.cpp | Q8_0 (GGUF) | ~8 | 128 k | 131 072 | Gemma 4 E4B with vision (mmproj). Used as vision worker. |

Per-model flags live in `config/llama-swap.yaml` with comments explaining each choice.

### Historical benchmarks (2026-04-16, pre-qwen3.6 config)

Kept for reference. These are from the pre-qwen3.6 lineup (before the `qwen36` co-resident pair replaced the 122B-UD-Q4 + Gemma-E4B agent stack). Model keys reflect the config of that era — `qwen3.5-35b-a3b` was later replaced by `qwen3.6-35b-a3b` (NVFP4), `qwen3.5-122b` (llama.cpp) was retired, etc.

Swap mode, 2026-04-16:

| Model | Cold start | Memory | tok/s (e2e) | TTFT | Decode tok/s |
|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` | 238 s | 102 GB | 62.2 | 8.2 s * | ~instant (MTP buffered) |
| `gemma-4-26b-a4b` | 79 s | 21 GB | 58.3 | 5.3 s * | 231 |
| `supergemma-4-26b` | 67 s | 32 GB | 44.0 | 124 ms | 46 |
| `qwen3.5-35b-distill` | 209 s | 107 GB | 42.4 | 8.8 s * | 156 (MTP) |
| `gemma-4-e4b` | 47 s | 13 GB | 37.9 | 70 ms | 39 |
| `minimax-m2.7` | 73 s | 108 GB | 23.6 | 9.9 s * | 87 |
| `qwen3.5-122b-nvfp4` | 613 s | 110 GB | 15.3 | 33 s * | unstable |

\* TTFT includes hidden thinking time (`--reasoning-parser qwen3` or model thinking behavior).

Old agent mode (Qwen 122B + Gemma E4B), 2026-04-16:

| Model | tok/s (single) | tok/s (both active) | Concurrency scaling |
|---|---|---|---|
| `qwen3.5-122b` | 21.7 | 19.0 | 3 sub-agents: 19.7 agg tok/s |
| `gemma-4-e4b` | 39.9 | 36.6 | Vision: working (70ms TTFT) |

### qwen3.6 pair spot-check (2026-04-23)

Ad-hoc bench of `qwen3.6-27b` (llama.cpp GGUF, solo — MoE not co-resident at the time) on `max_tokens=400` reasoning prompts:

| Concurrency | Aggregate tok/s | TTFT p50 | TTFT p95 | Per-req tok/s |
|---|---|---|---|---|
| 1 | 11.07 | 0.32 s | 0.32 s | 11.07 |
| 4 | 29.51 | 0.65 s | 0.65 s | 7.47 |
| 10 | 25.34 | **55.55 s** | **109.90 s** | 7.47 |

Throughput regresses from c=4 → c=10 because llama.cpp's slot count can't keep up. Request queueing rather than real batching dominates at c≥5. This is fine for the 27B's actual role (deep/thinking work) — the pair is designed so parallel sub-agent traffic lands on the 35B MoE and deep-reasoning calls land on the 27B dense, not the other way around.

## Technical decisions

### 1. Co-resident pair vs swap-exclusive

Running two models concurrently on Spark means splitting 119 GB between them. For anything at MiniMax (108 GB) or 122B-NVFP4 (110 GB) scale, that's untenable — the `main` group's `exclusive: true` gives each model the entire pool.

The `qwen36` pair works because the NVFP4 35B-A3B (~23 GB weights, ~65 GB nominal reservation at util=0.55) + the 27B UD-Q4_K_XL GGUF (~18 GB weights + KV) fit together in 119 GB. Unlike the old 122B+E4B pair (both llama.cpp), the new pair is **mixed**: vLLM for the 35B (pre-allocates via `--gpu-memory-utilization`), llama.cpp for the 27B (dynamic). The vLLM side's pre-allocation sets the memory ceiling; the llama.cpp side takes whatever's left.

### 2. KV cache quantization is mandatory, not optional

Because weights + KV cache share the unified 119 GB, KV cost at long context eats the memory budget.

- **vLLM entries**: `--kv-cache-dtype fp8` — ~lossless on Blackwell, halves KV memory.
- **llama.cpp entries**: `--cache-type-k q8_0 --cache-type-v q8_0` or `q4_0` for tighter budgets. Q8 is indistinguishable from fp16; q4_0 is used on SuperGemma, MiniMax, and agent-mode Qwen 122B where memory headroom matters.

### 3. Qwen3.5 is 262 k native — no YaRN

Earlier configs applied `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`. **Those are Qwen3 params, not Qwen3.5.** Qwen3.5 ships with 262 k native context baked into `config.json`. Applying YaRN on top scales *every* request (static YaRN) and degrades short-context quality. We removed all `--rope-scaling` for Qwen3.5 entries.

### 4. Jackrong distill is capped at 32 k

The HF model card specifies SFT was performed at 8 192 tokens. The underlying architecture inherits Qwen3.5's 262 k positional embeddings, but the distilled behavior was only trained under ≤8 k. We cap at 32 k (4× SFT length, still within benign extrapolation) and recommend the base `qwen3.5-35b-a3b` for anything genuinely long.

### 5. MiniMax M2.7 is full-attention, not linear

The earlier MiniMax-Text-01 used hybrid lightning/linear attention (KV cost sub-linear in context). **The M2 series reverted to full softmax attention** per MiniMax's own engineering post. KV cost scales linearly with context, so at 60 GB weights (UD-IQ4_XS) + remaining budget, ctx is tightly bounded. With q4_0 KV + flash attention we can safely run 64 k.

### 6. `--no-mmap` for llama.cpp on Spark

With default `mmap = true`, llama.cpp memory-maps the GGUF and copies tensors to CUDA one at a time via synchronous `cudaMemcpy`. Each copy blocks the next page fault, so NVMe only hits ~200 MB/s at ~17% utilization (observed). `--no-mmap` replaces this with `pread()` streaming, ~10× faster on Spark's 6.17 kernel (community-verified on NVIDIA forums).

### 7. `--load-format fastsafetensors` — considered, not applied

vLLM's default safetensors loader is single-threaded Python (GIL-bound), loading one shard at a time. `fastsafetensors` (pip package) issues large parallel reads and cuts cold load times by 10×+. **Not currently enabled** — the `cu130-nightly` image may or may not ship the package in any given daily build, and the flag hard-fails startup when missing. Warm-cache shard loads complete in ~80 s without it. Revisit if cold-start times become the bottleneck.

### 8. `-fa` flash attention for llama.cpp

Standard best practice on any CUDA-capable llama.cpp build. Reduces attention intermediate memory and improves decode throughput. Always on for our llama.cpp entries.

### 9. `VLLM_FLASHINFER_MOE_BACKEND=latency` env

The default FlashInfer MoE backend (`throughput`) emits SM120-generic kernels that misbehave on SM12.1 (Spark's GB10 chiplet). The community-tested workaround — per the avarok/vllm-dgx-spark project — is to force the latency backend, which uses a compatible kernel path. Set on every vLLM entry except `qwen3.5-122b-nvfp4` (which uses `--moe-backend flashinfer_cutlass` CLI flag instead — see §17).

### 10. `--enable-prefix-caching` for vLLM

Single-user workloads reuse the same system prompts repeatedly. Prefix caching keeps computed KV for identical prefixes across requests, eliminating redundant prefill. TTFT drops significantly on repeated conversations. Applied to all vLLM entries.

### 11. Container cleanup in systemd

llama-swap uses `docker run --rm` to ensure backends auto-clean on normal exit. But an unclean llama-swap shutdown (crash, `systemctl kill`) leaves orphan containers holding GPU memory reservations. We added `ExecStartPre` and `ExecStopPost` hooks to the systemd unit that run `docker ps -aq --filter name=llama- | xargs -r docker rm -f` to catch this case.

### 12. Service enabled at boot

`systemctl enable llama-swap` — the service was `disabled` at install time, so the gateway silently stayed down after reboots. Now starts automatically.

### 13. vLLM image must be `cu130-nightly` for Qwen3.5 MoE

Published image `vllm/vllm-openai:gemma4-cu130` ships vLLM 0.14-branch, which does **not** register `Qwen3_5MoeForConditionalGeneration`. The required class landed in vLLM 0.16+; we pull `vllm/vllm-openai:cu130-nightly` for all vLLM entries.

### 14. `--max-num-batched-tokens ≥ block_size` for Qwen3.5 hybrid Mamba

Qwen3.5 MoE is a **hybrid attention + SSM** architecture. With `--enable-prefix-caching`, vLLM enforces `block_size ≤ max_num_batched_tokens`. Default `max_num_batched_tokens` is 2048 but vLLM resolves `block_size = 2096` for Qwen3.5 — assertion fires. Fix: `--max-num-batched-tokens 8192` on all Qwen3.5 entries.

### 15. First-run cold load can exceed `VLLM_ENGINE_READY_TIMEOUT_S`

vLLM's APIServer has a 600 s default deadline. On first load with uncached weights, the HF download exceeds that. Every vLLM entry sets `-e VLLM_ENGINE_READY_TIMEOUT_S=1800`.

### 16. `--language-model-only` for the Qwen3.5 MoE family

Qwen3.5 MoE repos declare a vision tower in `config.json`, and vLLM loads those weights by default. `--language-model-only` skips the vision branch. Applied to all Qwen3.5 entries.

### 17. `--moe-backend flashinfer_cutlass` for NVFP4 MoE on GB10

FlashInfer's default MoE backend attempts SM 8.0/Ampere tactics that GB10 doesn't support. RedHat's own NVFP4 README sets `--moe-backend flashinfer_cutlass` (a CLI flag, not the env var). Applied to `qwen3.5-122b-nvfp4`.

### 18. `--reasoning-parser qwen3` for Qwen3.5 chat completions

Qwen3.5 injects `<think>` blocks. Without a parser, OpenAI clients see raw `</think>` in `message.content`. The parser routes thinking to `message.reasoning_content`. Applied to all Qwen3.5 entries. **Note**: this means TTFT includes hidden thinking time — the model reasons silently for seconds before producing visible output.

### 19. Tokenizer-class patch: `"TokenizersBackend"` → `"Qwen2TokenizerFast"`

Two Qwen3.5 repos ship `tokenizer_class: "TokenizersBackend"` (transformers v5.x). vLLM is pinned to <5 and raises a ValueError. Workaround: in-place patch of cached `tokenizer_config.json`:

```bash
for model in \
  "Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled" \
  "RedHatAI--Qwen3.5-122B-A10B-NVFP4"; do
  f=$(find ~/.cache/huggingface/hub/models--${model} -name tokenizer_config.json -path "*/snapshots/*" | head -1)
  [ -f "${f}.bak" ] || cp "$f" "${f}.bak"
  python3 -c "import json; d=json.load(open('$f')); d['tokenizer_class']='Qwen2TokenizerFast'; json.dump(d, open('$f','w'), indent=2, ensure_ascii=False)"
done
```

**`hf download --force` will revert the patch** — re-apply if weights are re-fetched.

### 20. MTP speculative decoding for Qwen3.5 / Qwen3.6

Both Qwen3.5 and Qwen3.6 MoE ship native Multi-Token Prediction (MTP) weights. MTP-1 predicts 1 additional token per step with high acceptance rate. Added `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to `qwen3.6-35b-a3b` (the RedHatAI NVFP4 variant keeps the MTP head) and `qwen3.5-35b-distill`. **Requires `--attention-backend FLASHINFER`** — MTP silently disables without it. Benchmark showed 156 tok/s pure decode on the distill model (up from 29 tok/s e2e without MTP). The 122B NVFP4 cannot use MTP — the RedHatAI checkpoint for that model stripped the MTP head during quantization.

### 21. `--attention-backend FLASHINFER` for vLLM on Spark

vLLM may default to FLASH_ATTN on SM121. FlashInfer has optimized kernels for the Blackwell memory hierarchy (+16% per albond benchmarks). Applied to all vLLM entries. Also required for MTP to function (see §20).

### 22. Gemma 4 switched from vLLM to llama.cpp

Both Gemma 4 models (26B-A4B and E4B) were switched from vLLM to llama.cpp due to two compounding vLLM bugs:

1. **vLLM #39407/#39049**: `--quantization fp8` produces garbage output on ALL Gemma 4 variants. FP8 checkpoints have activation scales pre-absorbed, but vLLM still applies dynamic per-token activation quantization on top, causing logit saturation.
2. **vLLM #38887**: Gemma 4's heterogeneous head dimensions (256 sliding-window, 512 global attention) force TRITON_ATTN fallback, causing 10-15x slowdown.

llama.cpp has no kernel compatibility issues on SM121. Results: Gemma E4B went from 18.5 → 37.9 tok/s (+105%), Gemma 26B from 37.7 → 58.3 tok/s (+55%), with 75-83% less memory.

### 23. `--parallel N` splits context across slots

llama.cpp's `--parallel N` creates N concurrent request slots but **divides the total `-c` context evenly**. So `-c 131072 --parallel 4` gives 32K per slot, not 131K. Weight reads ARE batched across slots (real batching, not sequential), but each slot gets `total_ctx / N` tokens.

For agent mode: `-c 262144 --parallel 4` = 64K per slot (reasoning), `-c 65536 --parallel 4` = 16K per slot (worker).

### 24. Vision via mmproj on Gemma E4B

Gemma 4 E4B supports multimodal input via the `--mmproj` flag in llama.cpp. The mmproj file (`mmproj-gemma-4-E4B-it-Q8_0.gguf`) adds ~200-500 MB overhead. Send images via the standard OpenAI `image_url` content type.

Qwen3.5-122B does NOT support vision via llama.cpp — the CLIP graph uses unsupported operators (llama.cpp issue #21268).

### 25. `qwen3.6-35b-a3b` launcher script (drop `--rm` for crash traceability)

The MoE has been observed silently crashing under sustained AsyncOpenAI workload: vLLM exits with status 0, llama-swap logs `<qwen3.6-35b-a3b> process exited but not StateStopping`, and auto-restarts. With `docker run --rm`, the dead container was wiped before we could read `docker logs`, so every crash was opaque.

Two issues to fix:

1. Drop `--rm` on the MoE container so logs survive the crash.
2. Clean up the named (now persistent) container before respawning, or `--name vllm-qwen` collides on the next start.

llama-swap parses `cmd:` with **shellwords** (no shell invocation), so compound commands like `docker rm -f X 2>/dev/null; exec docker run …` can't live inline — they'd be passed as literal argv to docker. The fix is to move the launcher into a script file:

```
bin/launch-vllm-qwen.sh   →   docker rm -f vllm-qwen 2>/dev/null || true
                              exec docker run --name vllm-qwen … (no --rm)
```

And point the yaml at it:

```yaml
qwen3.6-35b-a3b:
  cmd: /home/max/llm-stack/bin/launch-vllm-qwen.sh
```

Now: `docker inspect vllm-qwen --format '{{.HostConfig.AutoRemove}}'` → `false`, and `docker logs vllm-qwen` works on the dead container after the next crash.

Apply the same pattern to any other vLLM container where post-mortem logs matter.

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Logs (llama-swap itself)
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Logs (active backend container)
docker logs -f vllm-qwen                 # qwen36 group: qwen3.6-35b-a3b (NO --rm, survives crashes — see §25)
docker logs -f llama-qwen-27b            # qwen36 group: qwen3.6-27b
docker logs -f vllm-qwen-distill         # main group: qwen3.5-35b-distill
docker logs -f vllm-qwen-nvfp4           # main group: qwen3.5-122b-nvfp4
docker logs -f llama-gemma-26b           # main group: gemma-4-26b-a4b
docker logs -f llama-gemma-e4b           # main group: gemma-4-e4b
docker logs -f llama-minimax             # main group: minimax-m2.7
docker logs -f llama-supergemma          # main group: supergemma-4-26b

# Current state
curl http://192.168.1.12:8080/running     # which model is hot
curl http://192.168.1.12:8080/v1/models   # model list
docker ps --filter name=llama- --filter name=vllm-

# Force an unload without stopping the service
curl -X POST http://192.168.1.12:8080/unload

# Switch modes (see "Switching modes" above)
```

Cold-swap behaviour (swap mode): the first call after an idle timeout or after hitting a different model triggers a container spin-up. Expect 1–10 min. LiteLLM or any wrapping client should set a per-request timeout ≥ 900 s.

## Benchmarking

### Quick benchmark (swap mode)

`bin/bench-models.py` walks every model in the swap group and measures cold_s, peak_mem_gb, model_gb, and tok/s on a 256-token completion.

```sh
python3 bin/bench-models.py | tee logs/bench-$(date +%Y%m%d-%H%M).log
```

### Deep benchmark

`bin/bench-deep.py` provides richer profiling: TTFT (time to first token via SSE streaming), decode tok/s (separated from prefill), multi-prompt tiers (short/medium/long prefill), concurrency scaling (conc=3, conc=5), and 3-pass mean±stdev. Accepts optional model names as args.

```sh
# All models
python3 bin/bench-deep.py | tee logs/bench-deep.log

# Specific models only
python3 bin/bench-deep.py qwen3.5-35b-a3b gemma-4-e4b

# Compare against old baseline
python3 bin/bench-compare-deep.py logs/bench-results-old-20260414.json logs/bench-deep-latest.json
```

Results saved to timestamped `logs/bench-deep-YYYYMMDD-HHMM.json` with symlink at `logs/bench-deep-latest.json`. Expected runtime: 55-65 min for all 7 models.

## Adding a model

1. **Check the cache first:** `ls ~/.cache/huggingface/hub/ | grep -i <name>`.
2. **Download if missing:** `~/llm-stack/venv/bin/hf download <org>/<repo>`. Requires `max:max` ownership on `~/.cache/huggingface/hub`.
3. **Add a block under `models:`** in the appropriate config:
   - safetensors (BF16, FP8, NVFP4): use vLLM pattern (see `qwen3.5-35b-a3b`).
   - GGUF: use llama.cpp pattern (see `gemma-4-26b-a4b`).
4. **Append the new key** to the relevant group members list.
5. **Validate**: `python3 -c "import yaml; yaml.safe_load(open('config/llama-swap.yaml'))"`.
6. **Restart**: `kill -9 $(pgrep -f "llama-swap -config")` (systemd restarts automatically).
7. **Smoke test** by sending a 5-token completion — first request is the cold load.

## LiteLLM integration

LiteLLM points at the gateway, one entry per model. Minimum fields:

```yaml
model_list:
  - model_name: qwen3.6-35b-a3b          # what clients call
    litellm_params:
      model: openai/qwen3.6-35b-a3b      # must match the llama-swap map key
      api_base: http://192.168.1.12:8080/v1
      api_key: none
      timeout: 900                        # ≥ cold-load time
    model_info:
      max_input_tokens: 131072           # exposes ctx limit to OpenWebUI
```

The `openai/<key>` string in `litellm_params.model` must exactly match the key under `models:` in the active config — that's how llama-swap routes and decides which backend to swap in.

## Files

```
~/llm-stack/
├── config/
│   ├── llama-swap.yaml             # active config: main group (solo vLLM 27B NVFP4+MTP since 2026-04-24)
│   ├── llama-swap-swap.yaml        # legacy snapshot: pre-qwen3.6 swap-only (stale)
│   ├── llama-swap-agent.yaml       # legacy snapshot: 122B + E4B agent pair (stale)
│   └── llama-swap-bench.yaml       # benchmark variant
├── bin/
│   ├── launch-vllm-qwen.sh         # launcher for qwen3.6-35b-a3b MoE (DORMANT — on-demand via llama-swap)
│   ├── launch-vllm-27b-nvfp4.sh    # launcher for qwen3.6-27b vLLM NVFP4+MTP (active)
│   ├── bench-models.py             # quick benchmark (cold start + tok/s)
│   ├── bench-deep.py               # deep benchmark (TTFT, decode, concurrency)
│   └── bench-compare-deep.py       # before/after comparison report
├── venv/                           # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.log              # gateway stdout
│   ├── llama-swap.err              # gateway stderr
│   ├── bench-results.json          # quick benchmark results
│   ├── bench-deep-latest.json      # latest deep benchmark (symlink)
│   └── bench-deep-*.json           # timestamped deep benchmark runs
└── README.md

~/bin/llama-swap                     # gateway binary (v201)
/etc/systemd/system/llama-swap.service   # with ExecStartPre cleanup hook
~/.cache/huggingface/hub/            # model weights (max:max ownership required)
~/.cache/huggingface/token           # HF auth, chmod 600
```

## MTP patch (AlphaOxO Qwen3.6-27B-NVFP4)

The AlphaOxO NVFP4 repo ships the MTP weights (`model_mtp.safetensors`) but strips
`num_nextn_predict_layers` from `config.json`. Without that field, vLLM silently loads
the model with MTP disabled — you lose the 1.6–2× speculative throughput gain and
`--speculative-config` in the launcher becomes a no-op.

Re-apply after every `hf download` (or any time the HF cache is cleared):

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

**Verify** by grepping vLLM startup logs for `Detected MTP model. Sharing target model
embedding weights with the draft model.` — if present, MTP is wired. If absent,
speculation is silently off.

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| Gateway down after reboot | Service not enabled | `sudo systemctl enable llama-swap` |
| 502 with `upstream command exited prematurely` | Orphan container holds port/VRAM | `docker rm -f llama-<name>`; systemd unit handles via `ExecStartPre` |
| SIGKILL leaves orphan container | llama-swap's `healthCheckTimeout` kill doesn't trigger service cleanup | `docker rm -f <name>` manually; bump `healthCheckTimeout` |
| CUDA OOM on backend start | Previous container still resident | same as above; verify with `docker ps` |
| 122B NVFP4 fails with `Free memory ... less than desired GPU memory utilization` | Residual memory from prior models on unified memory | Lower `--gpu-memory-utilization` to 0.80, or use the llama.cpp GGUF in agent mode instead |
| Cold load 5× slower than expected | Default mmap path on Spark | ensure `--no-mmap` (llama.cpp) or `--load-format fastsafetensors` (vLLM) |
| 502 on first call, fine after | Model still loading | wait; watch `docker logs` for startup complete |
| Qwen3.5 exits `status 1` within ~25 s | Image vLLM too old for Qwen3.5 arch | switch to `vllm/vllm-openai:cu130-nightly` |
| `AssertionError: In Mamba cache align mode` | Qwen3.5 + prefix caching needs larger batch | `--max-num-batched-tokens 8192` |
| `TimeoutError: VLLM_ENGINE_READY_TIMEOUT_S` | 600 s default too short for large models | `-e VLLM_ENGINE_READY_TIMEOUT_S=1800` |
| `ValueError: Tokenizer class TokenizersBackend` | Repo exported on transformers v5 | patch cached `tokenizer_config.json` to `"Qwen2TokenizerFast"` (see §19) |
| `OSError: Can't load image processor` | vLLM treats Qwen3.5 as multimodal; repo missing preprocessor | `--language-model-only` (see §16) |
| Flood of `Skipping tactic` / `Failed to initialize cutlass` | FlashInfer MoE autotuner on GB10 | expected, non-fatal. For 122B-NVFP4 use `--moe-backend flashinfer_cutlass` |
| `--parallel N` gives less context than expected | llama.cpp divides `-c` total across N slots | set `-c` to `desired_per_slot × N` (see §23) |
| Gemma 4 garbage output via vLLM FP8 | vLLM #39407: double-quantization bug | use llama.cpp with GGUF instead (see §22) |
| Gemma 4 extremely slow via vLLM | vLLM #38887: heterogeneous head dims force Triton fallback | use llama.cpp with GGUF instead (see §22) |
| Vision not working on Qwen 122B | llama.cpp CLIP graph unsupported ops (#21268) | use Gemma E4B for vision tasks instead |
