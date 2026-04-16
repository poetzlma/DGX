# Spark LLM Stack

Single OpenAI-compatible endpoint serving models on a DGX Spark (ThinkStation PGX, 192.168.1.12). Two operating modes: **swap mode** (7 hot-swappable models, one at a time) and **agent mode** (Qwen 122B reasoning + Gemma E4B worker loaded concurrently for agentic workflows).

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

### Swap mode (default)

Config: `config/llama-swap-swap.yaml` — `swap: true, exclusive: true`, at most one model resident. Idle timeout `ttl: 3600` (1 h). Cold-swap latency 1–10 min depending on model.

### Agent mode

Config: `config/llama-swap-agent.yaml` — `swap: false, exclusive: false`, both models loaded simultaneously with `ttl: 0` (never unload). Designed for Hermes-style agents with sub-agents.

| Role | Model | Backend | Memory | Context/slot | Slots | Capabilities |
|---|---|---|---|---|---|---|
| Reasoning | `qwen3.5-122b` | llama.cpp UD-Q4_K_XL | 78 GB | 64K | 4 | Text, reasoning, code, tool calling |
| Worker | `gemma-4-e4b` | llama.cpp Q8_0 + mmproj | 15 GB | 16K | 4 | Text, vision, tool calling, 70ms TTFT |

Total: ~93 GB, 26 GB free. Both models serve concurrently via the same gateway.

### Switching modes

```sh
# Switch to agent mode
cp config/llama-swap-agent.yaml config/llama-swap.yaml
kill -9 $(pgrep -f "llama-swap -config")   # systemd restarts with new config

# Switch to swap mode
cp config/llama-swap-swap.yaml config/llama-swap.yaml
kill -9 $(pgrep -f "llama-swap -config")
```

## Models (swap mode)

All entries are OpenAI-compatible, reached through the gateway at the same URL. Pick a model by setting `"model": "<key>"` in the request body.

| Key | Backend | Quant | Weights (GB) | Native ctx | Served ctx | Notes |
|---|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` | vLLM | FP8 (on-disk) | ~35 | 262 k | 131 072 | MoE 35B/3B active. MTP speculative decode + FLASHINFER. General-purpose reasoning. |
| `gemma-4-26b-a4b` | llama.cpp | Q4_K_M (GGUF) | ~15 | 256 k | 131 072 | MoE 26B/4B active. Switched from vLLM due to FP8 + attention bugs. |
| `qwen3.5-35b-distill` | vLLM | BF16 (on-disk) | ~72 | 262 k (arch) | 8 192 | Claude Opus distilled. MTP + FLASHINFER. Ctx capped at 8 k = SFT ceiling. |
| `qwen3.5-122b-nvfp4` | vLLM | NVFP4 (on-disk) | ~75 | 262 k | 65 536 | Largest model via vLLM. Unstable under sustained load at 0.80 util. |
| `gemma-4-e4b` | llama.cpp | Q8_0 (GGUF) | ~8 | 128 k | 131 072 | Gemma 4 E4B with vision (mmproj). Switched from vLLM due to attention bugs. |
| `minimax-m2.7` | llama.cpp | UD-IQ4_XS (unsloth) | ~60 | ~200 k | 65 536 | 230B/10B MoE. Changed from Q3_K_XL; KV cache q4_0. |
| `supergemma-4-26b` | llama.cpp | Q8_0 (multimodal) | ~26 | 256 k | 65 536 | Gemma 4 26B abliterated + mmproj vision. Context reduced from 131K; KV q4_0. |

Per-model flags live in `config/llama-swap-swap.yaml` with comments explaining each choice.

### Benchmark results (2026-04-16, swap mode)

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

### Agent mode performance (2026-04-16)

| Model | tok/s (single) | tok/s (both active) | Concurrency scaling |
|---|---|---|---|
| `qwen3.5-122b` | 21.7 | 19.0 | 3 sub-agents: 19.7 agg tok/s |
| `gemma-4-e4b` | 39.9 | 36.6 | Vision: working (70ms TTFT) |

## Technical decisions

### 1. Swap mode vs agent mode

Running two models concurrently on Spark means splitting 119 GB between them. For anything at MiniMax (108 GB) or 122B-NVFP4 (110 GB) scale, that's untenable — swap mode gives each model the entire pool.

Agent mode works because the Qwen 122B UD-Q4_K_XL GGUF (77 GB) + Gemma E4B Q8_0 (13 GB) = 90 GB, leaving 26+ GB for KV caches. Both via llama.cpp (no vLLM memory pre-allocation headaches). This only works with smaller quantized GGUFs — vLLM models pre-allocate via `--gpu-memory-utilization` and can't share dynamically.

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

### 20. MTP speculative decoding for Qwen3.5

Qwen3.5 ships native Multi-Token Prediction (MTP) weights. MTP-1 predicts 1 additional token per step with high acceptance rate. Added `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` to `qwen3.5-35b-a3b` and `qwen3.5-35b-distill`. **Requires `--attention-backend FLASHINFER`** — MTP silently disables without it. Benchmark showed 156 tok/s pure decode on the distill model (up from 29 tok/s e2e without MTP). The 122B NVFP4 cannot use MTP — the RedHatAI checkpoint stripped the MTP head during quantization.

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

Gemma 4 E4B supports multimodal input via the `--mmproj` flag in llama.cpp. The mmproj file (`mmproj-gemma-4-E4B-it-Q8_0.gguf`) adds ~200-500 MB overhead. Enabled in both agent and swap configs. Send images via the standard OpenAI `image_url` content type.

Qwen3.5-122B does NOT support vision via llama.cpp — the CLIP graph uses unsupported operators (llama.cpp issue #21268).

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Logs (llama-swap itself)
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Logs (active backend container)
docker logs -f llama-qwen-122b           # agent mode: reasoning
docker logs -f llama-gemma-e4b           # agent mode: worker (also swap mode)
docker logs -f vllm-qwen                 # swap mode: Qwen 35B
docker logs -f vllm-qwen-distill         # swap mode: Qwen distill
docker logs -f vllm-qwen-nvfp4           # swap mode: Qwen 122B NVFP4
docker logs -f llama-gemma-26b           # swap mode: Gemma 26B
docker logs -f llama-minimax             # swap mode: MiniMax
docker logs -f llama-supergemma          # swap mode: SuperGemma

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
  - model_name: qwen3.5-35b-a3b          # what clients call
    litellm_params:
      model: openai/qwen3.5-35b-a3b      # must match the llama-swap map key
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
│   ├── llama-swap.yaml             # active config (symlinked or copied)
│   ├── llama-swap-swap.yaml        # swap mode: 7 models exclusive
│   ├── llama-swap-agent.yaml       # agent mode: 122B reasoning + E4B worker
│   └── llama-swap-bench.yaml       # benchmark variant
├── bin/
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
