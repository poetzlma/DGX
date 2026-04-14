# Spark LLM Stack

Single OpenAI-compatible endpoint serving 7 hot-swappable models on a DGX Spark (ThinkStation PGX, 192.168.1.12). One backend process resident at a time — llama-swap proxies each request, starts the matching container on demand, and tears it down when idle.

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
| Mode | `swap: true, exclusive: true` — at most one model resident |
| Auth | none (trusted LAN) |
| Model-list endpoint | `GET /v1/models` |
| Currently-loaded | `GET /running` |
| Web UI | `http://192.168.1.12:8080/ui` |

Idle timeout per model: `ttl: 3600` (1 h). A model stays hot for 1 h of inactivity before llama-swap tears the container down, freeing memory for the next swap.

## Models

All entries are OpenAI-compatible, reached through the gateway at the same URL. Pick a model by setting `"model": "<key>"` in the request body.

| Key | Backend | Quant | Weights (GB) | Native ctx | Served ctx | Notes |
|---|---|---|---|---|---|---|
| `qwen3.5-35b-a3b` | vLLM | FP8 (on-disk) | ~35 | 262 k | 131 072 | MoE 35B total / 3B active. General-purpose reasoning. Repo: `Qwen/Qwen3.5-35B-A3B-FP8` — pre-quantized to avoid flaky online MoE FP8 quant on SM 12.1. |
| `gemma-4-26b-a4b` | vLLM | FP8 (online) | ~26 | 256 k | 131 072 | MoE, multimodal-capable (text-only served here). |
| `qwen3.5-35b-distill` | vLLM | BF16 (on-disk) | ~72 | 262 k (arch) | 8 192 | `Jackrong/...Claude-4.6-Opus-Reasoning-Distilled`. No prequant available; served at BF16 with fp8 KV cache. Ctx hard-capped at 8 k = SFT ceiling per HF card. Prefix caching off (hybrid attention bug). |
| `qwen3.5-122b-nvfp4` | vLLM | NVFP4 (on-disk) | ~75 | 262 k | 65 536 | `RedHatAI/Qwen3.5-122B-A10B-NVFP4`. Largest dense-quality model we have; KV cache dominates memory past 64 k so we cap there. |
| `gemma-4-e4b` | vLLM | BF16 (on-disk) | ~8 | 128 k | 131 072 | Gemma 4 Efficient 4B, multimodal. Cheapest + fastest for simple tasks. |
| `minimax-m2.7` | llama.cpp | Q3_K_XL (unsloth) | ~96 | ~200 k | 65 536 | 230B.A10B MoE (full-attention, not lightning). Weights eat most of unified RAM; ctx bounded by KV budget. |
| `supergemma-4-26b` | llama.cpp | Q8_0 (multimodal) | ~26 | 256 k | 131 072 | Gemma 4 26B abliterated + mmproj vision encoder. |

Per-model flags live in `config/llama-swap.yaml` with comments explaining each choice.

## Technical decisions

### 1. Unified memory forces "swap" mode

Running two models concurrently on Spark means splitting 119 GB between them. For anything at MiniMax or 122B-NVFP4 scale, that's untenable. We use `swap: true, exclusive: true` so each active model gets the entire pool. Cold-swap latency (1–7 min depending on model) is the price; the alternative is running nothing.

### 2. KV cache quantization is mandatory, not optional

Because weights + KV cache share the unified 119 GB, KV cost at long context eats the memory budget.

- **vLLM entries**: `--kv-cache-dtype fp8` — ~lossless on Blackwell, halves KV memory. Lets every vLLM model serve 128 k+ context without bumping `--gpu-memory-utilization`.
- **llama.cpp entries**: `--cache-type-k q8_0 --cache-type-v q8_0` — llama.cpp benchmarks show Q8 KV is effectively indistinguishable from fp16; K is more quantization-sensitive than V but Q8 is fine for both.

### 3. Qwen3.5 is 262 k native — no YaRN

Earlier configs applied `--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`. **Those are Qwen3 params, not Qwen3.5.** Qwen3.5 ships with 262 k native context baked into `config.json`. Applying YaRN on top scales *every* request (static YaRN) and degrades short-context quality. We removed all `--rope-scaling` for Qwen3.5 entries.

### 4. Jackrong distill is capped at 32 k

The HF model card specifies SFT was performed at 8 192 tokens. The underlying architecture inherits Qwen3.5's 262 k positional embeddings, but the distilled behavior was only trained under ≤8 k. We cap at 32 k (4× SFT length, still within benign extrapolation) and recommend the base `qwen3.5-35b-a3b` for anything genuinely long.

### 5. MiniMax M2.7 is full-attention, not linear

The earlier MiniMax-Text-01 used hybrid lightning/linear attention (KV cost sub-linear in context). **The M2 series reverted to full softmax attention** per MiniMax's own engineering post. KV cost scales linearly with context, so at 96 GB weights + 23 GB free budget, ctx is tightly bounded. With Q8_0 KV + flash attention we can safely run 64 k; higher is possible but OOM-risky.

### 6. `--no-mmap` for llama.cpp on Spark

With default `mmap = true`, llama.cpp memory-maps the GGUF and copies tensors to CUDA one at a time via synchronous `cudaMemcpy`. Each copy blocks the next page fault, so NVMe only hits ~200 MB/s at ~17% utilization (observed). `--no-mmap` replaces this with `pread()` streaming, ~10× faster on Spark's 6.17 kernel (community-verified on NVIDIA forums).

### 7. `--load-format fastsafetensors` — considered, not applied

vLLM's default safetensors loader is also fault-driven on Spark UMA. `fastsafetensors` (pip package) issues large parallel reads and cuts cold load times by 10×+. **Not currently enabled** — the `cu130-nightly` image may or may not ship the package in any given daily build, and the flag hard-fails startup when missing. Warm-cache shard loads complete in ~80 s without it, which is acceptable for the `ttl: 3600` swap cadence. Revisit if first-load or reload times become the bottleneck.

### 8. `-fa` flash attention for llama.cpp

Standard best practice on any CUDA-capable llama.cpp build. Reduces attention intermediate memory and improves decode throughput. Always on for our llama.cpp entries.

### 9. `VLLM_FLASHINFER_MOE_BACKEND=latency` env

The default FlashInfer MoE backend (`throughput`) emits SM120-generic kernels that misbehave on SM12.1 (Spark's GB10 chiplet). The community-tested workaround — per the avarok/vllm-dgx-spark project — is to force the latency backend, which uses a compatible kernel path. We set it as an env var on every vLLM entry.

### 10. `--enable-prefix-caching` for vLLM

Single-user workloads reuse the same system prompts repeatedly. Prefix caching keeps computed KV for identical prefixes across requests, eliminating redundant prefill. TTFT drops significantly on repeated conversations. Cheap, always on.

### 11. Container cleanup in systemd

llama-swap uses `docker run --rm` to ensure backends auto-clean on normal exit. But an unclean llama-swap shutdown (crash, `systemctl kill`) leaves orphan containers holding GPU memory reservations. We added `ExecStartPre` and `ExecStopPost` hooks to the systemd unit that run `docker ps -aq --filter name=llama- | xargs -r docker rm -f` to catch this case.

### 12. Service enabled at boot

`systemctl enable llama-swap` — the service was `disabled` at install time, so the gateway silently stayed down after reboots. Now starts automatically.

### 13. vLLM image must be `cu130-nightly` for Qwen3.5 MoE

Published image `vllm/vllm-openai:gemma4-cu130` ships vLLM 0.14-branch, which does **not** register `Qwen3_5MoeForConditionalGeneration`. Every Qwen3.5 entry fails at arch-registry lookup during model init — observed as `exit status 1` within ~25 s of `docker run` start. Gemma 4 loads fine on the same image because its arch class *is* registered. The required class landed upstream in vLLM 0.16+; we pull `vllm/vllm-openai:cu130-nightly` (currently vLLM 0.19.1rc1) for all Qwen3.5 entries. The nightly also fixes the Hopper/Blackwell split so SM 12.1 (GB10) kernels are selected correctly. Gemma entries stay on `gemma4-cu130` — it's smaller, Gemma-optimised, and not affected.

### 14. `--max-num-batched-tokens ≥ block_size` for Qwen3.5 hybrid Mamba

Qwen3.5 MoE is a **hybrid attention + SSM** architecture. With `--enable-prefix-caching`, vLLM switches the Mamba cache to "align" mode, which enforces `block_size ≤ max_num_batched_tokens` at CUDA-graph profiling. Default `max_num_batched_tokens` is 2048 but vLLM resolves `block_size = 2096` for Qwen3.5 — assertion fires and the engine exits immediately after a clean weight load. Fix: set `--max-num-batched-tokens 8192` on all Qwen3.5 entries. Any value ≥ 2096 works; 8192 matches vLLM's recommended batching for interactive MoE workloads.

### 15. First-run cold load can exceed `VLLM_ENGINE_READY_TIMEOUT_S`

vLLM's APIServer has its own 600 s default deadline to wait for EngineCore to signal ready, separate from llama-swap's `healthCheckTimeout`. On first load of a model whose weights aren't cached, the 35–75 GB HF download runs inside that window — we hit the timeout at 600 s even though weight download + load was still healthy and completed at ~1200 s. Every vLLM entry now sets `-e VLLM_ENGINE_READY_TIMEOUT_S=1800`. Warm-cache cold loads complete in ~90 s, so this only matters on first fetch.

### 16. `--language-model-only` for the Qwen3.5 MoE family

`Qwen3_5MoeForConditionalGeneration` is an **image-text-to-text** architecture. Every Qwen3.5 MoE repo declares a 27-layer SigLIP-style vision tower in `config.json` (`vision_config.depth: 27`), and vLLM will load those weights + spin up a multimodal scheduler by default — even if we only send text. Worse, the Jackrong distill and the RedHatAI NVFP4 repos don't ship a `preprocessor_config.json`, so vLLM fails cold load with `OSError: Can't load image processor`. `--language-model-only` skips the vision branch and the image-processor loader. On the 35B distill this cuts loaded memory from ~72 GiB to ~65 GiB; on 122B NVFP4 from ~80 GiB to ~70 GiB. Applied to all three Qwen3.5 entries. No downside for our text-only workload.

### 17. `--moe-backend flashinfer_cutlass` for NVFP4 MoE on GB10

FlashInfer's default MoE backend on SM 12.1 attempts SM 8.0/Ampere tactics with shared-memory budgets GB10 doesn't have, and newer NVFP4 paths fail with "Failed to initialize cutlass TMA WS grouped gemm — Error Internal" (vllm #33416, #38971). RedHat's own Qwen3.5-122B NVFP4 README sets `--moe-backend flashinfer_cutlass`. This is a **CLI flag** (not the env var `VLLM_FLASHINFER_MOE_BACKEND`) and overrides per-entry. Applied to the `qwen3.5-122b-nvfp4` entry; the base FP8 and distill entries use the default backend and serve fine. The autotuner spews a flood of "skipping tactic" warnings during profiling — these are expected, the autotuner is enumerating candidate kernels and falling back when SM 8.0/12.0 tactics don't fit.

### 18. `--reasoning-parser qwen3` for Qwen3.5 chat completions

Qwen3.5 chat templates inject `<|im_start|>assistant\n<think>\n` on `add_generation_prompt`. Without a reasoning parser, OpenAI clients see `</think>` embedded inside `choices[].message.content` instead of the `message.reasoning_content` field. Every Qwen3.5 entry now sets `--reasoning-parser qwen3` so downstream clients (LiteLLM, OpenWebUI) can render the reasoning stream separately from the final answer.

### 19. Tokenizer-class patch: `"TokenizersBackend"` → `"Qwen2TokenizerFast"`

Two of our three Qwen3.5 repos (`Jackrong/...Distilled`, `RedHatAI/Qwen3.5-122B-A10B-NVFP4`) ship `tokenizer_config.json` with `tokenizer_class: "TokenizersBackend"`. That's a transformers v5.x class name; vLLM is pinned to transformers <5 and raises `ValueError: Tokenizer class TokenizersBackend does not exist`. `--trust-remote-code` does **not** help — `auto_map` is null, so there's nothing to remote-load. Upstream vLLM PR #38099 only improves the error message; the fix is pinned on a transformers bump.

Our workaround: in-place patch of the cached `tokenizer_config.json` files:

```bash
for model in \
  "Jackrong--Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled" \
  "RedHatAI--Qwen3.5-122B-A10B-NVFP4"; do
  f=$(find ~/.cache/huggingface/hub/models--${model} -name tokenizer_config.json -path "*/snapshots/*" | head -1)
  [ -f "${f}.bak" ] || cp "$f" "${f}.bak"
  python3 -c "import json; d=json.load(open('$f')); d['tokenizer_class']='Qwen2TokenizerFast'; json.dump(d, open('$f','w'), indent=2, ensure_ascii=False)"
done
```

Semantically safe: vocab (248,044) and BPE merges are identical to base `Qwen2TokenizerFast`; the extra added_tokens in these repos live in `added_tokens_decoder` and load regardless of the class string. **`hf download --force` will revert the patch** — re-apply if weights are re-fetched. The base `Qwen/Qwen3.5-35B-A3B-FP8` repo was exported on transformers 4.x and does not need this patch.

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Logs (llama-swap itself)
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Logs (active backend container)
docker logs -f vllm-qwen                  # or vllm-gemma / vllm-qwen-distill / vllm-qwen-nvfp4 / vllm-gemma-e4b
docker logs -f llama-minimax              # or llama-supergemma

# Current state
curl http://192.168.1.12:8080/running     # which model is hot
curl http://192.168.1.12:8080/v1/models   # model list
docker ps --filter name=llama- --filter name=vllm-

# Force an unload without stopping the service
curl -X POST http://192.168.1.12:8080/unload
```

Cold-swap behaviour: the first call after an idle timeout or after hitting a different model triggers a container spin-up. Expect 1–7 min. LiteLLM or any wrapping client should set a per-request timeout ≥ 900 s.

## Benchmarking

`bin/bench-models.py` walks every model in the swap group and measures:

- **cold_s** — wall time from request to first response, including previous-model teardown + container start + weight load + first-token generation.
- **peak_mem_gb** — max `/proc/meminfo` usage sampled at 2 Hz during load.
- **model_gb** — `peak_mem_gb` minus pre-load baseline (approximates weights + KV + activations).
- **tok/s** — end-to-end generation rate on a 256-token completion with the backend already warm.

Run:
```sh
/home/max/llm-stack/venv/bin/python3 /home/max/llm-stack/bin/bench-models.py \
  | tee logs/bench-$(date +%Y%m%d-%H%M).log
```

Results are saved to `logs/bench-results.json`. Expected total runtime: 30–60 min.

## Adding a model

1. **Check the cache first:** `ls ~/.cache/huggingface/hub/ | grep -i <name>`. Many downloads are already present.
2. **Download if missing:** `~/llm-stack/venv/bin/hf download <org>/<repo>`. Requires `max:max` ownership on `~/.cache/huggingface/hub` (one-time `chown`; container writes break ownership if we forget).
3. **Add a block under `models:`** in `config/llama-swap.yaml`:
   - safetensors (BF16, FP8, NVFP4/compressed-tensors): use the vLLM pattern (see `qwen3.5-35b-a3b` as the reference template — fp8 online quant, 131 k ctx, fp8 KV, prefix caching, fastsafetensors, flashinfer=latency).
   - GGUF: use the llama.cpp pattern (see `minimax-m2.7` — Q8 KV, flash attention, no-mmap, no warmup).
4. **Append the new key** to `groups.main.members`.
5. **Validate**: `python3 -c "import yaml; yaml.safe_load(open('config/llama-swap.yaml'))"`.
6. **Restart**: `sudo systemctl restart llama-swap`.
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

The `openai/<key>` string in `litellm_params.model` must exactly match the key under `models:` in `llama-swap.yaml` — that's how llama-swap routes and decides which backend to swap in. Inner `--served-model-name` / `--alias` flags don't matter for routing; they only affect the `model` field in response bodies.

## Files

```
~/llm-stack/
├── config/llama-swap.yaml      # model definitions, swap group
├── bin/bench-models.py         # warm-up + TPS benchmark
├── venv/                       # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.log          # gateway stdout
│   ├── llama-swap.err          # gateway stderr
│   ├── downloads/              # hf download logs
│   └── bench-*.log             # benchmark runs
└── README.md

~/bin/llama-swap                 # gateway binary (v201)
/etc/systemd/system/llama-swap.service   # with ExecStartPre cleanup hook
~/.cache/huggingface/hub/        # model weights (max:max ownership required)
~/.cache/huggingface/token       # HF auth, chmod 600
```

## Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| Gateway down after reboot | Service not enabled | `sudo systemctl enable llama-swap` |
| 502 with `upstream command exited prematurely but successfully` | Orphan container holds port/VRAM | `docker rm -f llama-<name>`; systemd unit now handles via `ExecStartPre` |
| Per-model `SIGKILL` leaves orphan container (memory/GPU pinned) | llama-swap's `healthCheckTimeout` kill doesn't run the service-level `ExecStopPost` cleanup hook | `docker rm -f <name>` manually; then bump `healthCheckTimeout` and/or `VLLM_ENGINE_READY_TIMEOUT_S` so the next load finishes before kill |
| CUDA OOM on backend start | Previous container still resident | same as above; verify with `docker ps` |
| Cold load 5× slower than expected | Default mmap path on Spark | ensure `--no-mmap` (llama.cpp) or `--load-format fastsafetensors` (vLLM, only on images that ship the package) |
| 502 on first call, fine after | Model still loading — healthcheck hasn't passed | wait; watch `docker logs` for `Application startup complete` |
| Qwen3.5 entry exits `status 1` within ~25 s | Image's vLLM is too old to register `Qwen3_5MoeForConditionalGeneration` | switch image to `vllm/vllm-openai:cu130-nightly` (vLLM 0.16+) |
| `AssertionError: In Mamba cache align mode, block_size (2096) must be <= max_num_batched_tokens (2048)` | Qwen3.5 hybrid arch + `--enable-prefix-caching` requires larger batch budget | add `--max-num-batched-tokens 8192` (≥ 2096 works) |
| `TimeoutError: ... VLLM_ENGINE_READY_TIMEOUT_S` during first-ever load | 600 s APIServer default deadline too short for 35–75 GB HF download + weight load | `-e VLLM_ENGINE_READY_TIMEOUT_S=1800` (set on every vLLM entry). Or pre-download weights with `hf download <repo>`. |
| `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported` | Repo exported on transformers v5 writes `"TokenizersBackend"` literal; vLLM pinned to <5 | patch cached `tokenizer_config.json` to `"Qwen2TokenizerFast"` (see §19). `--trust-remote-code` does **not** help. |
| `OSError: Can't load image processor for 'Jackrong/...'` | vLLM treats `Qwen3_5MoeForConditionalGeneration` as multimodal; repo doesn't ship `preprocessor_config.json` | add `--language-model-only` (see §16) |
| Flood of `[Autotuner]: Skipping tactic ... GPU lacks the shared memory resources` / `Failed to initialize cutlass TMA WS grouped gemm` | FlashInfer MoE autotuner trying SM 8.0/12.0 tactics that don't fit GB10 | expected for Qwen3.5 MoE. Non-fatal — autotuner picks a working tactic. For 122B-NVFP4 specifically set `--moe-backend flashinfer_cutlass` (see §17) |
| vLLM errors on NVFP4 weights | Image lacks compressed-tensors NVFP4 kernels | pull a newer vllm-openai image |
| `PermissionError` downloading to cache | `~/.cache/huggingface/hub` root-owned by prior docker writes | `sudo chown -R max:max ~/.cache/huggingface/{hub,xet}` |
| MoE throughput bad, garbled output | FlashInfer backend mismatch on SM12.1 | `-e VLLM_FLASHINFER_MOE_BACKEND=latency` (already set on all vLLM entries) |
| Poor long-context quality on Qwen3.5 | Leftover YaRN from Qwen3 template | confirm `--rope-scaling` is absent in that entry's cmd |
| Distill worse than base at long ctx | Jackrong SFT was at 8 k; quality degrades past that | use base `qwen3.5-35b-a3b` for long-context work |
