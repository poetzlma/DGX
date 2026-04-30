# Spark LLM Stack

Single OpenAI-compatible endpoint serving models on a DGX Spark (ThinkStation PGX, 192.168.1.12). One **`main` group** with `swap: true, exclusive: true` — all heavies hot-swap one at a time, with `qwen3.6-27b` (solo vLLM AEON-7 NVFP4 + native Qwen3.6-DFlash drafter, k=15, 200 k ctx) as the production coding default since 2026-04-30. The MTP-3 path it replaced is kept dormant as `qwen3.6-27b-mtp` for one-line rollback (see §26b). The `qwen3.6-35b-a3b` MoE is kept **dormant but routable** in the same group: a request cold-starts it and evicts whichever heavy is currently up.

**Architecture history.**
- Until 2026-04-24: `qwen36` co-resident pair (27B-GGUF llama.cpp + 35B-A3B MoE vLLM, both warm).
- 2026-04-24 → 2026-04-30: solo 27B NVFP4 + MTP-3 (AlphaOxO), c=10 ~149 tok/s aggregate.
- 2026-04-30 onward: solo 27B NVFP4 + DFlash k=15 (AEON-7 image + z-lab native drafter). Head-to-head bench: **c=1 41 vs 20, c=10 207 vs 169 tok/s aggregate vs the prior MTP path** (see §26). DFlash wins because GB10 unified-memory bandwidth bottlenecks decode and the block-diffusion drafter accepts more tokens per memory round than MTP's in-target head. Deep writeup at [`docs/qwen3.6-27b-dflash.md`](docs/qwen3.6-27b-dflash.md).

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

`swap: true, exclusive: true`, at most one model resident. Idle timeout `ttl: 3600` (1 h). Cold-swap latency 1–10 min depending on model. Requesting any member evicts whatever was up. Members listed in §Models below.

### Switching models

Just pick by model key — llama-swap handles eviction. The `config/llama-swap-swap.yaml` and `config/llama-swap-agent.yaml` files are **legacy templates from the pre-qwen3.6 era** and are not used by the running service; kept in-tree as reference snapshots of the old swap-only / 122B+E4B setups.

## Models

All entries are OpenAI-compatible, reached through the gateway at the same URL. Pick a model by setting `"model": "<key>"` in the request body. Group membership determines whether requesting one evicts others.

| Key | Group | Backend | Quant | Weights (GB) | Native ctx | Served ctx | Notes |
|---|---|---|---|---|---|---|---|
| `qwen3.6-27b` | main | vLLM | NVFP4 (AEON-7) | ~26 | 262 k | 200 000 | **Production coding default since 2026-04-30.** Dense 27B + DFlash k=15 speculative + flash_attn, util 0.85, 16 concurrent slots, AEON-7 patched image `vllm-aeon-ultimate-dflash:qwen36-v3`. Drafter `z-lab/Qwen3.6-27B-DFlash` v2. Launcher at `bin/launch-vllm-27b-dflash.sh` (see §26). |
| `qwen3.6-27b-mtp` | main | vLLM | NVFP4 (AlphaOxO) | ~14 | 262 k | 262 144 | **Dormant rollback slot** (was prod 2026-04-24 → 2026-04-30). Dense 27B + MTP-3 + FLASHINFER, KV fp8, util 0.85, 10 concurrent slots, stock `vllm/vllm-openai:cu130-nightly`. Launcher at `bin/launch-vllm-27b-nvfp4.sh` (see §26b). MTP requires the AlphaOxO config patch (see §MTP-patch). |
| `qwen3.6-35b-a3b` | main | vLLM | NVFP4 (RedHatAI) | ~23 | 262 k | 131 072 | **Dormant — routable on demand** (see §28). MoE 35B/3B-active, MTP-1 + FLASHINFER, util 0.55. A request cold-starts it and evicts whichever heavy is up (~5 min). Launcher at `bin/launch-vllm-qwen.sh` (see §25). NOTE: also aliased onto `qwen3.6-27b` (DFlash) at the gateway for legacy opencode sessions — explicit `qwen3.6-35b-a3b` requests still cold-start the real MoE. |
| `gemma-4-26b-a4b` | main | llama.cpp | Q4_K_M (GGUF) | ~15 | 256 k | 131 072 | MoE 26B/4B active. Switched from vLLM due to FP8 + attention bugs. |
| `qwen3.5-35b-distill` | main | vLLM | BF16 (on-disk) | ~72 | 262 k (arch) | 8 192 | Claude Opus distilled. MTP + FLASHINFER. Ctx capped at 8 k = SFT ceiling. |
| `qwen3.5-122b-nvfp4` | main | vLLM | NVFP4 (on-disk) | ~75 | 262 k | 65 536 | Largest model via vLLM. Unstable under sustained load at 0.80 util. |
| `minimax-m2.7` | main | llama.cpp | UD-IQ4_XS (unsloth) | ~60 | ~200 k | 65 536 | 230B/10B MoE. Changed from Q3_K_XL; KV cache q4_0. |
| `supergemma-4-26b` | main | llama.cpp | Q8_0 (multimodal) | ~26 | 256 k | 65 536 | Gemma 4 26B abliterated + mmproj vision. Context reduced from 131K; KV q4_0. |
| `gemma-4-e4b` | main | llama.cpp | Q8_0 (GGUF) | ~8 | 128 k | 131 072 | Gemma 4 E4B with vision (mmproj). Used as vision worker. |
| `nemotron-3-nano-omni` | main | vLLM | NVFP4 (NVIDIA) | ~21 | 262 k | 131 072 | **Multimodal omni** (text + image + audio + video) — Mamba2-Transformer hybrid MoE 30B/3B-active. Reasoning model. Custom image `cu130-nightly-omni` (av/soundfile baked in). util 0.65, max-num-seqs 8, c=8 throughput peak ~383 tok/s. Launcher `bin/launch-vllm-nemotron-omni.sh` (see §29). |

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

Throughput regresses from c=4 → c=10 because llama.cpp's slot count can't keep up. Request queueing rather than real batching dominates at c≥5. This was fine for the 27B's pair-era role (deep/thinking only) but motivated the 04-24 switch to vLLM NVFP4+MTP for proper batching.

### Solo 27B-NVFP4+MTP benchmark (2026-04-24, prior production)

Same prompt profile, server-side measurement via gateway, sweeping `num_speculative_tokens` (n) from 1 to 3:

| n | c=1 decode | c=10 peak agg | c=10 sustained | MTP accept (n=3) |
|---|---|---|---|---|
| 1 | ~13 tok/s | 120 tok/s | — | 1/1 (94 %) |
| 2 | ~16 tok/s | 136 tok/s | — | mean 1.74/2 (87 %, 75 %) |
| **3** | **~19 tok/s** | **149 tok/s** | 94–149 tok/s | mean 3.0/4 (85 %, 63 %, 51 %) |

n=3 was production from 2026-04-24 → 2026-04-30. Through the llama-swap gateway adds ~3 % vs direct-to-:9008 (149 vs 154); proxy overhead is negligible.

Net vs the GGUF-pair-era spot-check at c=10: 25 → 149 tok/s aggregate (~6×) with the same hardware, at higher quality-per-token.

### Solo 27B-NVFP4+DFlash benchmark (2026-04-30, current production)

Same `bin/bench-concurrency-sweep.py` SHORT_MSGS profile, 512 max_tokens, gateway-measured. DFlash k=15 vs the prior MTP n=3, head-to-head, sequential cold-runs evicted via llama-swap:

| c | DFlash agg tok/s | MTP agg tok/s | Δ | DFlash TTFT | MTP TTFT |
|---|---|---|---|---|---|
| 1 | **41.0** | 20.3 | **+102 %** (2.0×) | 394 ms | 446 ms |
| 5 | **139.0** | 92.0 | **+51 %** | 1428 ms | 2070 ms |
| 10 | **207.1** | 169.1 | **+22 %** | 703 ms | 978 ms |

vLLM engine-internal burst peaks (`loggers.py`, instantaneous Avg generation throughput across the running batch):

| c | DFlash burst | MTP burst | ratio |
|---|---|---|---|
| 1 | 79.5 tok/s | 22.0 | 3.6× |
| 5 | 175.8 | 106.0 | 1.66× |
| 10 | 264.6 | 182.4 | 1.45× |

DFlash wins everywhere; the gap narrows as concurrency rises because batch effects start dominating speculative-decode gains. AEON-7's published numbers reproduced (their 38.1 / 68.4 c=1 vs our 41.0 / 79.5). See [`docs/qwen3.6-27b-dflash.md`](docs/qwen3.6-27b-dflash.md) for the full per-burst table, gotchas, and rollback recipe.

## Technical decisions

### 1. Solo 27B vs. the retired qwen36 co-resident pair

The 27B and 35B-A3B used to run as a co-resident pair (qwen36 group, `swap: false`) — both warm at once, each at reduced `--gpu-memory-utilization` so they fit together in the 119 GB unified budget. The split was 27B-dense for deep reasoning + 35B-A3B-MoE for parallel sub-agent fanout.

In the 2026-04-24 sweep, solo 27B-NVFP4 with MTP-3 at c=10 hit ~149 tok/s aggregate, matching what the MoE used to deliver on parallel workers — at higher quality-per-token. Removing the MoE freed the full 119 GB so the 27B can run `--gpu-memory-utilization 0.85`, full 262 k ctx, and 10 concurrent slots without thrashing the page cache. Net: same fanout throughput, better single-stream quality, simpler memory model.

The MoE entry stays in `main` as **dormant routable** (§28) — a request cold-starts it and evicts the 27B. Reactivation cost: ~2–3 min cold start, no config edits needed.

Why not multi-model co-resident at all anymore: at MiniMax (108 GB) / 122B-NVFP4 (110 GB) scale you can't co-locate anything, and 27B-NVFP4 alone now gives the throughput we needed the pair for. `main`'s `exclusive: true` keeps the memory model uniform across the whole stack.

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

Apply the same pattern to any other vLLM container where post-mortem logs matter. The 27B production launcher (`bin/launch-vllm-27b-nvfp4.sh`, see §26) follows the same convention with container name `vllm-qwen-27b`.

### 26. Solo `qwen3.6-27b` NVFP4+DFlash production launcher (2026-04-30)

Replaces the prior NVFP4+MTP launcher (relegated to dormant rollback slot in §26b). Single launcher script `bin/launch-vllm-27b-dflash.sh`, container name `vllm-qwen-27b-dflash`, port 9013. Key choices:

- **Target repo `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4`** — AEON-7's lossless abliterated NVFP4 build (~26 GB on disk). Multimodal-capable (vision encoders preserved BF16) but launched here with `--language-model-only` to mirror the text-only coding workload.
- **Drafter `z-lab/Qwen3.6-27B-DFlash` v2** (~3.3 GB) — refreshed 2026-04-27 push. The native Qwen3.6 DFlash drafter; was the gating condition for retrying DFlash on this stack after the 2026-04-23 cross-gen Qwen3.5-DFlash failure.
- **Image `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3`** — vLLM 0.20.1.dev0+g88d34c640 with 5 patches (#40092 SWA backend, #40454 mamba-cache spec-decode align, #40191 ENABLE_NVFP4_SM100=0 guard, #40662 unified spec-decode acceptance metrics, #38479 TurboQuant K8V4 baked in) + FlashInfer 0.6.9rc1 (first SM121 NVFP4 GEMM path). Stock `vllm/vllm-openai:cu130-nightly` does NOT support DFlash.
- **DFlash `num_speculative_tokens=15`** — AEON-7's recommended k for this dense 27B. Acceptance high enough that the verifier-side overhead is well below the speedup. (Cross-gen pairings capped at k=3; the native v2 drafter is what made k=15 viable.)
- **`--gpu-memory-utilization 0.85`** — same envelope as the prior MTP launcher; do not push above 0.88 on Spark per AEON-7's docs (unified memory thrashes).
- **`--max-model-len 200000`** — AEON-7 recommends 200 k vs the model's 262 k native; KV budget at full 262 k tightens concurrency below useful levels with DFlash's drafter-side allocations. KV cache size reported by vLLM at boot: 205,632 tokens (vs MTP launcher's 552,000) → max concurrency 2.69× at 200 k ctx. Fine for current workload, tighter than MTP if traffic ever pushes >2 concurrent users at near-200 k input.
- **`--max-num-seqs 16`**, **`--max-num-batched-tokens 32768`**, **`--enable-chunked-prefill --enable-prefix-caching`**, **`--kv-cache-dtype auto`** (NVFP4 path lets vLLM pick), **`--attention-backend flash_attn`** (AEON-7 recipe; not FLASHINFER).
- **GB10 env vars (mandatory)**: `TORCH_CUDA_ARCH_LIST=12.1a`, `ENABLE_NVFP4_SM100=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_USE_FLASHINFER_MOE_FP4=0` (model is dense — disable FP4 MoE auto-probe to avoid SM121 PTX rejection log spam), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Without `ENABLE_NVFP4_SM100=0` the C-stable libtorch ABI fails to import (SM100-only mxfp4_experts_quant kernels don't exist on SM121).
- **`--served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-dflash`** — accepts the default name, the legacy MoE alias from old opencode sessions, and the explicit DFlash name. Gateway routes all three to this launcher (the MoE alias is also held at the llama-swap entry level; explicit `qwen3.6-35b-a3b` requests still cold-start the real MoE — see §28).
- **No `--reasoning-parser qwen3`** — raw `<think>` is streamed to the client (opencode parses it natively). Same TTFT rationale as §26b. Differs from AEON-7's docker-compose, which keeps the parser on; we drop it deliberately.
- **No `--rm`** on the container — same crash-traceability rationale as §25.

Cold start ~10 min (FlashInfer NVFP4 GEMM autotuner + CUDA-graph capture across 51 sizes; both cache to `/root/.cache/vllm/...` inside the container, which is *not* bind-mounted to host so caches are rebuilt on every container start). Boot summary verified at 2026-04-30 18:22 UTC: `Application startup complete.` after 10 min from trigger. **Real-traffic users hitting `qwen3.6-27b` immediately after a swap-eviction wait the full 10 min** — this is the operational difference vs MTP's ~6 min cold start.

Bench results in [§Solo 27B-NVFP4+DFlash benchmark](#solo-27b-nvfp4dflash-benchmark-2026-04-30-current-production) above; deep writeup in [`docs/qwen3.6-27b-dflash.md`](docs/qwen3.6-27b-dflash.md).

### 26b. Dormant `qwen3.6-27b-mtp` rollback launcher (was prod 2026-04-24 → 2026-04-30)

Kept on disk and addressable by explicit name `qwen3.6-27b-mtp` for one-line rollback. Launcher script `bin/launch-vllm-27b-nvfp4.sh`, container name `vllm-qwen-27b`, port 9008. Joins `main` swap-exclusive group; will not load unless explicitly requested. To restore as default, swap the `qwen3.6-27b` block's `cmd:` line in `config/llama-swap.yaml` back to this launcher and `proxy:` to 9008 (`-watch-config` picks it up — no restart). Backups for safety: `config/llama-swap.yaml.bak.20260430-pre-dflash`, `bin/launch-vllm-27b-nvfp4.sh.bak.20260430-pre-dflash`.

Original config (preserved in the launcher file, demoted comment block updated 2026-04-30):

- **Repo `AlphaOxO/Qwen3.6-27B-NVFP4`** — one of only two Qwen3.6-27B NVFP4 quants that preserve MTP weights (other: `ig1/Qwen3.6-27B-NVFP4`). Verify post-download: `ls ~/.cache/huggingface/hub/models--AlphaOxO--Qwen3.6-27B-NVFP4/snapshots/*/model_mtp.safetensors`.
- **MTP `num_speculative_tokens=3`**, **`--gpu-memory-utilization 0.85`**, **`--max-model-len 262144`**, **`--kv-cache-dtype fp8`**, **`--enable-prefix-caching`**, **`--max-num-seqs 10`**, **`--attention-backend FLASHINFER`** (mandatory for MTP, see §20).
- **`--served-model-name qwen3.6-27b-mtp`** only — does NOT alias the default `qwen3.6-27b` anymore. This is the deliberate change made on 2026-04-30 to ensure requests to the default name never accidentally land here.
- **No `--reasoning-parser qwen3`**, **no `--rm`** — same rationales as §26.

Requires the **MTP patch** (see §MTP-patch below) — without it MTP silently no-ops and you lose 1.6–2× decode throughput. Cold start ~6 min.

### 27. llama-swap `-watch-config` hot-reload (2026-04-26)

llama-swap runs with `-watch-config` via the drop-in at `/etc/systemd/system/llama-swap.service.d/watch-config.conf` (mirrored in repo at `systemd/llama-swap.service.d/watch-config.conf`). YAML edits to `config/llama-swap.yaml` apply within ~1 s **without restarting the proxy** and **without touching running model containers** — no cold-start tax on config changes.

Before this drop-in, every yaml edit needed `pkill -9 llama-swap`. Fresh llama-swap had no state, so the first request to a model re-ran its launcher, which begins with `docker rm -f <container>` to clean prior runs → killed the still-healthy container → ~6 min cold start. Hot-reload eliminates that tax.

Drop-in contents:

```ini
[Service]
ExecStart=
ExecStart=/home/max/bin/llama-swap -config /home/max/llm-stack/config/llama-swap.yaml -watch-config -listen 0.0.0.0:8080
```

The empty `ExecStart=` line is required to override the unit's original ExecStart before setting the new one (standard systemd drop-in pattern). Apply: `sudo install -m 0644 systemd/llama-swap.service.d/watch-config.conf /etc/systemd/system/llama-swap.service.d/watch-config.conf && sudo systemctl daemon-reload && sudo systemctl restart llama-swap` (one disruption, then permanent).

Fallback for changes that aren't yaml-only (binary upgrade, unit edits, stuck process): `pkill -9 llama-swap`. SIGTERM (plain `pkill`) is a clean exit so systemd does NOT respawn (`Restart=on-failure`); only SIGKILL or `sudo systemctl restart` works.

### 28. MoE `qwen3.6-35b-a3b` kept dormant-but-routable (2026-04-26)

The MoE entry stays in `main` group (swap-exclusive with the 27B and the other heavies). On disk: `RedHatAI/Qwen3.6-35B-A3B-NVFP4` weights (24 GB) + `Qwen/Qwen3.6-35B-A3B-FP8` for the tokenizer source. Launcher `bin/launch-vllm-qwen.sh` unchanged from the qwen36-pair era.

Reactivation cost: a request to `qwen3.6-35b-a3b` triggers `docker rm -f vllm-qwen-27b` (in the 27B's launcher cleanup chain — actually llama-swap stops the previous member because of `exclusive: true`), then cold-starts the MoE — ~2–3 min. Going back to 27B costs ~6 min (torch.compile + MTP + 262 k).

If the MoE is no longer needed at all: comment its block in `config/llama-swap.yaml` and remove from `main.members`. Weights stay in HF cache for future re-activation. Save → hot-reload (§27) does the rest.

### 29. `nemotron-3-nano-omni` multimodal slot (2026-04-29)

Added as the omni multimodal lane (text + image + audio + video). NVIDIA's Mamba2-Transformer hybrid MoE 30B/3B-active with CRADIO-v4-H vision and Parakeet audio encoders. Sits in `main` swap-exclusive group alongside the 27B; treat as orthogonal — qwen3.6-27b stays the coding default, nemotron-omni handles multimodal. Launcher `bin/launch-vllm-nemotron-omni.sh`, container `vllm-nemotron-omni`, port 9012.

**Custom image required.** vLLM's audio path needs `av` (PyAV) and `soundfile` (and `librosa`) at process import time. The base `cu130-nightly` image ships without them. Hot-installing into a running container leaves vLLM with a cached `PlaceholderModule` and audio requests fail with `AssertionError: PlaceholderModule should not be used when the original module can be imported`. Build a derived image once:

```bash
# in a running container started from cu130-nightly:
docker exec vllm-nemotron-omni pip install av soundfile librosa
docker commit vllm-nemotron-omni vllm/vllm-openai:cu130-nightly-omni
```

Launcher pins the new tag. Without this, image works but audio/video fail.

**`--max-num-batched-tokens 8192` is required.** Same Mamba block-size assertion as Qwen3.5 (§14): vLLM resolves Mamba `block_size = 2128` on this model, default `--max-num-batched-tokens=2048` fires `AssertionError: In Mamba cache align mode, block_size (2128) must be <= max_num_batched_tokens (2048)`.

**Sized down from the qwen 0.85 envelope.** 2026-04-29 sweeps showed:

- **Concurrency peak at c=8**: 383 tok/s aggregate. Sharp cliff at c=9 (-30 % throughput, +60 % latency). Bound by Mamba SSM state + MoE router contention, not KV. Set `--max-num-seqs 8`.
- **Memory math**: model + scaffolding ~25 GB (NVFP4 weights + vision/audio encoders in BF16 + CUDA graphs + activations). KV in-flight at c=8 with 100 k context ≈ 12 GB. Multimodal scratch peaks +1.8 GB (1080p 23 s video) and +1.2 GB (5 min audio). Prefix cache wants ~30 GB to be load-bearing on shared system prompts. Total ≈ 70 GB → **`--gpu-memory-utilization 0.65`** (~77 GB allocated, frees ~25 GB to OS vs the 27B's 0.85 envelope).
- **Headroom check**: at 0.65, baseline 91 GB → peak with 1080p video 94.6 GB. ~25 GB free. 1 hr audio extrapolated peak ~106 GB, still under 119 GB.

**Multimodal scratch curves** (Δ peak host mem above warm baseline, 0.65 util):

| modality | sample | prompt_tokens | TTFT | decode | Δmem |
|---|---|---|---|---|---|
| image | 640×1280 PNG | 827 | 4.7 s | 55 t/s | +10 MB |
| image | 1920×1080 JPEG | 2068 | 5.3 s | 48 t/s | +91 MB |
| audio | 1.5 s mp3 | 242 | 4.7 s | 55 t/s | +0 MB |
| audio | 23 s wav | 312 | 7.0 s | 55 t/s | +21 MB |
| audio | 5 min wav | 3771 | 20.0 s | 19 t/s | **+1.2 GB** |
| video | 518×294 15.8 s | 2507 | 8.7 s | 44 t/s | +323 MB |
| video | 1920×1080 23 s | 3630 | 11.5 s | 33 t/s | **+1.8 GB** |

Image scratch is cheap (linear in pixels). Audio and 1080p video are the heavy cases — run a real test before committing to 1 hr audio or 2 min 1080p.

**Reasoning parser TTFT**. `--reasoning-parser nemotron_v3` is set per the model card. The parser buffers the entire `<think>…</think>` block before emitting any `delta.content` chunk to the stream, so on a medium-output reasoning prompt TTFT measures ~18 s even though the model is generating fine. Drop the flag if interactive UX matters more than parsed `message.reasoning_content` (qwen3.6-27b deliberately omits it for the same reason — see §26).

**Other notable flags** (see launcher comments for the full set):
- `--kv-cache-dtype fp8` per model card.
- `--tool-call-parser qwen3_coder` per model card.
- `--allowed-local-media-path /home/max` enables `file://` URLs in chat content blocks for local testing.
- `--max-model-len 131072` (model native is 262 k; conservative — multimodal KV grows fast at long ctx).
- `--video-pruning-rate 0.5` and `--media-io-kwargs '{"video":{"fps":2,"num_frames":256}}'` per model card.
- No `--attention-backend FLASHINFER`. Mamba2 hybrid path may not be covered; let vLLM pick the default.
- No `--rm` on the container (same crash-traceability rationale as §25).

**Cold start ~150 s** in steady state (warm HF cache, post-image-pull). First-ever load on a fresh image was ~10 min including the docker pull and CUDA graph capture.

## Operations

```sh
# Service
sudo systemctl start|stop|restart|status llama-swap

# Logs (llama-swap itself)
tail -f ~/llm-stack/logs/llama-swap.log
tail -f ~/llm-stack/logs/llama-swap.err

# Logs (active backend container)
docker logs -f vllm-qwen-27b-dflash      # qwen3.6-27b (production coding default since 2026-04-30, NO --rm — see §25/§26)
docker logs -f vllm-qwen-27b             # qwen3.6-27b-mtp (DORMANT rollback slot, NO --rm — see §25/§26b)
docker logs -f vllm-qwen                 # qwen3.6-35b-a3b (DORMANT MoE, NO --rm — see §25)
docker logs -f vllm-qwen-distill         # qwen3.5-35b-distill
docker logs -f vllm-qwen-nvfp4           # qwen3.5-122b-nvfp4
docker logs -f llama-gemma-26b           # gemma-4-26b-a4b
docker logs -f llama-gemma-e4b           # gemma-4-e4b
docker logs -f llama-minimax             # minimax-m2.7
docker logs -f llama-supergemma          # supergemma-4-26b
docker logs -f vllm-nemotron-omni        # nemotron-3-nano-omni (NO --rm — see §25/§29)

# Current state
curl http://192.168.1.12:8080/running     # which model is hot
curl http://192.168.1.12:8080/v1/models   # model list
docker ps --filter name=llama- --filter name=vllm-

# Force an unload without stopping the service
curl -X POST http://192.168.1.12:8080/unload
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
6. **Save** — that's it. llama-swap runs with `-watch-config` (see §27) and reloads within ~1 s without process restart, preserving running model containers. Verify via `curl -s http://localhost:8080/v1/models`. Fallback for binary/unit changes: `pkill -9 llama-swap` (SIGKILL → systemd respawn; SIGTERM does not).
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
│   ├── launch-vllm-qwen.sh             # launcher for qwen3.6-35b-a3b MoE (DORMANT — on-demand via llama-swap)
│   ├── launch-vllm-27b-dflash.sh       # launcher for qwen3.6-27b NVFP4+DFlash (ACTIVE production default since 2026-04-30, see §26)
│   ├── launch-vllm-27b-nvfp4.sh        # launcher for qwen3.6-27b-mtp NVFP4+MTP (DORMANT rollback slot, see §26b)
│   ├── launch-vllm-nemotron-omni.sh    # launcher for nemotron-3-nano-omni multimodal (active, see §29)
│   ├── bench-models.py                 # quick benchmark (cold start + tok/s)
│   ├── bench-deep.py                   # deep benchmark (TTFT, decode, concurrency)
│   ├── bench-compare-deep.py           # before/after comparison report
│   ├── bench-concurrency-sweep.py      # ad-hoc concurrency sweep on a warm model
│   ├── bench-context-sweep.py          # context-length sweep (TTFT, prefill, decode by ctx)
│   ├── bench-multimodal-smoke.py       # multimodal smoke (single image + single audio)
│   └── bench-multimodal-large.py       # multimodal scaling (image / audio / video at multiple sizes)
├── systemd/
│   ├── llama-swap.service          # main unit (ExecStartPre/Post cleanup hooks)
│   └── llama-swap.service.d/
│       └── watch-config.conf       # drop-in: enables -watch-config (§27)
├── venv/                           # python + huggingface_hub + hf_transfer
├── logs/
│   ├── llama-swap.log              # gateway stdout
│   ├── llama-swap.err              # gateway stderr
│   ├── bench-results.json          # quick benchmark results
│   ├── bench-deep-latest.json      # latest deep benchmark (symlink)
│   └── bench-deep-*.json           # timestamped deep benchmark runs
└── README.md

~/bin/llama-swap                                            # gateway binary (v201)
/etc/systemd/system/llama-swap.service                      # main unit
/etc/systemd/system/llama-swap.service.d/watch-config.conf  # drop-in (§27)
~/.cache/huggingface/hub/                                   # model weights (max:max ownership required)
~/.cache/huggingface/token                                  # HF auth, chmod 600
```

## MTP patch (AlphaOxO Qwen3.6-27B-NVFP4)

> **As of 2026-04-30 this only applies to the dormant `qwen3.6-27b-mtp` rollback slot (§26b).**
> The active production default `qwen3.6-27b` runs DFlash on AEON-7's NVFP4 weights and
> does not need this patch. Re-apply only if you reactivate the MTP launcher.

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
| `AssertionError: In Mamba cache align mode` | Qwen3.5 / Nemotron-Omni + prefix caching needs larger batch | `--max-num-batched-tokens 8192` |
| `PlaceholderModule should not be used` (audio path) | vLLM cached the missing-module placeholder before `av`/`soundfile` were installed | Rebuild the omni image with the deps baked in (see §29) — hot-install + restart isn't enough; you need a fresh image |
| `TimeoutError: VLLM_ENGINE_READY_TIMEOUT_S` | 600 s default too short for large models | `-e VLLM_ENGINE_READY_TIMEOUT_S=1800` |
| `ValueError: Tokenizer class TokenizersBackend` | Repo exported on transformers v5 | patch cached `tokenizer_config.json` to `"Qwen2TokenizerFast"` (see §19) |
| `OSError: Can't load image processor` | vLLM treats Qwen3.5 as multimodal; repo missing preprocessor | `--language-model-only` (see §16) |
| Flood of `Skipping tactic` / `Failed to initialize cutlass` | FlashInfer MoE autotuner on GB10 | expected, non-fatal. For 122B-NVFP4 use `--moe-backend flashinfer_cutlass` |
| `--parallel N` gives less context than expected | llama.cpp divides `-c` total across N slots | set `-c` to `desired_per_slot × N` (see §23) |
| Gemma 4 garbage output via vLLM FP8 | vLLM #39407: double-quantization bug | use llama.cpp with GGUF instead (see §22) |
| Gemma 4 extremely slow via vLLM | vLLM #38887: heterogeneous head dims force Triton fallback | use llama.cpp with GGUF instead (see §22) |
| Vision not working on Qwen 122B | llama.cpp CLIP graph unsupported ops (#21268) | use Gemma E4B for vision tasks instead |
