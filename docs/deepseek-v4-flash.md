# DeepSeek-V4-Flash on DGX Spark (GB10) — Deployment Notes

Last updated: 2026-08-13

> **Status: V4-Flash is the production coding default** (since 2026-08-01), on the
> **Entrpi/ds4 fork v0.5.6.2** at `ctx 131072`. Everything from "Historical" down
> is the 2026-04 llama.cpp-fork era, preserved because its bandwidth analysis is
> still the reason this lane behaves the way it does. Read
> [Engine lineage](#engine-lineage-2026-05--2026-08) first — the 2026-04 numbers
> (3 GEN tok/s, 64 k cap, "parked specialty model") were all superseded.

## Engine lineage (2026-05 → 2026-08)

The model never changed shape; the *engine* did, five times. Every number below is
measured on this box, same GGUF family, coding-shaped prompts.

| When | Engine | Prefill | Decode | Why it changed |
|---|---|---:|---:|---|
| 2026-04-29 | `phuongncn` llama.cpp fork | ~32 t/s | ~3 t/s | bandwidth wall, 64 k cap, parked specialty lane |
| 2026-05-12 | antirez/ds4 (from-scratch C/CUDA) | ~96 t/s | ~13 t/s | purpose-built engine + persistent disk-KV; 131 k ctx |
| 2026-05-18 | ngc-shj/ds4 fork (Q4 dp4a) | 396 t/s | 20.8 t/s | opt-in Q4 decode path; cost `--mtp` (Q4 gates on `n_tok == 1`) |
| 2026-08-06 | antirez mainline `b030961` | **857 t/s** | 14.2 t/s | aligned-artifact repack kills the i-quant dequant wall (§39) |
| 2026-08-10 | **Entrpi/ds4 v0.5.6.2** | (32.7 s TTFT @34.6 k) | **19.6 t/s** | best decode without speculation; current prod (§40) |

Weights moved once in that span: the V4-Flash **preview** quant → the official
**0731** release (2026-08-01), same recipe, imatrix recalibrated on 0731. Parity on
speed, smoke 9/10, and it is what production serves today.

**The load-bearing constraints, all still true:**

- **The quant recipe is hard-pinned by the engine.** `ds4.c` accepts only
  IQ2_XXS / Q2_K / Q4_K experts + Q8_0 shared-expert + F16 router. No unsloth
  `UD-*` quant can ever load on ds4 — those must go to mainline llama.cpp, which
  has had `deepseek4` support since 2026-06-29 (the old `~/llama.cpp-gx10-dsv4`
  fork runs DSV4 ops CPU-only, ~10× slower). Measured there: UD-IQ3_XXS at
  473 t/s prefill / 16.2 t/s decode @2 k.
- **Prefill-bound, not decode-bound.** Traffic is ~100 k:4 k, so an engine that
  trades 18 % decode for 2.33× prefill wins whenever `prompt/output > ~8.4`.
- **The disk-KV prefix cache is load-bearing** (~6.3× TTFT on a warm prefix).
  Each engine gets its own `--kv-disk-dir`; cross-engine format compatibility is
  unverified.
- **No concurrency.** c=4 aggregate is 0.92× of c=1, fully serialized
  (`tok_per_step` 1.000). Upstream calls it planner-only for this reason.
- **Context is capped at 131072 by an outage, not by the model.** 262144 grew
  demand-mapped KV slabs until the memory floor refused every deep request — 33
  refusals, zero completions, ~35 min, all health checks green (§41).
- **Benchmark hygiene specific to this lane:** the disk-KV replay poisons prefill
  benches (use a unique token-0 prefix per run), and GB10 decode is dominated
  ~44× by allocator state (idle 2–3 min between runs, 3-run medians).

Full reasoning for each step: [decisions §30, §36–§41](decisions.md#36-ds4-is-the-coding-default-laguna-pulled-2026-08-01).

## Historical: the 2026-04 llama.cpp-fork era

*Everything below is from 2026-04-29 and is kept for the bandwidth analysis and
the NVFP4 watchlist. The deployment described here no longer exists.*

## TL;DR

V4-Flash runs on a single Spark only via the heaviest community lossy quant
(`antirez/deepseek-v4-gguf` IQ2_XXS, ~87 GB) on a custom llama.cpp fork. It
serves at ~32 PP tok/s and ~3 GEN tok/s — at the bandwidth wall for an
87 GB read per token on Spark's 273 GB/s LPDDR5X. Hard ceiling: **64k
context**. 100k requests get refused (HTTP 400, soft-OOM in the
allocator). A vLLM-loadable NVFP4 quant that fits 119 GB does not exist
yet — expected mid-May → early-June 2026 via Red Hat / Neural Magic
(llm-compressor PR #2655). Until then, V4 is a parked specialty model
behind llama-swap, production stays on `qwen3.6-27b`.

## Hardware reality

| | |
|---|---|
| Box | Lenovo ThinkStation PGX (NVIDIA GB10 / DGX Spark, 2026-04) |
| Unified memory | 124,545 MiB total (~119 GiB usable for engines) |
| Memory bandwidth | 273 GB/s LPDDR5X |
| Compute | sm_121a Blackwell, ~1 PFLOP sparse FP4 tensor |
| CUDA toolkit | 13.0.88 at /usr/local/cuda-13.0 |

Bandwidth-limited decode ceiling for 13B-active V4 at FP8: 273 / 13 = 21
tok/s. For our IQ2_XXS at 87 GB read per token ceiling is ~3 tok/s, which
matches what we measure.

## V4-Flash architecture (relevant facts)

- **284B total / 13B active MoE.** 256 routed experts + 1 shared, top-6 routing, expert intermediate 2048, `noaux_tc` routing.
- 43 layers, hidden 4096, 64 attention heads.
- **Attention: MLA with q_lora_rank=1024, o_lora_rank=1024.** The variant uses CSA/HCA (Compressed/Heavily-Compressed Attention) — *not* Gated Delta Net (the phuongncn fork README is misread).
- Native release on HF (`deepseek-ai/DeepSeek-V4-Flash`, 2026-04-22): FP8 e4m3 (block 128x128) for attention/router/norm, MXFP4 for expert weights — **~158 GB total**. Does not fit single Spark.
- Context: 1,048,576 (yarn 16x scaling, base 65,536).

## What we actually deployed

`config/llama-swap.yaml` entry `deepseek-v4-flash`:

- Binary: `/home/max/llama.cpp-gx10-dsv4/build/bin/llama-server` from
  the `phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4` fork (the only
  contribution vs upstream/nisparks: bumps ggml metadata pool from
  +2048 to +450,000 slots to prevent overflow in
  `dsv4_build_compressor_decode_chunk` at n_ubatch=512).
- Weights: `antirez/deepseek-v4-gguf` IQ2_XXS, mixed recipe
  (w2=Q2_K, AProj/SExp/Out=Q8), 86.7 GB on disk.
- **`--ctx-size 65536`** (capped from 128k after observing soft-OOM).
- `--parallel 1` (multi-seq trips the compressor assert — required).
- `-ctk f32 -ctv f32` (the fork mandates this; V4 compressor is
  numerically unstable with f16 KV — pure 2x tax on KV bandwidth).
- `-b 4096 -ub 512`, `-fa on`, `--no-mmap`, `--reasoning-budget -1`,
  `--jinja`.
- Serves under `main:` swap-exclusive group at port 9011, 1h TTL.

## Bench results (2026-04-29, standalone llama-server)

| Test | Prompt tok | PP tok/s | GEN tok/s | Outcome |
|---|---|---|---|---|
| smoke (~30 prompt) | 17 | 9.4 | 5.9 | OK |
| medium (~10k) | 9015 | 48.7 | 2.0 | OK |
| 32k context | 42702 | 31.71 | 2.91 | OK |
| 64k context | 85368 | — | — | client timeout @ 60min (server still alive, prefill near done) |
| 100k context | ~100,000 | — | — | **HTTP 400 in 27s — server allocator refused** |

Memory breakdown at end of run, captured by llama.cpp itself:

```
CUDA0 (GB10) | 124,545 = 15,154 free + (89,703 self = 81,687 model + 1,829 ctx + 6,186 compute) + 19,687 unaccounted
```

The "unaccounted" 19.7 GB is the prompt cache + context checkpoints
growing during prefill (each 8k chunk creates a 0.3-1.1 GB checkpoint,
up to 32 max). Compute buffer ballooned 22x expected size at 100k. This
is the failure mode at high context — not a kernel OOM-kill, the
server's internal allocator declines new requests once it can't
satisfy the new compute buffer reservation. Functionally similar to
OOM from the client perspective.

**Why 64k cap:** observed safe through 64k bench; soft-OOM consistently
at 100k. A safer engineering margin would be 48k, but 64k is round.

## What we are NOT doing and why

- **Not pursuing K-quants** (Preyazz Q2_K, etc.). Tested: Preyazz Q2_K
  is metadata-incompatible with the phuongncn fork (missing
  `deepseek4.attention.output_lora_rank`). Even if it loaded, it
  shares the same bandwidth wall.
- **Not building IQ3_XXS overnight** from Preyazz Q8_0 source. Same
  IQ-grid kernel family, same throughput class. Not worth the build cost.
- **Not switching to nisparks branch.** Loses the metadata pool fix
  needed for high-context stability; gains nothing in throughput.
- **Not pursuing single-Spark vLLM with native FP8.** The 158 GB
  release does not fit in 119 GB — period.

## The realistic path forward

| When | What | Why |
|---|---|---|
| **Today** | Keep IQ2_XXS @ 64k cap behind llama-swap; production on `qwen3.6-27b`. | Spark single-node V4 is near hardware ceiling. Nothing to optimize at this layer. |
| **Mid-May → June 2026** | Watch for vLLM-loadable NVFP4 V4 (~80-95 GB). | Red Hat / Neural Magic llm-compressor PR #2655 already in flight; 5-layer test artifact published 2026-04-28. |
| **If second Spark added** | vLLM TP=2 with `sgl-project/DeepSeek-V4-Flash-FP8`, `FLASHINFER_MLA_SPARSE`, fp8 KV. | Confirmed working on dual-Spark by 120jpy on NVIDIA dev forum (~20 tok/s gen). |
| **Watch** | NVIDIA NIM container `deepseek-v4-flash` to add GB10 to its tested-hardware list (currently A100/H100/H200/B200 only). | NIM ships pre-tuned, would skip all the ops work. |

## NVFP4 V4 watchlist

| URL | Why |
|---|---|
| https://github.com/vllm-project/llm-compressor/pull/2655 | Most concrete NVFP4 V4 signal — Neural Magic / kylesayrs draft |
| https://huggingface.co/inference-optimization | Neural Magic staging org — full NVFP4 V4 will land here first |
| https://huggingface.co/RedHatAI | Production drop after PR #2655 merges |
| https://huggingface.co/nvidia (filter: deepseek-v4) | NVIDIA's TRT-LLM-first NVFP4 |
| https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/1346 | NVIDIA modelopt feature request |

Historical NVFP4 cadence after DeepSeek native release:

| Model | Native | First NVFP4 | Lag |
|---|---|---|---|
| R1 | 2025-01-20 | 2025-02-21 | ~1 month |
| V3-0324 | 2025-03-24 | 2025-05-03 | ~6 weeks |
| R1-0528 | 2025-05-28 | 2025-06-03 | ~1 week |
| V3.1 | ~2025-11 | 2025-11-21 | ~days |
| V3.2 | ~2025-12-01 | 2026-01-20 | ~7 weeks |
| **V4-Flash** | **2026-04-22** | **expected mid-May → early-June** | |

## References

- Native release: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- IQ2_XXS GGUF in use: https://huggingface.co/antirez/deepseek-v4-gguf
- Fork in use: https://github.com/phuongncn/llama.cpp-gx10-dgx-sparks-deepseekv4
- Upstream V4 PR (reference, not for merge): https://github.com/ggml-org/llama.cpp/pull/22378
- Dual-Spark vLLM benchmark thread: https://forums.developer.nvidia.com/t/deepseek-v4-released/367696
- vLLM recipe for V4-Flash: https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- LMSYS Day-0 review (SGLang on H200): https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- LMSYS Spark in-depth: https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
