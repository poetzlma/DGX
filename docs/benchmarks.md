# Benchmarking

The bench tooling (all in `bin/`), plus the historical benchmark archive that
documents each production era. Methodology gotchas worth knowing before running
anything: spec-decode must never be benched with random text (acceptance
collapses on noise — [decisions §33](decisions.md#33-laguna-s-21-is-the-coding-default-v2-spinquantless-weights-2026-07-22--07-23)), and GB10 decode numbers
are dominated ~44× by allocator state — idle 2–3 min between runs and use
3-run medians or the numbers are noise.

Three more, learned the hard way on the ds4 lane: **the disk-KV cache replays
prefixes**, so a prefill bench repeats itself into fiction unless every run gets a
unique token-0 prefix; **speed is the easy half** — an engine swap can be faster
and *wrong*, so `bin/smoke-ds4-0731.py` gates every cutover (code-exec, exact
arithmetic, needle, tool-calls); and **a rejection expires with the commit it was
measured on** — DSpark measured −23 % on one mainline commit and +11.5 % on
another four days later ([§40](decisions.md#40-entrpids4-fork-is-production-dspark-measured-twice-2026-08-10)).

## Bench tooling

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
| `bin/bench-ds4-0731.py` | ds4 prefill **and** decode, streaming, context-swept — real source text, unique prefix per run | `DS4_PORT=9010 python3 bin/bench-ds4-0731.py [--quick]` |
| `bin/bench-concurrency.py` | Concurrency without assuming parallel prefill (ds4 serializes; an "aggregate" over serialized prefills is meaningless) | `python3 bin/bench-concurrency.py --port 9010 --levels 1,2,4 --model deepseek-v4-flash-0731` |
| `bin/smoke-ds4-0731.py` | **Correctness** gate for a weights/engine swap: code-exec, exact arithmetic, needle @30 k, tool-calls | `DS4_PORT=9010 python3 bin/smoke-ds4-0731.py` |
| `bin/probe-commits-to-action.py` | Does the lane commit to an action or plan forever? (the failure that ended the laguna era) | `MODEL=deepseek-v4-flash-0731 python3 bin/probe-commits-to-action.py` |

Results go to `logs/bench-*.json` and `logs/bench-*.log` with timestamped names. `logs/bench-deep-latest.json` is the symlink to the most recent deep run.

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

Wins on short-context but acceptance collapsed on long-context coding (37 % @ 13 k → 6 % @ 100 k). Demoted. Deep writeup: [`docs/qwen3.6-27b-dflash.md`](qwen3.6-27b-dflash.md).

### Real-traffic-shaped A/B (2026-05-08, drove the AM rollover)

| Bucket | AEON DFlash k=4 | sakamaki MTP n=3 | Δ wall-clock |
|---|---:|---:|---:|
| 13 k / 1024 out | 75.0 s | 50.6 s | **−32 %** |
| 60 k / 1500 out | 274.4 s | 132.8 s | **−52 %** |
| 100 k / 2048 out | 534.6 s | 230.2 s | **−57 %** |

Decode tok/s @100 k: 4.9 (DFlash) → 18.5 (MTP), +278 %. Acceptance @100 k: 6.0 % vs 66.2 %. Full report: `logs/bench-coding-AB-report-20260508.md`.

### 2026-05-17 quant comparison (Qwen 27B era, prod 05-17 → 07-08)

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

### ds4 / V4-Flash era (2026-08-01 → current production)

Engine A/Bs on the same 0731 IQ2_XXS GGUF, coding-shaped prompts, `--temp 0`,
decode timed first→last token so prefill is excluded.

| Engine | prefill @34.6 k | decode @34.6 k | decode short | TTFT @34.6 k |
|---|---:|---:|---:|---:|
| ngc-shj fork (prod 05-18 → 08-06) | 368 t/s | 17.38 | 17.95 | — |
| antirez mainline `b030961` | **857 t/s** | 14.18 | — | 40.2 s |
| mainline + DSpark, commit `84cc882` | — | 10.93–13.84 | — | — |
| mainline + DSpark, commit `0e89a0e` | — | +11.5 % vs plain | — | — |
| **Entrpi fork v0.5.6.2, `--no-spec` (prod)** | — | **19.55** | **20.14** | **32.7 s** |
| Entrpi fork + DSpark | — | 17.90 (−8.4 %) | +9.9 % | — |

At 100 k the production lane runs ~14.3 tok/s decode with ~60 s TTFT on a warm
disk-KV prefix (~370 s cold). Concurrency: c=4 aggregate **0.92× of c=1**, fully
serialized — at c=4 every stream's TTFT was 67 s on an 8 k prompt.

Quant candidates evaluated 2026-08-02 (none promoted):

| Quant | Size | Engine | Prefill @2 k | Decode @2 k | Verdict |
|---|---:|---|---:|---:|---|
| IQ2_XXS-0731 (prod) | 81 GB | ds4 | 404 t/s | 19–20 | incumbent |
| Q4K-hybrid-0731 | 97.6 GB | ds4 | — | — | evaluated during the parked window; not promoted |
| unsloth UD-IQ3_XXS | 104 GB | llama.cpp mainline | 473 t/s | 16.2 | **cannot load on ds4** — engine rejects the layout |

For the pre-ds4 production eras (Qwen 27B, Nemotron 75B, Laguna) see the tables
above; the coding default's full lineage is [decisions §26–§41](decisions.md).
