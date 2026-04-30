# Qwen3.6-27B + DFlash on DGX Spark (GB10) — Deployment Notes

Last updated: 2026-04-30

## TL;DR

Production coding default since 2026-04-30. Replaces the prior `qwen3.6-27b` NVFP4+MTP path (which is kept dormant as `qwen3.6-27b-mtp` for one-line rollback). Stack:

- **Target weights**: `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4` (~26 GB on disk; lossless abliterated NVFP4 build with multimodal towers preserved BF16 — launched here language-model-only).
- **Drafter**: `z-lab/Qwen3.6-27B-DFlash` v2 (~3.3 GB on disk; refresh pushed 2026-04-27 with mean accepted length per round 2.60).
- **Image**: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3` (vLLM 0.20.1.dev0+g88d34c640 with 5 patches; FlashInfer 0.6.9rc1 — first SM121 NVFP4 GEMM path).
- **Spec method**: DFlash k=15 via `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'`.

Head-to-head bench vs the prior MTP n=3 path on Spark, same gateway, same `bench-concurrency-sweep.py` SHORT_MSGS profile, sequential cold-runs evicted via llama-swap:

| c | DFlash agg tok/s | MTP agg tok/s | Δ |
|---|---|---|---|
| 1 | **41.0** | 20.3 | **+102 %** (2.0×) |
| 5 | **139.0** | 92.0 | **+51 %** |
| 10 | **207.1** | 169.1 | **+22 %** |

DFlash wins everywhere. Per-burst peaks (vLLM `loggers.py` instantaneous Avg generation throughput across the running batch) show even higher gains: c=1 79.5 vs 22.0 (3.6×), c=10 264.6 vs 182.4 (1.45×). TTFT lower across the board.

## Hardware reality

| | |
|---|---|
| Box | Lenovo ThinkStation PGX (NVIDIA GB10 / DGX Spark, 2026-04) |
| Unified memory | 124,545 MiB total (~119 GiB usable for engines) |
| Memory bandwidth | 273 GB/s LPDDR5X |
| Compute | sm_121a Blackwell, ~1 PFLOP sparse FP4 tensor |
| CUDA toolkit | 13.0.88 at /usr/local/cuda-13.0 |

Decode for a 27B dense model in NVFP4 (~14 GB read per token at the weight layer) is bandwidth-limited on Spark. Bandwidth ceiling for raw 27B-NVFP4 decode at c=1 is roughly 273 GB/s ÷ 14 GB ≈ 19.5 tok/s. The MTP launcher's 19 tok/s c=1 number sits exactly on that ceiling. DFlash's 41 tok/s c=1 is **above the bandwidth ceiling for non-speculative decode** because each memory round produces multiple accepted output tokens (the drafter is small and stays in cache, the verifier reads the target weights once per accepted block). That's the architectural reason DFlash beats MTP here: the drafter's quality (block-diffusion, k=15) is high enough that the average tokens-per-memory-round goes from MTP's ~3.0 to DFlash's higher effective rate.

## Why DFlash, not MTP — the architectural argument

Both methods are speculative decoders on top of the same target. They differ in how they propose draft tokens:

- **MTP** (Multi-Token Prediction): the target model itself has trained MTP heads that predict the next k tokens autoregressively in a shared forward pass. Acceptance rate decays sharply with position (94 % / 80 % / 63 % at n=3 in our 04-24 sweep). Effective tokens-per-round mean 3.0 / 4. Drafter cost ≈ 0 because it's the target.
- **DFlash** (Block-Diffusion Flash): a separate small drafter (~2 B params, BF16, ~3.3 GB) generates a full block of k tokens in *one parallel* forward pass via block diffusion. Verifier still reads target weights once. Drafter cost is small but non-zero. Acceptance is higher and flatter across the block because the drafter is purpose-trained for the architecture and uses the target's hidden states as context features.

On Spark specifically:
- **GB10 unified-memory bandwidth is the bottleneck** for decode. Anything that increases tokens-per-memory-round wins.
- **MTP's drafter cost is ~zero** but its acceptance decays. Net: 3.0/4 effective tokens at c=1.
- **DFlash's drafter is ~3 GB extra read per round** but its acceptance is high enough that net tokens per round exceeds MTP's.
- The win is largest at low concurrency (c=1, where decode is purely bandwidth-bound) and narrows at higher concurrency (c=10, where batch effects start to dominate over per-stream speculation gains).

This inverts the picture on dedicated-VRAM hardware (RTX 5090/6000), where MTP's lower drafter cost makes it +10 % over DFlash. Same code, different memory hierarchy, different winner.

## What we actually deployed

`config/llama-swap.yaml` entry `qwen3.6-27b`:

```yaml
qwen3.6-27b:
  cmd: /home/max/llm-stack/bin/launch-vllm-27b-dflash.sh
  proxy: http://127.0.0.1:9013
  checkEndpoint: /health
  ttl: 3600
  aliases:
    - qwen3.6-35b-a3b      # legacy opencode sessions land here
    - qwen3.6-27b-dflash   # explicit DFlash name still valid
```

Launcher `bin/launch-vllm-27b-dflash.sh` (key flags only — see file for the full set):

```bash
docker run --name vllm-qwen-27b-dflash \
  --runtime=nvidia --gpus all --ipc=host \
  -p 0.0.0.0:9013:9013 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3 \
  bash -c '
    exec vllm serve AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4 \
      --served-model-name qwen3.6-27b qwen3.6-35b-a3b qwen3.6-27b-dflash \
      --host 0.0.0.0 --port 9013 \
      --quantization compressed-tensors \
      --max-model-len 200000 \
      --max-num-seqs 16 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.85 \
      --enable-chunked-prefill --enable-prefix-caching \
      --attention-backend flash_attn \
      --language-model-only \
      --speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":15}"
  '
```

Three deliberate deviations from AEON-7's published `docker-compose.yml`:

1. **Drop `--reasoning-parser qwen3`** — AEON-7's compose has it; we omit it. Their parser-on TTFT (247 ms claimed) is measured to first reasoning chunk; with parser off, all chunks land in `content` and opencode parses `<think>` client-side. Avoids the 3 s buffered TTFT path documented in `feedback_reasoning_parser_ttft`.
2. **Add `--language-model-only`** — text-only coding workload. Multimodal towers stay on disk but don't get loaded.
3. **`--served-model-name` accepts three names** — default `qwen3.6-27b`, legacy MoE alias `qwen3.6-35b-a3b` (so old opencode persisted defaults still land here), and explicit `qwen3.6-27b-dflash`.

## Bench results (2026-04-30, full)

`bin/bench-concurrency-sweep.py`, model name `qwen3.6-27b-dflash` and `qwen3.6-27b` (MTP path renamed `qwen3.6-27b-mtp` and re-benched fresh on the same day):

```
Sweep: qwen3.6-27b-dflash  levels=[1, 5, 10]
  c  agg_tok/s  per_req  ttft_ms  lat_s wall_s errs
------------------------------------------------------------
  1       41.0     41.0      394   12.5   12.5    0
  5      139.0     29.2     1428   17.6   18.4    0
 10      207.1     22.4      703   22.9   24.7    0

Sweep: qwen3.6-27b (MTP n=3)  levels=[1, 5, 10]
  c  agg_tok/s  per_req  ttft_ms  lat_s wall_s errs
------------------------------------------------------------
  1       20.3     20.3      446   25.2   25.2    0
  5       92.0     18.4     2070   27.8   27.8    0
 10      169.1     17.0      978   30.1   30.3    0
```

vLLM engine `loggers.py` instantaneous burst peaks during the sweep (Avg generation throughput across the running batch):

| c | DFlash burst | MTP burst | ratio |
|---|---|---|---|
| 1 | 79.5 tok/s | 22.0 | 3.6× |
| 5 | 175.8 | 106.0 | 1.66× |
| 10 | 264.6 | 182.4 | 1.45× |

Bench-script aggregate is lower than burst because it averages over the full request including TTFT and setup; bursts are instantaneous mid-batch. Both numbers are useful — bench-script is what users perceive end-to-end, burst is the engine ceiling.

AEON-7's published numbers (model card + repo README) for the same image: median 38.1 / peak 68.4 tok/s c=1 with `--reasoning-parser qwen3` on. Our 41.0 / 79.5 numbers (parser off) reproduce within noise, with a slight upside from skipping the parser's stream buffering.

## Operational gotchas

- **Cold start ~10 min** (vs MTP's ~6 min). FlashInfer NVFP4 GEMM autotuner + CUDA-graph capture across 51 sizes (1, 2, 4, 8, ... 512). Both cache to `/root/.cache/vllm/...` *inside* the container, which is **not** bind-mounted, so caches are rebuilt on every container start. **First request after llama-swap evicts to a different `main`-group model waits the full 10 min** — same swap-eviction model as before, just longer per-cold-start. If real-traffic volume is low (`ttl: 3600` rarely triggers) this only matters at deployment time.
- **KV budget shrinks**: vLLM at boot reports GPU KV cache size 205,632 tokens for DFlash vs 552,000 for MTP. Max concurrency at full ctx is 2.69× (200 k DFlash) vs 7.73× (262 k MTP). Fine for the current ~100 k:4 k coding workload at c≤2 long-context users, tighter than MTP if traffic ever pushes >2 concurrent users at near-200 k input.
- **`max-model-len` capped at 200 000, not 262 144**. AEON-7's recommendation; DFlash's drafter-side allocations make 262 k uneconomical. Existing opencode requests within 200 k input are fine; the 04-26 incident where a 123 k+8 k=131 073 request was rejected at the old 131 072 ceiling stays resolved.
- **AEON-7 image is multimodal-capable** (vision encoders preserved BF16 in NVFP4 build). Currently launched `--language-model-only` to mirror the prior text-only prod. To enable vision in place, drop that flag and add `--limit-mm-per-prompt '{"image":4,"video":2}'` and the AEON-7 compose's `--mm-encoder-tp-mode data --mm-processor-cache-type shm`. No re-download needed; expect KV budget to tighten further.
- **Drafter is auto-gated on Hugging Face**. First download prompts a click-accept page (instant approval). Already accepted on this Spark.
- **AEON-7 finetune persona drift**. Target weights are AEON-7's "Ultimate Uncensored" abliterated finetune of base Qwen3.6-27B, not the AlphaOxO weights. KL divergence to BF16 source is ≤ 0.001 per AEON-7's docs, but persona / refusal-posture is different from AlphaOxO's. If opencode output ever looks oddly over-helpful or off-tone, this is the lever to be aware of. Rollback to MTP slot (`qwen3.6-27b-mtp`) gets back to AlphaOxO weights.
- **No `--rm` on the container** (same crash-traceability rationale as §25 of the main README). `docker logs vllm-qwen-27b-dflash` after a crash works.

## What we are NOT doing and why

- **Not running `--reasoning-parser qwen3`** — the parser buffers the entire `<think>…</think>` block before emitting any `delta.content` to the stream (see `feedback_reasoning_parser_ttft`). On reasoning-heavy prompts that shows up as multi-second TTFT in opencode, where users expect immediate first-chunk response. opencode parses `<think>` client-side anyway. AEON-7 leaves it on in their published bench, so our reproduced numbers run slightly higher than theirs.
- **Not sweeping `num_speculative_tokens` again** — AEON-7 already published k=15 as the dense-27B optimum and our bench at k=15 already wins the head-to-head. k-sweep would be incremental tuning; revisit only if real-traffic acceptance metrics regress.
- **Not bind-mounting `/root/.cache/vllm`** to persist FlashInfer autotuner / CUDA-graph captures across container restarts. Risk: cache poisoning if image changes (vLLM version bump → silent kernel mismatches). Cold-start cost (~10 min) is paid rarely enough that the trade isn't worth it yet. Revisit if cold starts become frequent.
- **Not removing the MTP slot from disk**. Kept dormant for fast rollback (60 s yaml edit, no downloads). The AlphaOxO weights (~14 GB) and the launcher cost almost nothing.

## Rollback recipe

In `~/llm-stack/config/llama-swap.yaml`, the `qwen3.6-27b` block:

```yaml
qwen3.6-27b:
  cmd: /home/max/llm-stack/bin/launch-vllm-27b-dflash.sh   # ← swap to launch-vllm-27b-nvfp4.sh
  proxy: http://127.0.0.1:9013                              # ← swap to 9008
```

Save → `-watch-config` hot-reloads → next request to `qwen3.6-27b` cold-starts MTP (~6 min). Or skip the rollback entirely: send all traffic to the explicit dormant name `qwen3.6-27b-mtp` (already wired). Safety backups: `config/llama-swap.yaml.bak.20260430-pre-dflash`, `bin/launch-vllm-27b-nvfp4.sh.bak.20260430-pre-dflash`.

## References

- z-lab DFlash project page: https://z-lab.ai/projects/dflash/
- DFlash paper: https://arxiv.org/abs/2602.06036
- z-lab/Qwen3.6-27B-DFlash drafter: https://huggingface.co/z-lab/Qwen3.6-27B-DFlash
- AEON-7 NVFP4 + DFlash repo: https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash
- AEON-7 NVFP4 weights: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4
- AEON-7 vLLM image (DFlash): `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v3`
- vLLM PR adding DFlash: https://github.com/vllm-project/vllm/pull/40898
- NVIDIA Developer Forums Spark thread: https://forums.developer.nvidia.com/t/qwen3-6-27b-dflash-link/367803
- "Speculative Decoding Handbook" (DFlash vs MTP on different hardware): https://dasroot.net/posts/2026/04/speculative-decoding-dflash-lorbus-mtp-speed/
