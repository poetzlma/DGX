# Decision log

Reasons for non-obvious config choices, in roughly the order they were made.
**Numbering is stable and preserved** — launcher scripts and yaml comments
cross-reference entries by §number.


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

Both ship native MTP weights. MTP-1 predicts 1 additional token per step with high acceptance. Added `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (MoE) or `qwen3_next_mtp`/`qwen3_5_mtp` (dense variants). **Requires `--attention-backend FLASHINFER` on the older NVFP4 paths** — MTP silently disables without it. The current dense prod uses DFlash (different drafter), not MTP — see [models.md](models.md#per-launcher-details). The 122B NVFP4 cannot use MTP — RedHatAI stripped the head during quantization.

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

### 32. Two-tier residents + codeserver copy-back offload *(2026-07-13)*

The Spark's 916 GB NVMe hit 95 %. Rather than delete any of the ~11 dormant eval/rollback models, their weights were moved to **codeserver** (`192.168.1.16`, 1.9 TB, 2.5 GbE) and are pulled back on demand — dropping local usage to ~48 % (**~408 GB freed, nothing deleted**). This also formalized the **two-tier** split implicit since the 2026-07-08 resident-pair revival:

- **Resident tier** (`resident` group, `persistent: true`, co-resident): `nemotron-3-puzzle-75b` + `qwen3.6-35b-a3b-vision` — always local, always loaded. Supersedes the "every model loads solo with the full 119 GB" framing of §1/§31 (which still holds for the dormant pool).
- **Dormant tier** (`experiments` group): weights on codeserver, each slot's `cmd:` wrapped by `bin/copyback-launch.sh` (pull-on-start, evict-on-stop, one-at-a-time via `etc/copyback-models.txt`).

Design choices worth recording: **evict-immediately-after-use** (leanest disk; re-pulls each cold start) was chosen over LRU retention — dormant models are rarely hit, so paying the pull each time beats holding tens of GB resident. **`healthCheckTimeout` raised to 1200 s** so a pull can't trip the health check mid-download (ds4's 85 GB GGUF is a ~13 min pull). **HF-cache blobs are root-owned** (docker populates the cache as root), so `rm` as `max` can't remove them — delete via a throwaway `docker run --rm -v ~/.cache/huggingface:/hf alpine rm -rf /hf/hub/models--…` (no sudo; `max` is in the `docker` group). **New dependency:** dormant models require codeserver online to load; the residents do not. Full writeup: `memory project_weights_offload_codeserver`.

### 33. Laguna S-2.1 is the coding default; v2 "spinquantless" weights *(2026-07-22 / 07-23)*

**poolside/Laguna-S-2.1-NVFP4** (118 B / 8.5 B-active MoE) + its **DFlash NVFP4 drafter** replaced `nemotron-3-puzzle-75b` as coding default on 2026-07-22 — 33 tok/s @100 k beat the 75B's 20, with 3.3× KV concurrency at the full native 256 k window. The old default names (`qwen3.6-27b`, `qwen3.6-35b-a3b`, `nemotron-3-puzzle-75b`) were remapped to it **at the LiteLLM gateway** (llama-swap v201 drops yaml aliases — see [operations.md](operations.md#litellm-integration)). The vision lane went **dark** — no headroom beside solo laguna.

On 2026-07-23 the lane moved to poolside's re-uploaded **"spinquantless norot" weights** (revision `0761412`, launcher `-v2.sh`): re-quantized without SpinQuant rotations, the apparent fix for the looping reports (HF discussions #4–#7, #10 — #11 suggests the original rotate checkpoint never ran correctly on public runtimes). Smoke-verified: looping gone, and DFlash decode *improved* (43–48 tok/s code/math). **Both launchers are revision-pinned** so an upstream re-upload can't silently change prod; v1 (`b482b5d`, original weights) is the one-line rollback. Lesson worth keeping: **never benchmark speculative decoding with a random-text harness** — acceptance collapses on noise and the numbers say nothing about real code traffic.

### 34. Laguna concurrency ceiling — seqs-8 deadlock; qwen co-residency parked *(2026-07-24)*

Dev-box asked for `max-num-seqs` 4 → 8 (DGX issue #2) to serve a multi-agent workload. The day produced four durable findings:

1. **DFlash n=15 + `max-num-seqs 8` deadlocks the vLLM engine core under real traffic** — twice reproduced (~48 k-token prompts, chunked prefill; token counter freezes while `/health` stays 200, so llama-swap never restarts it). A synthetic c=8 probe passed; only the real traffic mix triggers it. Drafterless seqs=8 is stable but ~19 tok/s single-stream vs 43–54 with the drafter — the drafter is worth more than the extra slots. **Community corroboration:** DFlash crashes vLLM outright at the default seqs=256; the one published stable config (MiaAI-Lab) pins seqs=4 / n=7. → `LAG_SEQS=4` is a hard ceiling while `LAG_SPEC=1`. Detection: stall watchdog on `generation_tokens_total` frozen while `num_requests_running > 0`.
2. **Co-residency math for GB10 (measured via 4 crash-loop attempts):** laguna's non-KV footprint is **~70.6 GB** (weights + drafter + runtime), KV costs **38.4 KB/token** fp8, and vLLM refuses to start unless one full `max-model-len` request fits in the KV pool. `util 0.66 / ctx 131072` is the smallest proven laguna shape (leaves ~30 GB for a neighbor).
3. **A second vLLM engine cannot load next to resident laguna.** The qwen 35B-A3B NVFP4 fast lane (`launch-vllm-35b-moe-nvfp4-colag.sh`) ran healthy once at 70 tok/s c=1, but vLLM's **weight-load transient spikes ~5–6 GB above its steady reservation** — MemAvailable hit 1 GB and the OOM guard killed it, twice. Since the **GB10 hard-hangs on host OOM** (no remote recovery), the lane is **parked**: yaml entry removed so nothing can launch it by name. Revival preconditions (in the launcher header): re-shrink laguna to 0.66/131 k *first*, then either `sudo drop_caches` pre-launch or switch the lane to llama.cpp/GGUF (mmap pages are reclaimable, unlike vLLM's anonymous allocation).
4. **Observability gap:** LiteLLM now sends **`/v1/responses`** (Responses API) for laguna traffic; `log-proxy` only parses `/v1/chat/completions`, so per-request logs and `traffic-tui` are currently **blind to prod traffic** (zero files written all morning despite healthy 200s). Until the proxy is extended, judge engine health from vLLM `/metrics` + llama-swap logs, not the proxy tree. A related client-side note: a long thinking generation at 20–26 tok/s legitimately runs 2–4 min — downstream harnesses need timeout budgets that don't misread it as a hang.

Config restored same evening to the proven solo shape: `util 0.85 / ctx 262144 / seqs 4 / DFlash n=15` (KV 856,686 tokens, 54.2 tok/s smoke, 2 h watchdog clean). Full audit trail on DGX issue #2; measured math in the launcher headers and `memory project_coresident_split_20260724`.

### 35. vLLM 0.26.0 trial — reverted; drafter revision pinned *(2026-07-30)*

An upstream-update sweep produced one config fix, two measured rejections, and an instrumentation bug fix. Every number below is from `bin/bench-coding-realistic.py` (13 k / 60 k / 100 k real-traffic buckets) against the live resident.

1. **The DFlash drafter was silently unpinned — now pinned.** The target had `--revision` but `speculative_config` named the drafter bare, so it resolved `main` on every engine start. poolside pushed drafter `4cdcc6e` ("Spinquant removal adaptation", 2026-07-27) and it reached prod unnoticed on the 07-28 bounce. The commit's only substantive change is `rope_theta` 10000 → 500000 — the drafter had shipped with a 50× RoPE-base mismatch against the target since release. **Measured payoff of the fix: none** (100 k decode 33.3 / 33.1 / 32.5 / 32.0 tok/s across four runs spanning both drafters) — the drafter's 6× sliding-window-512 attention bounded the damage. The pin is the real win: no upstream push can silently change prod again (`LAG_DRAFT_REV` to override).
2. **vLLM v0.26.0 trialled and reverted.** Thesis: decode is already healthy (accept_len 4.31 vs poolside's ~3.1 reference) but 100 k prefill eats half the wall clock (64 s TTFT @~1560 tok/s), so 0.26.0's MoE kernel work (`fused_topk_bias`, 1.5–2×) was the target. Result: TTFT unimproved at every context (13 k actually +20 %), acceptance *down* (accept_len 4.31 → 3.70), and spec-decode + FlashInfer forces `cudagraph_mode` FULL_AND_PIECEWISE → PIECEWISE on 0.26.0. sm_121 itself was fine on the stock `v0.26.0-aarch64-ubuntu2404` image (FlashInfer resolves `arch=sm121`, ~10 min cold start, no AEON build needed) — the image *works*, it just isn't better. Rollback = the yaml `cmd:` line.
3. **`speculative_config.kv_cache_dtype` (0.26.0's #48787) is catastrophic here — the drafter's KV dtype must MATCH the target's.** Giving the BF16 drafter bf16 KV against the fp8-KV target dropped acceptance to **6 tokens of 51,030 drafted (0.01 %, accept_len 1.002)** — every draft rejected, spec-decode reduced to pure overhead — and cost 22.7 % of the KV pool. Not documented in the PR or the vLLM recipe, which cite it as a free acceptance win for exactly this setup. `LAG_SPEC_KV` stays unset.
4. **`repetition_penalty=1.15` rejected as a looping mitigation.** HF #16 (runaway reasoning, upstream-open, reproduces with DFlash off) reports it as the strongest mitigation; A/B'd here with thinking on (`bin/ab-laguna-reppen.py`, 3 repro prompt classes × 2 penalties × 2 reps): rp=1.15 drove 5/6 runs into the length cap, produced the only genuine 12-gram loop of the day, and cost 32 % throughput. rp=1.0: 0/6 runaway, 0/6 looped. Community mitigations don't transfer — measure before adopting.
5. **Instrumentation: laguna emits `reasoning`, not `reasoning_content`.** The poolside_v1 parser names the thinking field `reasoning` in both the final message and streaming deltas. `log-proxy` matched only `reasoning_content`, so `ttft_s` started at the first *content* token — thousands of tokens late on every thinking response — and `reasoning_chars` was always 0. Fixed (both spellings accepted); **`ttft_s`/`reasoning_chars` in proxy logs before 2026-07-30 are unusable for laguna.** Corollary: a `reasoning_chars=0` row is now meaningful — laguna genuinely skips thinking on some requests.
6. **Bench-noise honesty:** single-fire buckets carry real variance — a "+43 % at 60 k" from the rope-fixed drafter evaporated on the next run, and the KV pool varies run-to-run on identical config (821,116 vs 858,126 tokens on consecutive v0.25.1 starts), so pool size does not discriminate between versions. Decision-grade comparisons need 3-run medians (same rule as the GB10 allocator confounder, §30).

### 36. ds4 is the coding default; laguna pulled *(2026-08-01)*

`deepseek-v4-flash-0731` replaced `laguna-s-2.1` as the **resident coding default** — user's call after judging laguna unacceptable in real use. The failure that ended it is not one the throughput benches could see: laguna **plans forever without committing to an action** (~2000 plans generated for a single task, never acting; the HF Laguna-S-2.1 "runaway reasoning" issue #16 that §35 already documents as upstream-open and drafter-independent). `bin/probe-commits-to-action.py` was written that day to make that behavior measurable rather than anecdotal.

Not a co-residency split: laguna needs ~113 GB of the 121 GB unified pool and ds4 needs ~86 GB, so this is a **straight replacement**. Same day, the lane was promoted to the **official DeepSeek-V4-Flash-0731 weights** (antirez re-quant, imatrix recalibrated on 0731 — 202,100 chunks vs 90,042 for the preview) after verifying they load unmodified on the pinned binary: smoke 9/10, prefill/decode at parity with the preview (404 t/s @2 k, 19–20 t/s), disk-KV prefix hit worth **6.3× TTFT**. The `sm_121` rebuild of the binary was measured **worse** (−3…−6 % prefill, −2…−4 % decode) — it ships sm_75 cubins and runs via driver JIT, which is correct here.

**What the swap costs, measured on this box** (`logs/ds4-0731-window-*`):

| | laguna + DFlash | ds4 0731 |
|---|---:|---:|
| decode @100 k | 33 tok/s | 14.3 tok/s |
| decode @2 k | ~45 tok/s | 19.4 tok/s |
| TTFT @100 k | ~64 s | 370 s cold / ~60 s warm prefix |
| concurrency | 3.28× KV pool at 256 k | **none** — prefills serialize |
| max ctx | 262 144 | 131 072 |

The concurrency line is the operational risk for resale traffic: ds4 is single-stream **by design** (upstream calls it planner-only for that reason) — measured c=4 aggregate is **0.92× of c=1**, fully serialized, `tok_per_step` 1.000. In exchange: no runaway-planning failure mode, a 304 B model instead of 118 B-A8.5 B, and a disk-KV prefix cache that survives restarts (load-bearing at ~100 k:4 k traffic).

Mechanics of the cutover: all six ds4-backed route names (`laguna-s-2.1`, `qwen3.6-27b`, `qwen3.6-35b-a3b`, `nemotron-3-puzzle-75b`, `deepseek-v4-flash-ds4`, and the real key `deepseek-v4-flash-0731`) resolve to the one engine **at the LiteLLM gateway** — clients kept working unchanged. The keepalive cron switched `laguna-keepalive.sh` → `bin/ds4-keepalive.sh`. **Rollback is no longer one command:** `config/llama-swap.yaml.bak.20260801-prelaguna-swap` restores the config, but the laguna target weights were **deleted 2026-08-02** to reclaim disk (the 4.2 GB DFlash drafter is still cached), so a rollback re-downloads 67 GiB — budget ~1 h, not 10 min. `nemotron-3-puzzle-75b` (the §33 rollback) and the Hy3 295 B eval weights were deleted in the same pass.

### 37. ds4 quant eval; the upstream host-registration OOM hard-hang *(2026-08-02)*

Two higher-bpw candidates were evaluated against the IQ2_XXS production quant, and the box was hard-hung once in the process.

1. **The ds4 engine's quant layout is hard-pinned, not a preference.** `ds4.c` accepts only IQ2_XXS / Q2_K / Q4_K experts + Q8_0 shared-expert + F16 router. **No unsloth `UD-*` quant can ever load on it** (bf16 router, q6_k shexp, iq3_xxs/iq2_xs experts are all rejected). So unsloth's `UD-IQ3_XXS` (104 GB, 3.06 bpw) had to be evaluated on **llama.cpp** instead — and mainline llama.cpp has had `deepseek4` support since 2026-06-29, which makes the old `~/llama.cpp-gx10-dsv4` fork obsolete for this: the fork runs DSV4 ops **CPU-only**, ~10× slower. On mainline it measured **473 t/s prefill / 16.2 t/s decode @2 k** — beats the then-prod prefill at +1 bpw, costs ~20 % decode. Not promoted (it cannot serve the ds4 lane at all), but it is the reference point that made §39's mainline cutover look worth trying.
2. **Upstream antirez/ds4 `54b36ed` OOM'd the machine.** Its no-copy host registration of the 80 GiB model mmap **fails on this box**, and the fallback is a *device-side copy* — doubling real footprint against a *planned* 93.92 GiB. Result: host OOM, hard hang, **power cycle** (GB10 has no remote recovery from host OOM). `bin/launch-ds4-upstream-q4k.sh` is now guarded behind `ALLOW_OOM_RISK=1`.

The eval machinery from that day is kept because it is what made the window safe: **`bin/park-prod-ds4.sh` / `restore-prod-ds4.sh`** (points the live model's `cmd:` at `bin/eval-window-blocked.sh` *and* clears the `on_startup` preload, so neither a stray request nor a llama-swap reload can spawn a second ~90 GB engine — this is what let the box come back cleanly from the power cycle), **`bin/mem-watchdog.sh`** (SIGKILLs a runaway engine before it hangs the host — contrast the *reload*-based §41 watchdog), and **`bin/pardl.py`** (HF CDN gives ~9–16 MB/s single-stream from here; 4 parallel ranges gets ~33 MB/s, and egress is a hard ~32 MB/s ceiling, so quant downloads are serialized, never concurrent).

### 38. Gateway pricing for the ds4 lane; the silent zero-spend window *(2026-08-03)*

ds4 is priced at **$0.10 / 1M input, $0.40 / 1M output** — cost recovery (roughly power + Spark amortization; DeepSeek's own flash tier is ~3× that). Everything else stays at $1.00 / $5.00.

Two things worth keeping:

- **Price every alias, not just the real key.** All six ds4-backed `model_name` entries carry the same rate. Pricing only `deepseek-v4-flash-ds4` would have reported **$0**, because live clients call `deepseek-v4-flash-0731` and `laguna-s-2.1`. Corollary hazard: a price left behind on a route that now points at a *different* engine bills the wrong rate silently — re-check `deployed.yaml` pricing whenever the resident changes.
- **Pricing had silently vanished for two months.** Every request from 2026-06-01 to 08-03 recorded `spend=0` with non-zero `prompt_tokens`; the backfill was **+$60.04**. The check that catches it: after *any* gateway change, query for rows with `spend = 0 AND prompt_tokens > 0`. Note that ds4's disk-KV cache hits are **not** discounted — the engine does not report cached-token counts in `usage`, so LiteLLM cannot see them. Traffic is ~45:1 prefill-heavy, so input cost dominates revenue.

### 39. ngc-shj fork to antirez mainline: 2.33x prefill for 18% less decode *(2026-08-06)*

The lane moved off the ngc-shj Q4 fork (prod since §30/2026-05-18) onto **antirez/ds4 mainline `b030961`**. Mainline's aligned-artifact repack + vendored mmq prefill tier removes the i-quant dequant wall the lane had sat behind since May. Same GGUF, same prompts, `--temp 0`:

| | ngc-shj fork | mainline b030961 |
|---|---:|---:|
| prefill @34.6 k coding ctx | 368 t/s | **857 t/s** (2.33×) |
| prefill shape | decayed 401 → 349 | flat ~850 from 2 k to 32 k |
| decode @34.6 k | 17.38 t/s | 14.18 t/s (−18 %) |

Net win because the lane is prefill-bound: mainline wins whenever `prompt/output > ~8.4`, and our traffic is ~25:1. **Also gained:** §37's host-registration hard-hang is structurally gone — mainline leaves the model mmap unpinned, so there is no 80 GiB registration to be declined and no device-copy fallback.

Consequences that bit or nearly bit:

- **Two fork-only env vars are gone.** `DS4_CUDA_Q4_DECODE` (the whole source of the fork's decode advantage) and `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` do not exist upstream. The reserve override is *unnecessary* now: mainline special-cases hosts with ≥112 GiB total and defaults to a 512 MiB reserve, so the 5 %-of-total default that forced §30's override is gone.
- **`--warm-weights` was removed** — the lane refuses to start if you re-add it. Obsolete anyway: mainline eagerly builds ~78.7 GiB of aligned CUDA artifacts at load (~22 s).
- **Thinking moved to `reasoning_content`** (the fork emitted it inline in `content`). log-proxy already splits that field, but on a tight `max_tokens` the whole budget can go to thinking and `content` comes back **empty** — measured: `max_tokens=300` → 0 chars, `finish=length`; `max_tokens=2000` → real code. Same class of failure as the Qwen3.6 thinking issue, same fix (bigger client budget).
- **The only local delta is the Prometheus `/metrics` endpoint** (187 additive lines, `~/ds4-metrics-endpoint-b030961.patch`), now under a neutral **`ds4:`** namespace instead of `vllm:` — upstream would not take a vllm-branded one. `bin/stack/engine.py` aliases `ds4:` → `vllm:` so every existing consumer keeps working. Re-apply the patch after any upstream pull.
- **Disk-KV gets its own directory per engine.** On-disk KV format compatibility across engines is unverified, so each binary points at its own `--kv-disk-dir`; the previous engine's warm cache stays intact for rollback. Cost is a cold prefix cache after each cutover.

DSpark speculative decode was rejected here: **23–24 % slower** at both context lengths on coding prompts, greedy-only, +6 GB. That rejection was **overturned four days later** — see §40.

### 40. Entrpi/ds4 fork is production; DSpark measured twice *(2026-08-10)*

Production is the **Entrpi/ds4 fork v0.5.6.2** (`~/entrpi-src`, launcher `bin/launch-ds4-entrpi.sh`). Measured against mainline `84cc882`, same GGUF (hard-linked, one inode — the fork's copy costs no disk), server harness, decode timed first→last token so prefill is excluded:

| | mainline | Entrpi v0.5.6.2 |
|---|---:|---:|
| decode @34.6 k coding ctx | 14.10 t/s | **19.55 t/s** (1.39×) |
| decode short | 17.95 t/s | 20.14 t/s (1.12×) |
| TTFT @34.6 k | 40.2 s | 32.7 s (1.23×) |

Quality gate: `bin/smoke-ds4-0731.py` scored **10/10** (prod scored 9/10). The weights are the only thing that did not change — kernels, sampling and tool-call handling are all fork code. Boot is ~90 s vs mainline's ~27 s (the fork builds its aligned artifacts in-process); the installer also produced a resident `ds4_weight_server` that could cut this to seconds over IPC — not wired up, worth doing if this lane persists. **Rollback is one line** in `llama-swap.yaml` back to `bin/launch-ds4-server.sh`; that script and the mainline binary are untouched.

**`--no-spec` is deliberate.** The fork's own DSpark drafter is a **net loss at long context here**: 17.90 vs 19.55 plain (−8.4 %, reproduced), while winning +9.9 % on short prompts. Our traffic is ~100 k:4 k, so long-context behavior decides it. The counters were *healthy while losing* (accept_ratio 0.64–0.68, `tok_per_step` 2.5, 0 quench events) — a break-even/scheduling issue, not a broken drafter.

**§39's DSpark rejection was wrong, and was reversed the same day.** Mainline `0e89a0e` fixed an accept-replay bug; re-tested on `~/ds4-mainline-0810`, the −23 % became **+11.5 % (server) / +14.8 % (CLI)** on long coding context. The lesson is about *when* a rejection expires: a measured "slower" verdict on a moving upstream is only valid against the commit it was measured on. It changes nothing operationally — fork-plain still beats mainline-with-DSpark — but the entry it invalidated was four days old.

Still unproven under real traffic on this engine: disk-KV behavior at 131 k and opencode tool-call parsing.

### 41. The 256 K context outage; a memory floor that refuses instead of shrinking *(2026-08-10)*

Same day as §40, `--ctx` was raised 131072 → **262144** and it took the lane **hard-down for ~35 minutes**: 33 refusals, **zero completions**, ~50 ms HTTP 503 each, straight to the customer. Reverted to 131072; **do not re-apply**.

The reasoning for the bump was that KV slabs are **demand-mapped** (boot log: *"comp/index slabs demand-mapped, virtual 2238 MiB/bank, floor 97.8 MiB/bank"*), so a deep budget is virtual and backed only as contexts actually grow — "nearly free." **The virtual budget is free; the backing is not, and nothing caps its growth.** Real traffic grew the slabs until the engine's 116.9 GiB allocation left `MemAvailable` at **1.7 GiB** against `--mem-floor-gb 8`. From there every deep admission was unfundable (`ds4.c:35841`), and each rejected job bounced to the serial path where the deep-serial guard (`ds4_server.c:15708`, `DS4_SERVER_SERIAL_MAX_TOKENS`, default 65536) refused it 503 — because our coding prompts are 65–100 k tokens.

Three durable lessons:

1. **`--mem-floor-gb` is enforced at admission, not by shrinking.** It correctly refused to fund work — the 503s *are* the floor working. It protects the box from the §37 host-OOM hard-hang; it does **not** protect the lane from becoming unservable. Both are true. And it does not self-recover: the guard's comment assumes a transient memory dip, but once the slabs have grown `MemAvailable` never climbs back.
2. **Every health signal was green.** Process alive, `/v1/models` answering, no OOM, no kernel event, `ds4_cont_batch_failures_total` 0, `ds4_requests_inflight` 0 — while `ds4_requests_total{outcome="completed"}` sat frozen at 58 and `{outcome="refused_deep_serial"}` climbed 12 → 33. That signature — **refusals rising while completions stay flat** — is now the trigger for **`bin/ds4-degraded-watchdog.sh`** (cron `*/2`, two curls when healthy). Its action is a **reload, not a kill**: the floor breach is latched, not fatal, and unloading releases the grown slabs (measured 20 s to recover, artifacts already built, disk-KV prefix cache survives). Low `MemAvailable` alone is deliberately *not* a trigger — it dips legitimately under healthy load, and reloading then would destroy live work. `bin/ds4-keepalive.sh` cannot cover this: it only acts when the engine is **gone**.
3. **Do not retry deep ctx without a cap on backed slab growth** (or a floor breach that *evicts* rather than refuses). Raising ctx alone just moves the cliff — at 131072 the live margin over the 8 GiB floor is only ~1.1–1.8 GiB, thinner than the slab growth observed that day (871 pages ≈ 5.2 GiB), so recurrence is possible; **140 k is not a safe intermediate step.** Concurrency is not a reason to want it either (c=4 measured 0.92× of c=1, §36), and upstream's own data near 248 k is ~146 ms/token (~7 t/s). Upstream ships `-c 262144` as the `ds4-serve` default and documents 524288 as deepest tested — neither on a 119 GiB box holding a 116 GiB resident.

## §42 Qwen3.8-27B NVFP4 is the coding default; ds4 dormant (2026-08-16)

Cut over 2026-08-16 ~14:15 during a zero-traffic window. Resident:
`unsloth/Qwen3.8-27B-NVFP4` (22.4 GB compressed-tensors mixed quant) on vLLM
pinned image `vllm-qwen38:prod-20260816` (= nightly-aarch64 @ sha256:677afd5b…),
port 9030, launcher `bin/launch-vllm-qwen38-prod.sh` — every flag justified in
its header. All seven gateway route names resolve to it; ds4 stays DORMANT
(defined in llama-swap but cmd-blocked; weights + disk-KV intact; rollback =
`bin/rollback-to-ds4.sh`, which stops qwen FIRST — order is load-bearing).

Why (measured, this box, fixed streaming harness in `bin/bench-deep.py`):

| | qwen3.8 NVFP4 (MTP n=3) | ds4 Entrpi (incumbent) |
|---|---|---|
| decode @~34k | ~23 t/s (probe interp.) | 18.0 |
| decode @141k sustained | 10.5–11.1 (median c=1) | n/a — ctx caps at 131k |
| ctx | 262144 (verified @259,778) | 131072 (§41 forbids raising) |
| cold TTFT @~100–141k | ~165 s | ~370 s |
| c=4 @120k | works; 14.1 t/s agg | prefill-serialised |
| answer discipline | 84–124 tok, stops | ate a full 1024 budget thinking |
| smoke | 13/14, 0 hard fails | 10/10 (its own gate) |

Key findings that shaped the config, in one place:
- **MTP n=3 is the single biggest lever** (+25–50% vs n=1; 83–90% draft
  acceptance). No DFlash/Eagle drafter exists for 3.8; the native head ships in
  both the BF16 and NVFP4 weights (15 `mtp.*` tensors, separate
  `model_mtp.safetensors` in the unsloth repo).
- **NVFP4 wins by bandwidth, not FP4 compute.** sm_121 runs Marlin
  dequant→BF16; the chip has FP4 silicon but no vLLM path uses it yet. FP8 is
  −30% (community-measured on this exact pair, vLLM 0.27.1); BF16 is ~2.6× the
  bytes; AutoRound MixedInt4 ties decode but −30% prefill (kept on disk as the
  quality fallback, smoke 13/14).
- **`UTIL=0.70`, not 0.85.** 0.80+ makes NVRM log `NV_ERR_NO_MEMORY` during
  warmup — the precursor signature of the 2026-08-15 host-OOM hard-hang (power
  cycle #3). The keepalive greps the kernel journal for it every 5 min.
  `free`'s "available" is NOT the safety metric; NVRM errors are.
- Aggregate decode is flat past c=2 (13.8–14.1 t/s @120k) — bandwidth wall,
  same as §36-era findings. SEQS=4 kept for KV/queueing, not throughput.
- No `--reasoning-parser`: thinking arrives inline in `content` (clean on
  0.27.2, no tag leak). Parser buffering vs clean-content A/B still open.

Bench integrity note: every number above postdates fixing a harness bug where
TTFT was measured on `content` deltas only — on thinking models that collapsed
the decode window and inflated tok/s (ds4's own 19.55 was mildly affected;
re-measured 18.0). Contended-engine runs (two matrix instances) were discarded;
`bench-qwen38-matrix.sh` now takes an flock and refuses a busy engine.

### §42 addendum — recipe-alignment round (same day)

After the official vLLM recipe went live, three deltas were tested and ALL
adopted, plus vision:
- **Vision enabled** (`--limit-mm-per-prompt '{"image":4,"video":1}'`, dropped
  `--language-model-only`): image canary correct, NO 3.6-style thinking spiral
  (73 tok on an image answer), text decode unchanged. First vision in the stack
  since 2026-08-01. Encoder costs ~35k tokens of KV pool.
- **`--kv-cache-dtype fp8`** (recipe): required dropping the flash_attn pin —
  vLLM auto-picks FLASHINFER, which also autotunes fp4 GEMMs. KV pool 812k →
  **1,521,344 tokens** (5.8× full-256k) and long-ctx decode jumped 13.2 →
  **22.9 t/s @141k** (+73%; halved KV reads). Short ctx +20-25% too.
- **`--reasoning-parser qwen3`** (recipe): the 3.6-era buffering is gone —
  reasoning streams incrementally (first delta 0.16 s, 33 chunks before
  content). NOTE the API contract: thinking arrives in **`reasoning`**, not
  `reasoning_content` (ds4 used the latter; log-proxy reads both). A first
  diagnostic misread this as "parser swallows reasoning" by checking only
  reasoning_content — the exact feedback_reasoning_field_name trap.
- **Dropped `--generation-config vllm`** (bug): it silently replaced the
  model's recommended sampling (temp 1.0/top_p 0.95/top_k 20) with vLLM
  generics. Copy-through from the 3.6 lanes.

Final prod figures (single-run probe; c-sweep medians pending re-run on this
config): 26.8 @2k / 22.7 @60k / 22.9 @141k decode, prefill ~2,100→790 by ctx,
cold TTFT 178 s @141k. Recipe flags NOT adopted: none — but note the recipe's
`--reasoning-parser qwen3` field-name behavior above before writing clients.

## §43 One model, one route: aliases retired and keys re-scoped (2026-08-17)

The docs, the gateway catalog, and the loaded engine had drifted into three
different answers, and an agent configuring a client could pick any of them.

**What was wrong.** `deployed.yaml` still described ds4 as the resident in its
header (`context_window: 131072`, "vision DARK", thinking in `content`) while
its own `model_list` 200 lines below routed everything to `qwen3.8-27b`.
`docs/gateway-setup.md` — the page AGENTS.md marks safe to hand to an outside
agent — was wrong on model id, context window, vision, and concurrency.
`AGENTS.md` carried a hard rule forbidding the 262144 context that production
was running. The gateway advertised **18 routes of which 7 worked**; the other
11 returned `could not find suitable inference handler`. And no API key was
scoped to `qwen3.8-27b` at all, so the canonical id the docs recommended worked
for nobody.

**What changed.**
- `deployed.yaml` restructured: a `live:` block that is the complete client
  contract, and a `not_serving:` block that answers nothing. No route's
  behaviour can be inferred from history any more.
- `model_list` pruned 18 → **1**. Aliases `deepseek-v4-flash-0731`,
  `deepseek-v4-flash-ds4`, `laguna-s-2.1`, `nemotron-3-puzzle-75b`,
  `qwen3.6-27b`, `qwen3.6-35b-a3b` are gone.
- All **12 API keys** re-scoped to `models: ["qwen3.8-27b"]`. Four had been on
  `all-team-models` with no team, which resolves *permissively* — they could
  call anything, including the dead routes.

**Why not keep the aliases.** They were the reason the catalog lied. `/v1/models`
is filtered per key, so scoping every key to one model makes the catalog, the
allowlist, and the engine agree by construction — there is no second place for
the truth to drift to. The cost is real and was accepted: four clients that
still sent legacy names broke at cutover and needed a one-line config change.

**Verified after the change:** `/v1/models` returns exactly `qwen3.8-27b`; a
call on it succeeds; `deepseek-v4-flash-0731` returns 400; billing reconciles
exactly (59 pt + 27 ct = $0.0000167 at $0.10/$0.40 per 1M).

**Contract gotcha re-confirmed here:** thinking arrives in `reasoning`. The live
response has no `reasoning_content` key at all — a client reading only that
field gets silence, not an error.

**Rollback:** key state is backed up as JSON (token hash + models per alias) at
`~/private-backups/litellm-keys-20260817-pre-single-route.json` — deliberately
OUTSIDE this repo, which is being prepped to go public;
`deployed.yaml` and `litellm-config.yaml` have `.bak-20260817-*-pre-single-route`
copies on .7. Re-adding a route now takes three edits — `model_list`,
`llama-swap.yaml`, and the key allowlist. Miss the third and the failure looks
like an auth error, not a routing one.

**Noted, not fixed:** ds4-era spend rows were under-priced (08-13: $0.0277 on
4.19 M prompt tokens, ~15× low against the documented $0.10/1M). qwen3.8 rows
are correct. Also, the repo disagrees with itself on traffic shape — §38/`deployed.yaml`
say ~45:1 prompt-to-output, §39/README say ~25:1. Neither is measurable from the
current logs: proxy meta carries bytes, not tokens.

### §42 addendum 2 — the prod decode figures re-measured as medians (2026-08-17)

§42's addendum recorded `26.8 @2k / 22.7 @60k / 22.9 @141k` and flagged itself
as a "single-run probe; c-sweep medians pending re-run on this config". This is
that re-run, and **26.8 did not reproduce**.

Harness `bin/bench-fresh-gen.py` (new): single stream, 400 output tokens, short
prompt, 3 runs, median, 150 s idle between runs, unique prefix per run so
prefill is never a prefix-cache replay. Decode window is
`completion_tokens / (last_token − first_token)` where a token-bearing delta
counts if it carries **either** `reasoning` or `content` — the §42 harness bug
was counting `content` only, which collapses the window on a thinking model and
inflates tok/s. That is the most likely source of the 26.8.

```
thinking      18.01  (17.89, 18.01, 18.60)   <- tight, real
non-thinking  16.62  (14.34, 16.62, 16.73)   <- run 1 cold-ish outlier
```

Treat **18.0 tok/s** as the honest single-stream fresh-generation number for
prod, not 26.8. The 60 k/141 k points have not been re-measured as medians yet
and should be assumed similarly optimistic.

Two external recipes measured on the same chip, for calibration:

| recipe | fresh gen 400 tok | notes |
|---|---:|---|
| ours: unsloth NVFP4 + MTP n=3 | **18.01** | this measurement |
| MiaAI-Lab SGLang, MTP | 16.9 / 21.0 | thinking / non-thinking, their bench |
| 0xBakeer vLLM, NVFP4 + DSpark k=7 | 29.23 | **same weights, same engine family** |
| 0xBakeer vLLM, NVFP4 + DSpark k=14 | 29.55 | edit-heavy 72.6–75.0 |

So SGLang is roughly a wash (we win on thinking, lose on non-thinking), while
DSpark on our own engine claims ~1.6× on fresh generation and far more on
edit-heavy work — the shape our ~100k:4k coding traffic actually has. Eval
launcher staged at `bin/eval-qwen38-dspark.sh`; runbook `~/sglang-eval/RUNBOOK.md`.

**Open, and worth fixing regardless of the benchmark:** that recipe calls
`VLLM_MARLIN_USE_ATOMIC_ADD=1` non-optional on SM121 — a Marlin race that
produces *incorrect output rather than an error*. Our 4-bit weights dequantize
through Marlin, and prod sets the flag nowhere. A race is intermittent, so the
launcher header's "output verified clean" does not clear it.

## §44 Speculative-decoding bake-off: DSpark k=4 is the best draft depth, SGLang's DGX Spark config will not run here (2026-08-17)

Two community recipes for this exact model+chip were evaluated against prod
across three eval windows. Net outcome: **one correctness fix adopted, no
performance change adopted, both external recipes rejected.**

### The harness, and the confound that nearly invalidated everything

`bin/bench-fresh-gen.py` (new): single stream, 400 output tokens, 3 runs,
median, 150 s idle, unique prefix per run. `bin/bench-concurrent-fresh.py` (new)
reports aggregate AND per-request at c=N, because a draft-depth knob trades one
for the other.

**Prompt content is part of the measurement, not decoration.** The same engine
and config measured 14.05 tok/s on a prose prompt and 28.23 on a code prompt —
a 2x swing — because speculative acceptance tracks output predictability. The
first half of this investigation used prose and concluded DSpark was a large
loss; on code it is a small win. Both external repos benchmark *code*
generation. Always state the prompt class next to a spec-decode number.

### Draft depth: shallow wins, and the acceptance *rate* is a trap

Code prompt, c=1, median, all with the two env vars set:

| config | thinking | non-thinking | accept rate | accepted/draft |
|---|---:|---:|---:|---:|
| prod: MTP n=3 | 21.58 | 26.69 | — | — |
| MTP n=3 + env vars | 20.23 | 26.73 | 62.8 % | 1.89 |
| DSpark k=2 | 19.85 | 22.20 | 68.0 % | 1.36 |
| **DSpark k=4** | 25.13 | **28.72** | 49.9 % | **2.00** |
| DSpark k=7 | — | — | 13.2 %(prose) | 0.92 (prose) |
| DSpark k=14 | 20.79 | 28.23 | 15.3 % | 2.14 |

c=4, code, non-thinking: k=4 **97.92** agg / 25.52 per-req; prod 93.67 / 25.13;
MTP+env 91.50 / 24.71; k=2 78.60 / 20.19.

**Read accepted-per-draft, not accept rate.** On prose, k=7 and k=14 both
yielded ~0.92 tokens per draft — doubling depth bought nothing and doubled draft
compute, which is why k=14 was the slowest config tested. k=2 is too shallow
(1.36). The optimum is k=4, consistent with MTP n=3 working well. Upstream's
default block size of 7 is NOT optimal here.

**Verdict: DSpark k=4 is +7.5 % at c=1 and +4.5 % agg at c=4 over prod — NOT
adopted.** That margin is close to the run-to-run spread this box shows, it adds
a 2.6 GB drafter to keep revision-pinned, and the drafter logs
`does not support external multimodal embeddings`, so image requests would draft
text-only — a real cost now that vision is on. Revisit only with a confirmation
run. Drafter kept cached at `Doopeworld/Qwen3.8-27B-DSpark-vLLM`.

### RETRACTED: the "+16 % from env vars" claim

An earlier prose comparison showed prod 18.01 vs MTP+env-vars 21.06 and was
reported as a 16 % win. It is not real. On the code prompt the same pair
measures 26.69 vs 26.73, and at c=4 the env-var build is *slower* (91.50 vs
93.67). The 21.06 came from a freshly started engine whose three runs trended
upward (17.75, 21.06, 23.78) — warmup, not the flag. Lesson repeated from §42:
never compare across engine instances of different age on this box.

### VLLM_MARLIN_USE_ATOMIC_ADD=1 — ADOPTED, as correctness only

Added to `bin/launch-vllm-qwen38-prod.sh`, live 2026-08-17. There is a race in
the Marlin kernel on SM121 that yields **incorrect output rather than an error**,
and our 4-bit weights dequantize through Marlin, so it sat in prod's decode path
unset. Measured performance impact: none (see retraction above). Adopted purely
because a race is intermittent — the launcher's prior "output verified clean"
note was never evidence of absence. `VLLM_USE_FLASHINFER_MOE_FP4=0` was NOT
adopted: vLLM logs it as an unknown env var on this build, and the model is dense.

### SGLang's official cookbook config does not run on this box

`docs.sglang.io` cookbook, hw=dgx-spark quant=nvfp4 tier=low-latency
ssmDtype=float32, both spec variants. It hardcodes `--mem-fraction-static 0.95`.
Both loads died ~5 min in, each preceded by NVRM `NV_ERR_NO_MEMORY` bursts; a
watchdog in `~/sglang-eval/launch-cookbook.sh` killed the container on the first
warning, so the box did not hang. Consistent with the fact that **no `dgx-spark`
cell in that cookbook is marked `verified:true`** while h200/rtx6000/rtx5090/
gb300 all are.

Retesting at a lower mem-fraction was declined: it would no longer be their
config, and SGLang's best published figure (21.0 tok/s) is well under our
measured 28.72, so shrinking its KV pool cannot close the gap.

One SGLang flag has no vLLM equivalent and is the only mechanism by which it
could win on this hybrid-GDN model: `--enable-linear-replayssm-spec` (replays
linear-attention state during speculation instead of allocating per-draft
slots). Confirmed absent from vLLM: `SpeculativeConfig` mentions no
replay/linear/ssm/mamba, and `MambaConfig` exposes only backend,
enable_stochastic_rounding, stochastic_rounding_philox_rounds, ssu_algorithm.
Also note `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark` deviates from the cookbook on
three axes (extra_buffer_lazy instead of extra_buffer, bf16 instead of float32
SSM, and no replayssm flag), so its 16.9/21.0 figures likely understate SGLang.

### OPEN: NVRM warnings at UTIL 0.70, which §42 says should be clean

§42 states 0.70 gives "0 hits" and only 0.80+ logs NVRM. Not true today. Prod
warmup at 0.70 logged **108+21 events at 15:24 and 66 at 15:44**, while eval
engines at the same 0.70 earlier logged only 8 and 2. The bursts appear only
*after* ~62 GB of artifacts (a 38.6 GB image + 24 GB of weights) had been pulled
into page cache — the hang-#3 shape, but from the download's residue rather than
an active download. The artifacts were deleted (109 -> 169 GB free). **Unverified
hypothesis; confirm on the next natural prod restart before trusting 0.70 as a
clean baseline again.** If bursts persist with a cold page cache, the safe-util
figure in §42 is wrong and needs re-deriving.

### Fixed in passing: park-prod-ds4.sh could not park the current resident

It matched only `launch-ds4-*.sh`, so after the 2026-08-16 cutover it silently
matched nothing — and its verify grep still PASSED, because the dormant ds4
entry already read `eval-window-blocked.sh`. It also cleared a preload hardcoded
to `deepseek-v4-flash-0731` (live: `qwen3.8-27b`) and killed only `ds4-server`
processes, never the vLLM container. Now: blocks every `launch-*` cmd, verifies
no launcher line survives, clears whatever preload names, and removes `vllm-*`
containers. Dry-tested against a config copy before use. Three windows opened
and closed on it since, each restoring config+crontab byte-identical.
