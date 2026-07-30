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
