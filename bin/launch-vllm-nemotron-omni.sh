#!/bin/bash
# Production launcher — Nemotron-3-Nano-Omni-30B-A3B-Reasoning NVFP4 (added 2026-04-29).
#
# Mamba2-Transformer hybrid MoE with vision (CRADIO v4-H) + audio (Parakeet)
# encoders. Multimodal: text/image/audio/video. NVFP4 = 20.9 GB weights.
#
# Decisions:
#   - 0.65 util (~77 GB) — right-sized after 2026-04-29 sweep:
#     model+scaffolding ~25 GB, KV in-flight at c=8 ~12 GB, prefix cache room
#     ~30 GB, multimodal scratch ~5 GB. Frees ~24 GB back for OS/swap headroom.
#   - --max-num-seqs 8: throughput peak from c-sweep (c=8 → 383 tok/s agg,
#     c=9 dropped 30%; sharp Mamba/MoE contention cliff above 8).
#   - --max-model-len 131072: model supports 262144 but KV cache for multimodal
#     workloads dominates; start conservative, raise if traffic stays text-heavy.
#   - --max-num-batched-tokens 8192: required (Mamba block_size=2128 must be
#     <= max-num-batched-tokens, default 2048 trips an assertion).
#   - --kv-cache-dtype fp8: per model card recommendation.
#   - --reasoning-parser nemotron_v3 + --tool-call-parser qwen3_coder: from card.
#     Note: parser BUFFERS the full <think> block before any stream output,
#     so medium-output reasoning prompts feel "stuck" for ~18s. Drop the flag
#     if interactive UX matters more than parsed reasoning_content fields.
#   - NO --attention-backend FLASHINFER: hybrid Mamba2 path may not be covered;
#     let vLLM pick the default (likely TRITON_ATTN for the SSM blocks).
#   - --allowed-local-media-path /home/max/llm-stack/media: enables local
#     file:// URLs for multimodal, scoped to a dedicated empty media dir.
#     Do NOT point this at /home/max — an unauthed multimodal request could
#     then read file:///home/max/.cache/huggingface/token, ~/.ssh keys, etc.
#     Only this dir is bind-mounted into the container (read-only).
#   - Image tag cu130-nightly-omni: derived from cu130-nightly with av/soundfile/
#     librosa pre-installed (vLLM caches a PlaceholderModule for soundfile at
#     startup; hot-install after the fact errors with PlaceholderModule
#     assertion). Build: `docker exec <ctr> pip install av soundfile librosa
#     && docker commit <ctr> vllm/vllm-openai:cu130-nightly-omni`.
#   - No --rm: persist for `docker logs` post-mortem on crash.
set -e
docker rm -f vllm-nemotron-omni 2>/dev/null || true
exec docker run --name vllm-nemotron-omni \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 0.0.0.0:9012:9012 \
  -v /home/max/.cache/huggingface:/root/.cache/huggingface \
  -v /home/max/llm-stack/media:/home/max/llm-stack/media:ro \
  -e HF_TOKEN_FILE=/root/.cache/huggingface/token \
  -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  vllm/vllm-openai:cu130-nightly-omni \
  nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --host 0.0.0.0 --port 9012 \
  --served-model-name nemotron-3-nano-omni \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.72 \
  --video-pruning-rate 0.5 \
  --allowed-local-media-path /home/max/llm-stack/media \
  --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}' \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
