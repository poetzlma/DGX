#!/bin/bash
# Hy3 (Tencent Hy3, 295B-A21B MoE) — AngelSlim IQ1_M GGUF + MTP self-speculative.
#
# Weights: ~/models/hy3-gguf/Hy3-IQ1_M-mtp.gguf (91.8 GB, 1-bit).
# Engine:  ~/llama.cpp-hyv3 — a PATCHED llama.cpp. The hy_v3 arch is NOT in
#          mainline; AngelSlim ship two line-anchored patches against llama.cpp
#          commit 19bba67c. Do NOT point this at ~/llama.cpp (the 35B lane's
#          mainline build) — it cannot load this GGUF.
#          Rebuild with: CUDAARCHS=121 GGML_NATIVE=1 bash \
#            ~/models/hy3-gguf/setup_hyv3_llama.sh ~/llama.cpp-hyv3
#          (CUDAARCHS=121 is required for GB10/sm_121 — their script omits it.)
#
# EXCLUSIVE OCCUPANCY: 91.8 GB of weights on 119 GB unified. Nothing else can be
# co-resident. This is an eval slot, not a prod lane.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp-hyv3/build/bin/llama-server}"
MODEL="${HY3_MODEL:-$HOME/models/hy3-gguf/Hy3-IQ1_M-mtp.gguf}"

# Context. Computed from the GGUF header, NOT guessed:
#   80 real layers (blk.80 is the MTP/nextn block, not a KV layer), GQA with
#   head_count_kv=8, key_length=value_length=128  =>  q8_0 KV is ~170 KB/token.
#     32k -> 5.7 GB | 65k -> 11.4 GB | 131k -> 22.8 GB
#   Weights are 91.8 GB of the 121.6 GB unified pool, so 65k (91.8 + 11.4 + ~3
#   compute ~= 107 GB) fits; 131k (~118 GB) does not. f16 KV would not fit either.
CTX="${HY3_CTX:-65536}"

# MTP self-speculative decoding. The drafter is EMBEDDED in this GGUF
# (blk.80.nextn.*, nextn_predict_layers=1) — there is no separate draft model,
# hence no -md. -ctkd/-ctvd size the MTP block's own small KV cache.
# n-max 3 per the model card.
SPEC_N_MAX="${HY3_SPEC_N_MAX:-3}"
SPEC_N_MIN="${HY3_SPEC_N_MIN:-1}"

# ---------------------------------------------------------------------------
# OOM PREFLIGHT GUARD — added after the 2026-07-14 crash.
#
# This is a 91.8 GB exclusive-occupancy model. On 2026-07-14 a SECOND
# llama-server was launched while a first was still tearing down its CUDA
# context, so two ~100 GB processes were co-resident on the 119 GB unified
# pool. Global OOM spiraled (killed dbus/pipewire/NVRM), the box hung and went
# unpingable for ~8.5h until a manual reboot. Never again: refuse to start
# unless the box can actually hold this model.
#
# Set HY3_SKIP_GUARD=1 to bypass (only if you KNOW the box is clear).
# ---------------------------------------------------------------------------
if [[ "${HY3_SKIP_GUARD:-0}" != "1" ]]; then
  self=$$
  # 1) Any OTHER llama-server already alive? (pgrep on the binary basename;
  #    exclude our own PID and the llama-swap supervisor, which is a different
  #    binary.) A live server means its weights are pinned — do not stack.
  others=$(pgrep -x llama-server 2>/dev/null | grep -v "^${self}$" || true)
  if [[ -n "$others" ]]; then
    echo "GUARD ABORT: another llama-server is alive (PIDs: $(echo $others | tr '\n' ' '))." >&2
    echo "  A resident server pins its weights; launching a 2nd would OOM the box." >&2
    echo "  Kill it and WAIT for the PID to vanish (CUDA teardown of a 90GB model" >&2
    echo "  takes up to ~60s) before retrying, or set HY3_SKIP_GUARD=1 if certain." >&2
    exit 1
  fi
  # 2) Port already bound?
  if ss -lntH 'sport = :9028' 2>/dev/null | grep -q ':9028'; then
    echo "GUARD ABORT: port 9028 already in use — a server is likely up." >&2
    exit 1
  fi
  # 3) Enough unified memory to actually hold weights + KV + compute?
  #    Require MemAvailable >= model_size + 12 GiB headroom.
  model_bytes=$(stat -c %s "$MODEL" 2>/dev/null || echo 0)
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  need_kb=$(( model_bytes/1024 + 12*1024*1024 ))
  if (( avail_kb < need_kb )); then
    echo "GUARD ABORT: only $((avail_kb/1024/1024)) GiB available, need ~$((need_kb/1024/1024)) GiB" >&2
    echo "  (weights $((model_bytes/1024/1024/1024)) GiB + 12 GiB KV/compute headroom)." >&2
    echo "  Something else is holding memory. Free it before launching." >&2
    exit 1
  fi
  echo "GUARD OK: no other llama-server, port free, $((avail_kb/1024/1024)) GiB available." >&2
fi

exec "$LLAMA_SERVER" \
  --host 127.0.0.1 --port 9028 \
  --alias hy3-295b-iq1m \
  -m "$MODEL" \
  --ctx-size "$CTX" \
  --n-gpu-layers 99 \
  --spec-type draft-mtp \
  --spec-draft-n-max "$SPEC_N_MAX" \
  --spec-draft-n-min "$SPEC_N_MIN" \
  -ctk q8_0 -ctv q8_0 \
  -ctkd q8_0 -ctvd q8_0 \
  -fa on \
  -b 4096 -ub 2048 \
  --no-mmap \
  --jinja \
  --chat-template-file "$HOME/models/hy3-gguf/hyv3_opensource_chat_template.jinja"
