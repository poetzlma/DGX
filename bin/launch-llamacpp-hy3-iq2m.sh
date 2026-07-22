#!/bin/bash
# Hy3 (Tencent, 295B-A21B MoE) — vcruz305 IQ2_M GGUF. A/B counterpart to the
# IQ1_M lane: IQ2_M experts are block-type IQ2_S, which HAS a CUDA MMQ kernel
# (IQ1_M does NOT — that is the IQ1_M prefill wall). Same patched hy_v3 engine.
#
# EXCLUSIVE OCCUPANCY (~93 GiB). The preflight guard below refuses to start if
# any other llama-server is alive, port is bound, or memory is short — added
# after the 2026-07-14 two-server OOM crash that took the box down ~8.5h.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp-hyv3/build/bin/llama-server}"
# Auto-find the first shard (or a single-file gguf) unless HY3_MODEL is set.
if [[ -n "${HY3_MODEL:-}" ]]; then
  MODEL="$HY3_MODEL"
else
  MODEL="$(ls "$HOME"/models/hy3-iq2m/Hy3-IQ2_M/*-00001-of-*.gguf 2>/dev/null | head -1)"
  [[ -z "$MODEL" ]] && MODEL="$(ls "$HOME"/models/hy3-iq2m/Hy3-IQ2_M/*.gguf 2>/dev/null | head -1)"
fi
PORT="${HY3_PORT:-9029}"
CTX="${HY3_CTX:-65536}"
# MTP: only enable if this GGUF actually embeds the nextn block. Default OFF;
# set HY3_MTP=1 once confirmed from the header.
MTP="${HY3_MTP:-0}"
SPEC_N_MAX="${HY3_SPEC_N_MAX:-3}"
SPEC_N_MIN="${HY3_SPEC_N_MIN:-1}"

# ---- OOM PREFLIGHT GUARD (see 2026-07-14 crash; HY3_SKIP_GUARD=1 bypasses) ----
if [[ "${HY3_SKIP_GUARD:-0}" != "1" ]]; then
  self=$$
  others=$(pgrep -x llama-server 2>/dev/null | grep -v "^${self}$" || true)
  if [[ -n "$others" ]]; then
    echo "GUARD ABORT: another llama-server is alive (PIDs: $(echo $others|tr '\n' ' ')). Kill+wait for teardown first." >&2
    exit 1
  fi
  if ss -lntH "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; then
    echo "GUARD ABORT: port $PORT already in use." >&2; exit 1
  fi
  model_bytes=$(du -bc "$(dirname "$MODEL")"/*.gguf 2>/dev/null | tail -1 | cut -f1)
  avail_kb=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  need_kb=$(( ${model_bytes:-0}/1024 + 12*1024*1024 ))
  if (( avail_kb < need_kb )); then
    echo "GUARD ABORT: only $((avail_kb/1024/1024)) GiB avail, need ~$((need_kb/1024/1024)) GiB." >&2; exit 1
  fi
  echo "GUARD OK: model=$MODEL avail=$((avail_kb/1024/1024)) GiB." >&2
fi

[[ -z "$MODEL" || ! -f "$MODEL" ]] && { echo "no IQ2_M gguf found under ~/models/hy3-iq2m/Hy3-IQ2_M/" >&2; exit 1; }

ARGS=(
  --host 127.0.0.1 --port "$PORT"
  --alias hy3-295b-iq2m
  -m "$MODEL"
  --ctx-size "$CTX"
  --n-gpu-layers 99
  -ctk q8_0 -ctv q8_0
  -fa on
  -b 4096 -ub 2048
  --no-mmap
  --jinja
  --chat-template-file "$HOME/models/hy3-gguf/hyv3_opensource_chat_template.jinja"
)
if [[ "$MTP" == "1" ]]; then
  ARGS+=( --spec-type draft-mtp --spec-draft-n-max "$SPEC_N_MAX" --spec-draft-n-min "$SPEC_N_MIN" -ctkd q8_0 -ctvd q8_0 )
fi

exec "$LLAMA_SERVER" "${ARGS[@]}"
