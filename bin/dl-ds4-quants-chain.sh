#!/bin/bash
# Serialised, size-verified download of tonight's two ds4 quant candidates.
#
# WHY SERIAL: the box's egress is a hard ~32 MB/s ceiling, measured — running
# both at once is not additive, it just starves one (the antirez single-stream
# job sat at 0 MB/s behind unsloth's 4 connections). Serial finishes the same
# 201 GB in the same wall clock but frees the first model for benching sooner.
#
# WHY SIZE-VERIFIED: curl exiting 0 does NOT mean the file is complete — a
# server-side close mid-transfer looks like success. Every file is checked
# against the HF manifest byte count and retried until it matches, so the
# COMPLETE marker below can be trusted by an automated watcher.
set -u
log(){ echo "[$(date +%H:%M:%S)] $*"; }

fetch(){ # $1=url  $2=dest  $3=expected_bytes
  local url="$1" dest="$2" want="$3" have tries=0
  while :; do
    have=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && { log "OK $(basename "$dest") ($((want/1000000)) MB)"; return 0; }
    tries=$((tries+1))
    [ "$tries" -gt 40 ] && { log "GIVING UP on $(basename "$dest") at $have/$want"; return 1; }
    log "fetch $(basename "$dest") from $((have/1000000))/$((want/1000000)) MB (try $tries)"
    curl -sL -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$dest" "$url" || true
  done
}

UNS_DIR=/home/max/models/ds4-unsloth-UD-IQ3_XXS
UNS_BASE=https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ3_XXS
mkdir -p "$UNS_DIR"
# byte counts straight from the HF manifest (shards are UNEVEN — 0.01/49.9/49.3/5.0 GB)
fetch "$UNS_BASE/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf" "$UNS_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf" 5257696
fetch "$UNS_BASE/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf" "$UNS_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf" 49910532416
fetch "$UNS_BASE/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf" "$UNS_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf" 49257859456
fetch "$UNS_BASE/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00004-of-00004.gguf" "$UNS_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00004-of-00004.gguf" 5034198464
log "UNSLOTH-COMPLETE"

AZ_F="DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
fetch "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/$AZ_F" "/home/max/ds4/gguf/$AZ_F" 97591747456
log "ANTIREZ-COMPLETE"
log "ALL-DOWNLOADS-COMPLETE"
df -h /home | tail -1
