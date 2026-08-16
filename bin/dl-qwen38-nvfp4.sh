#!/bin/bash
# Resumable download of unsloth/Qwen3.8-27B-NVFP4 into a plain directory.
#
# WHY NOT `hf download`: on 2026-08-14 it stalled twice mid-file (process parked
# in futex_do_wait, 0 MB/s, log frozen) and — critically — it does NOT resume.
# Restarting truncated a 20,743 MB partial back to 64 MB. Every stall therefore
# costs the whole file. curl -C - does real HTTP range resume, so a stall costs
# only the seconds since the last byte.
#
# vLLM loads a local directory path directly, so we skip the HF cache layout
# entirely and pass this dir as MODEL_ID.
#
# Re-run this script as many times as needed; completed files are skipped by
# size check and partial files resume in place. Safe to run concurrently with
# nothing else — it is idempotent.
set -u
REPO="unsloth/Qwen3.8-27B-NVFP4"
DEST=/home/max/models/qwen3.8-27b-nvfp4
TOK=$(cat /home/max/.cache/huggingface/token)
mkdir -p "$DEST"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# rfilename<TAB>size, so we can tell "done" from "truncated".
curl -s -H "Authorization: Bearer $TOK" \
  "https://huggingface.co/api/models/$REPO?blobs=true" \
 | python3 -c "
import json,sys
for s in json.load(sys.stdin)['siblings']:
    print(f\"{s['rfilename']}\t{s.get('size') or 0}\")
" > "$DEST/.manifest"

fail=0
while IFS=$'\t' read -r f want; do
  out="$DEST/$f"
  mkdir -p "$(dirname "$out")"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$want" -gt 0 ] && [ "$have" -eq "$want" ]; then
    log "OK      $f ($((want/1048576)) MB)"
    continue
  fi
  log "FETCH   $f  have=$((have/1048576))MB want=$((want/1048576))MB"
  # --retry with backoff, -C - resumes at the byte we already have.
  curl -L --fail --show-error --silent \
       -C - \
       --retry 20 --retry-delay 5 --retry-all-errors \
       --speed-time 60 --speed-limit 1024 \
       -H "Authorization: Bearer $TOK" \
       -o "$out" \
       "https://huggingface.co/$REPO/resolve/main/$f"
  rc=$?
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$want" -gt 0 ] && [ "$have" -ne "$want" ]; then
    log "SHORT   $f  got $((have/1048576))MB of $((want/1048576))MB (curl rc=$rc) — re-run to resume"
    fail=1
  else
    log "DONE    $f"
  fi
done < "$DEST/.manifest"

total=$(du -sb "$DEST" | cut -f1)
log "total on disk: $((total/1048576)) MB"
[ "$fail" = "0" ] && log "ALL FILES COMPLETE" || { log "INCOMPLETE — re-run this script"; exit 1; }
