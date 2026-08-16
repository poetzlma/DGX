#!/bin/bash
# Resumable HF repo download that does NOT go through hf-xet.
#
# USAGE: dl-hf-repo.sh REPO DEST [LIMIT_RATE]
#   e.g. dl-hf-repo.sh Qwen/Qwen3.8-27B /home/max/models/qwen3.8-27b-bf16 8M
#
# WHY THIS EXISTS — measured on this box 2026-08-14, hf-xet 1.4.3 /
# huggingface_hub 1.10.1:
#   1) RESUME IS BROKEN. On restart with a partial file, the Xet client asks the
#      CAS server to reconstruct the remaining range and gets back
#      "416 Range Not Satisfiable" (8 such fatal errors logged on one restart).
#      It then refetches from byte 0, TRUNCATING what you had. We lost a 14 GB
#      partial and a 20,743 MB partial this way. The partials survive a kill
#      fine — it is the restart that destroys them.
#   2) IT HANGS. One transfer parked in futex_do_wait with 0 MB/s and no error;
#      its log just stops mid-stream. Beforehand the adaptive concurrency
#      controller was logging nonsense: "Decreased concurrency from 1 to 1;
#      reason: success ratio below threshold (connection struggling)
#      (success_ratio = 1.000, threshold = 0.500)" — 1.000 is not below 0.500.
#
# curl -C - does real HTTP range resume, so a stall costs seconds, not the file.
# LIMIT_RATE (curl syntax, e.g. 8M) keeps a background download from perturbing
# a concurrent benchmark — this box's decode numbers are allocator-sensitive and
# we bench with 3-run medians, so do not let a download fight for I/O.
#
# Idempotent: completed files are skipped by exact size, partials resume.
# Re-run until it prints ALL FILES COMPLETE.
set -u
REPO="${1:?usage: dl-hf-repo.sh REPO DEST [LIMIT_RATE]}"
DEST="${2:?usage: dl-hf-repo.sh REPO DEST [LIMIT_RATE]}"
LIMIT="${3:-}"
TOK=$(cat /home/max/.cache/huggingface/token)
mkdir -p "$DEST"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

curl -s -H "Authorization: Bearer $TOK" \
  "https://huggingface.co/api/models/$REPO?blobs=true" \
 | python3 -c "
import json,sys
d=json.load(sys.stdin)
if 'siblings' not in d:
    sys.exit('repo lookup failed: '+str(d)[:200])
for s in d['siblings']:
    print(f\"{s['rfilename']}\t{s.get('size') or 0}\")
" > "$DEST/.manifest" || { log "could not fetch manifest for $REPO"; exit 1; }

rate_args=()
[ -n "$LIMIT" ] && rate_args+=(--limit-rate "$LIMIT")

fail=0
while IFS=$'\t' read -r f want; do
  out="$DEST/$f"; mkdir -p "$(dirname "$out")"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$want" -gt 0 ] && [ "$have" -eq "$want" ]; then
    log "OK    $f ($((want/1048576)) MB)"; continue
  fi
  log "FETCH $f  have=$((have/1048576))MB want=$((want/1048576))MB"
  # NEVER combine `-C -` with curl's own --retry. `-C -` computes the resume
  # offset ONCE at invocation; when curl retries internally mid-transfer it
  # re-requests from that stale offset and APPENDS the bytes it already wrote,
  # producing a file LARGER than the source. Measured 2026-08-14: one shard came
  # out 3,980,927,600 bytes against an expected 3,966,730,552 — 14,197,048 bytes
  # of duplicated data, which would have loaded as a corrupt safetensors.
  # Instead: each attempt is a FRESH curl, so `-C -` recomputes the offset from
  # the file's current size, and we verify the exact byte count after every try.
  attempt=0
  while [ "$attempt" -lt 20 ]; do
    attempt=$((attempt+1))
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$want" -gt 0 ] && [ "$have" -gt "$want" ]; then
      log "  OVERSHOOT ($have > $want) — truncating and refetching from scratch"
      : > "$out"
    fi
    [ "$want" -gt 0 ] && [ "$have" -eq "$want" ] && break
    nice -n 10 curl -L --fail --show-error --silent -C - \
         --speed-time 60 --speed-limit 1024 \
         "${rate_args[@]}" \
         -H "Authorization: Bearer $TOK" -o "$out" \
         "https://huggingface.co/$REPO/resolve/main/$f"
    rc=$?
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    { [ "$want" -eq 0 ] || [ "$have" -eq "$want" ]; } && break
    log "  attempt $attempt: $((have/1048576))/$((want/1048576)) MB (rc=$rc), retrying"
    sleep 5
  done
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$want" -gt 0 ] && [ "$have" -ne "$want" ]; then
    log "SHORT $f  $((have/1048576))/$((want/1048576)) MB (rc=$rc) — re-run to resume"; fail=1
  else
    log "DONE  $f"
  fi
done < "$DEST/.manifest"

log "on disk: $(( $(du -sb "$DEST" | cut -f1) / 1048576 )) MB"
[ "$fail" = "0" ] && log "ALL FILES COMPLETE" || { log "INCOMPLETE — re-run"; exit 1; }
