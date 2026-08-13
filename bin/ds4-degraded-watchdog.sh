#!/bin/bash
# Catch the ds4 lane when it is UP but REFUSING — the failure keepalive misses.
#
# WHY THIS EXISTS: on 2026-08-10 ~15:53 the lane went hard-down for all >64k
# traffic for at least 12 minutes and NOTHING noticed. Every health signal was
# green: the process was alive, /v1/models answered, no OOM, no kernel event,
# ds4_cont_batch_failures_total 0, ds4_requests_inflight 0. But
# ds4_requests_total{outcome="completed"} sat frozen at 58 while
# ds4_requests_total{outcome="refused_deep_serial"} climbed 12 -> 33, each a
# ~50 ms HTTP 503 straight to the customer.
#
# Root cause that day was --ctx 262144 growing demand-mapped KV slabs until
# MemAvailable fell to 1.7 GiB under the 8 GiB --mem-floor-gb, after which
# every deep admission was unfundable (ds4.c:35841) and each rejected job hit
# the deep-serial guard (ds4_server.c:15708, default 65536 tokens) instead of
# being served. ctx is back to 131072, but the margin above the floor (~1.8 GiB)
# is still THINNER than the slab growth observed that day (871 pages ~ 5.2 GiB),
# so recurrence is possible and this watchdog is the detection we lacked.
#
# WHY ds4-keepalive.sh DOES NOT COVER THIS: keepalive polls /v1/models and only
# acts when the engine is GONE. Here the engine answers /v1/models perfectly
# while refusing real work. Different failure, different probe. Both should run.
#
# THE SIGNATURE, and why it is the right trigger: refusals rising while
# completions stay flat. That is unambiguous — it means requests are arriving
# and none are being served. Low MemAvailable alone is NOT a trigger: it dips
# legitimately under healthy heavy load, and reloading then would destroy live
# work for no reason. Memory is logged as a WARN only, as an early signal.
#
# ACTION is a reload, not a kill (contrast bin/mem-watchdog.sh, which SIGKILLs a
# runaway before it hard-hangs the box — that guards a different, fatal case).
# The floor breach is not fatal, it is latched: unloading releases the grown
# slabs and the lane comes back clean. Measured 2026-08-10: 20 s, because the
# fork's derived weight artifacts are already built and get reused. The disk-KV
# prefix cache survives an unload, so warm prefixes are not lost.
#
# Usage: ds4-degraded-watchdog.sh [--dry-run]
# Runs from cron every 2 min — a cheap no-op (two curls) when healthy.
set -u

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

LOG=/home/max/llm-stack/logs/ds4-degraded-watchdog.log
STATE=/home/max/llm-stack/logs/ds4-degraded-watchdog.state
MODEL=deepseek-v4-flash-0731
ENGINE=http://127.0.0.1:9010
SWAP=http://127.0.0.1:8080

# Refusals must climb by at least this much in one tick to count as degraded.
# 1 could be a single genuinely-too-deep prompt; a real floor breach refuses
# every deep request that arrives, so it climbs fast.
MIN_REFUSAL_DELTA="${MIN_REFUSAL_DELTA:-3}"
# Do not reload more than once per this many seconds — prevents a flap loop if
# the reload does not actually fix the condition.
COOLDOWN="${COOLDOWN:-900}"
# MemAvailable below this (GiB) is logged as approaching --mem-floor-gb 8.
MEM_WARN_GIB="${MEM_WARN_GIB:-9}"

log() { echo "$(date -Is) $*" >> "$LOG"; }

met=$(curl -sf -m 5 "$ENGINE/metrics" 2>/dev/null) || {
  # Engine not answering at all: that is ds4-keepalive.sh's job, not ours.
  exit 0
}

field() { # field <outcome>  -> counter value, 0 if absent
  echo "$met" | awk -v k="$1" '$0 ~ "ds4_requests_total\\{outcome=\""k"\"\\}" {print $2; f=1} END{if(!f) print 0}'
}
refused=$(field refused_deep_serial)
completed=$(field completed)
inflight=$(echo "$met" | awk '/^ds4_requests_inflight /{print $2; f=1} END{if(!f) print 0}')
avail_gib=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)

# Non-numeric means the metrics format moved under us — say so, do not guess.
case "$refused$completed" in *[^0-9]*) log "ERROR unparseable metrics (refused='$refused' completed='$completed'); check ds4 /metrics format"; exit 1;; esac

prev_refused=0; prev_completed=0; last_reload=0; prev_warn=0
# shellcheck source=/dev/null
[ -f "$STATE" ] && . "$STATE"

d_refused=$((refused - prev_refused))
d_completed=$((completed - prev_completed))

# Engine restarted (counters reset): re-baseline, never trigger on the dip.
if [ "$refused" -lt "$prev_refused" ] || [ "$completed" -lt "$prev_completed" ]; then
  log "counters reset (engine restarted); re-baselining refused=$refused completed=$completed"
  d_refused=0; d_completed=0
fi

# Early signal only — deliberately NOT a trigger, see header. Logged on
# TRANSITION rather than every tick: MemAvailable hovers near the threshold
# during normal deep traffic, and a line every 2 min would bury the real events
# in this same log. Crossing back out is logged too, so a WARN is never left
# looking open after it has recovered.
warn=0
awk -v a="$avail_gib" -v w="$MEM_WARN_GIB" 'BEGIN{exit !(a < w)}' && warn=1
if [ "$warn" -ne "$prev_warn" ]; then
  if [ "$warn" -eq 1 ]; then
    log "WARN MemAvailable ${avail_gib} GiB fell below ${MEM_WARN_GIB} GiB, approaching --mem-floor-gb 8 (refused=$refused completed=$completed inflight=$inflight)"
  else
    log "INFO MemAvailable recovered to ${avail_gib} GiB (above ${MEM_WARN_GIB} GiB)"
  fi
fi

degraded=0
[ "$d_refused" -ge "$MIN_REFUSAL_DELTA" ] && [ "$d_completed" -le 0 ] && degraded=1

save_state() {
  printf 'prev_refused=%s\nprev_completed=%s\nlast_reload=%s\nprev_warn=%s\n' \
    "$refused" "$completed" "$1" "$warn" > "$STATE"
}

if [ "$degraded" -eq 0 ]; then
  save_state "$last_reload"
  exit 0
fi

now=$(date +%s)
log "DEGRADED refused +$d_refused (=$refused) while completed flat (=$completed); MemAvailable ${avail_gib} GiB, inflight=$inflight"

if [ "$DRY" -eq 1 ]; then
  log "dry-run: would reload $MODEL now"
  save_state "$last_reload"
  exit 0
fi

if [ $((now - last_reload)) -lt "$COOLDOWN" ]; then
  log "still degraded but within ${COOLDOWN}s cooldown of last reload; NOT reloading (investigate: reload is not fixing it)"
  save_state "$last_reload"
  exit 0
fi

# Never yank a live request out from under a client. During a real floor breach
# inflight is 0 (nothing can be admitted), so this does not block the fix.
if [ "$inflight" -gt 0 ]; then
  log "degraded but inflight=$inflight; deferring reload to next tick"
  save_state "$last_reload"
  exit 0
fi

log "reloading $MODEL (unload + warm poke)"
curl -sf -m 30 "$SWAP/unload" >/dev/null 2>&1 || log "WARN unload request returned non-zero"
sleep 2
# Long timeout: ride the cold start out here so a real customer does not.
curl -sf -m 1500 "$SWAP/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  >/dev/null 2>&1
post_avail=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
health=$(curl -sf -m 10 "$ENGINE/v1/models" >/dev/null 2>&1 && echo ok || echo STILL-DOWN)
log "reload finished: health=$health MemAvailable ${post_avail} GiB (was ${avail_gib})"

# Re-baseline against the restarted engine's counters so the next tick compares
# like with like instead of against the pre-reload totals.
met=$(curl -sf -m 5 "$ENGINE/metrics" 2>/dev/null) || met=""
if [ -n "$met" ]; then refused=$(field refused_deep_serial); completed=$(field completed); fi
save_state "$now"
