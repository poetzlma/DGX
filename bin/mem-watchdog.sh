#!/bin/bash
# Kill a runaway engine before it hard-hangs the box.
#
# WHY THIS EXISTS: on 2026-08-02 an engine whose *planned* footprint was
# 93.92 GiB silently fell back to a device-side copy and doubled its real
# footprint, exhausting the 121.6 GiB unified pool. GB10 does not recover from
# host OOM — the whole machine hung and needed a physical power cycle.
# "Planned" memory in an engine log is not a safety margin.
#
# Usage: mem-watchdog.sh <process-name> [limit_gib] [poll_s]
#   e.g. mem-watchdog.sh llama-server 112
# Polls MemAvailable; if used crosses the limit, SIGKILLs matching processes.
set -u
NAME="${1:?usage: mem-watchdog.sh <process-name> [limit_gib] [poll_s]}"
LIMIT="${2:-112}"
POLL="${3:-3}"
LOG=/home/max/llm-stack/logs/mem-watchdog.log
TOTAL=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
echo "[$(date -Is)] watchdog armed: $NAME, kill above ${LIMIT} GiB used (total ${TOTAL} GiB)" >> "$LOG"
# Wait for the process to appear (up to GRACE s) before treating "gone" as done.
# Without this the watchdog loses a race against a slow-starting engine and
# exits immediately, leaving the dangerous load phase completely unguarded.
GRACE="${GRACE:-120}"
for _ in $(seq 1 "$GRACE"); do
  pgrep -x "$NAME" >/dev/null && break
  sleep 1
done
if ! pgrep -x "$NAME" >/dev/null; then
  echo "[$(date -Is)] $NAME never appeared within ${GRACE}s; watchdog exiting" >> "$LOG"
  exit 0
fi
echo "[$(date -Is)] $NAME seen; monitoring" >> "$LOG"
while :; do
  avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  used=$((TOTAL - avail))
  if [ "$used" -ge "$LIMIT" ]; then
    pids=$(pgrep -x "$NAME" || true)
    echo "[$(date -Is)] TRIPPED at ${used} GiB used (limit ${LIMIT}); killing: ${pids:-none}" >> "$LOG"
    [ -n "$pids" ] && kill -9 $pids
    echo "MEM-WATCHDOG-TRIPPED at ${used} GiB"
    exit 1
  fi
  pgrep -x "$NAME" >/dev/null || { echo "[$(date -Is)] $NAME gone; watchdog exiting" >> "$LOG"; exit 0; }
  sleep "$POLL"
done
