#!/bin/bash
# Undo park-ds4-for-eval.sh and put the production ds4 lane back.
# Rewritten 2026-08-14: reads etc/eval-window.current instead of hardcoding a
# dated backup filename (the old 2026-08-02-specific version is kept as
# restore-prod-ds4.sh.bak.20260802-window).
#
# Run this when an eval window is done OR the moment anything looks wrong. It
# is safe to run twice.
set -uo pipefail
STACK=/home/max/llm-stack
ETC=$STACK/etc
PTR=$ETC/eval-window.current
log(){ echo "[$(date +%H:%M:%S)] $*"; }

if [ ! -f "$PTR" ]; then
  echo "no $PTR — nothing was parked by park-ds4-for-eval.sh." >&2
  echo "If you parked by hand, restore config/llama-swap.yaml from the newest" >&2
  echo "*-prepark backup and reinstall crontab from etc/ yourself." >&2
  exit 1
fi
# shellcheck disable=SC1090
BAK_CFG=$(awk -F= '/^cfg=/{print $2}' "$PTR")
BAK_CRON=$(awk -F= '/^cron=/{print $2}' "$PTR")
log "restoring from window opened $(awk -F= '/^parked_at=/{print $2}' "$PTR")"

log "stopping any eval engines"
docker rm -f vllm-qwen38-27b 2>/dev/null || true
pkill -f 'llama-server .*--port 909' 2>/dev/null || true
pkill -f 'ds4-server .*--port 909'   2>/dev/null || true
sleep 3

if [ -f "$BAK_CFG" ]; then
  log "restoring llama-swap.yaml from $BAK_CFG"
  cp "$BAK_CFG" "$STACK/config/llama-swap.yaml"   # -watch-config picks this up live
else
  echo "MISSING $BAK_CFG — restore the cmd: line to launch-ds4-entrpi.sh by hand" >&2
fi

if [ -f "$BAK_CRON" ]; then
  log "restoring crontab from $BAK_CRON"
  crontab "$BAK_CRON"
else
  echo "MISSING $BAK_CRON — reinstall the keepalive/watchdog cron entries by hand" >&2
fi

log "waking the prod lane (cold start on this model is ~10 min)"
curl -sf -m 1800 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null 2>&1
health=$(curl -sf -m 5 http://127.0.0.1:9010/v1/models >/dev/null 2>&1 && echo OK || echo STILL-DOWN)
log "health: $health"
free -g | head -2
if [ "$health" = "OK" ]; then
  mv "$PTR" "$PTR.closed.$(date +%Y%m%d-%H%M%S)"
  log "RESTORED. Window closed."
else
  echo "ds4 did NOT come back. Check: journalctl -u llama-swap; tail logs/ds4-keepalive.log" >&2
  exit 1
fi
