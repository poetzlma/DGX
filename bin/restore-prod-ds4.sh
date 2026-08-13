#!/bin/bash
# Undo the 2026-08-02 quant-eval window and put the prod ds4 lane back.
# Run this when the eval is done (or if anything goes sideways).
set -u
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "stopping any eval servers on 9098/9099"
pkill -f "llama-server .*ds4-unsloth-UD-IQ3_XXS" 2>/dev/null || true
pkill -f "ds4-server .*--port 9099"              2>/dev/null || true
sleep 3

log "restoring llama-swap.yaml"
cp /home/max/llm-stack/config/llama-swap.yaml.bak.20260802-preeval \
   /home/max/llm-stack/config/llama-swap.yaml
# -watch-config picks this up live; no restart needed.

log "restoring keepalive cron"
crontab /home/max/llm-stack/etc/crontab.bak.20260802-preeval

log "waking the prod lane (cold start is ~10 min on this model)"
curl -sf -m 1800 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null 2>&1
log "health: $(curl -sf -m 5 http://127.0.0.1:9010/v1/models >/dev/null 2>&1 && echo OK || echo STILL-DOWN)"
free -g | head -2
