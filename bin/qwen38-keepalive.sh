#!/bin/bash
# Keep the qwen3.8-27b resident warm + NVRM early-warning. Staged 2026-08-16.
# Cron */5 after cutover (replaces ds4-keepalive.sh — swap in the same crontab
# edit that restores cron at cutover; see cutover-qwen38.sh).
LOG=/home/max/llm-stack/logs/qwen38-keepalive.log

# NVRM early warning: NV_ERR_NO_MEMORY precedes GB10 hard-hangs by minutes
# (2026-08-15 post-mortem). Log it loudly even while the engine is healthy.
hits=$(journalctl -k --since "-5 min" 2>/dev/null | grep -c "NV_ERR_NO_MEMORY")
if [ "${hits:-0}" -gt 0 ]; then
  echo "$(date -Is) WARNING: ${hits} NVRM NV_ERR_NO_MEMORY in last 5 min — OOM-hang precursor" >> "$LOG"
fi

if curl -sf -m 5 http://127.0.0.1:9030/v1/models >/dev/null 2>&1; then
  exit 0
fi

echo "$(date -Is) engine down — triggering llama-swap relaunch" >> "$LOG"
# Rides through the ~8 min cold start so the engine warms without a customer.
curl -sf -m 1500 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null 2>&1
echo "$(date -Is) relaunch trigger finished (health: $(curl -sf -m 5 http://127.0.0.1:9030/v1/models >/dev/null 2>&1 && echo ok || echo STILL-DOWN))" >> "$LOG"
