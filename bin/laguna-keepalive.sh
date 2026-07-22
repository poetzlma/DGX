#!/bin/bash
# Keep the laguna-s-2.1 resident warm: if the engine is down, poke llama-swap
# so it relaunches (covers crashes; reboots are covered by hooks.on_startup
# preload in llama-swap.yaml). Runs from cron every 5 min — cheap no-op when
# the engine is healthy. Cron-installed 2026-07-22 (crontab -l to inspect).
LOG=/home/max/llm-stack/logs/laguna-keepalive.log

if curl -sf -m 5 http://127.0.0.1:9030/health >/dev/null 2>&1; then
  exit 0
fi

echo "$(date -Is) engine down — triggering llama-swap relaunch" >> "$LOG"
# Long timeout: this request rides through the ~10 min cold start, so the
# engine is warm again without waiting for a real customer request.
curl -sf -m 1500 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null 2>&1
echo "$(date -Is) relaunch trigger finished (health: $(curl -sf -m 5 http://127.0.0.1:9030/health >/dev/null 2>&1 && echo ok || echo STILL-DOWN))" >> "$LOG"
