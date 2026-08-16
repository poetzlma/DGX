#!/bin/bash
# Promote qwen3.8-27b to the resident production lane. Staged 2026-08-16.
# DO NOT RUN until the user has signed off on the matrix medians and the
# reasoning-parser decision. Pairs with rollback-to-ds4.sh.
#
# What it does (all local to .12; the GATEWAY remap on .7 is a separate,
# manual step — see the checklist it prints at the end):
#   1. sanity: ds4 must be parked, no engine resident
#   2. rewrite llama-swap.yaml: qwen3.8-27b becomes the resident+preload;
#      the deepseek lane stays DEFINED but pointed at eval-window-blocked.sh
#      (a restored ds4 cmd beside a live qwen = stray request -> 2nd engine ->
#      hard-hang; rollback-to-ds4.sh handles the order safely)
#   3. install qwen38-keepalive.sh in cron (replacing ds4 entries)
#   4. wake the lane through llama-swap and verify
set -euo pipefail
STACK=/home/max/llm-stack
CFG=$STACK/config/llama-swap.yaml
STAMP=$(date +%Y%m%d-%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1. sanity
pgrep -x ds4-server >/dev/null 2>&1 && { echo "ABORT: ds4 is running. park-prod-ds4.sh first." >&2; exit 1; }
grep -q 'eval-window-blocked.sh' "$CFG" || { echo "ABORT: ds4 lane not parked in $CFG." >&2; exit 1; }
docker ps --format '{{.Names}}' | grep -q vllm && log "note: eval engine running — llama-swap will manage it via the new config"

log "backing up $CFG -> .bak.$STAMP-precutover"
cp "$CFG" "$CFG.bak.$STAMP-precutover"

# 2. write the new config (llama-swap is live-watched: this DEPLOYS on write)
cat > "$CFG" <<'YAML'
# PRODUCTION = qwen3.8-27b (cut over via bin/cutover-qwen38.sh; see that script
# and bin/launch-vllm-qwen38-prod.sh headers for every measured rationale).
# Rollback = bin/rollback-to-ds4.sh (order matters: stop qwen FIRST).
#
# The deepseek lane below is deliberately BLOCKED, not removed: its route names
# still exist at the gateway and a stray request must fail loudly rather than
# spawn a second ~113 GB engine beside qwen (GB10 hard-hangs on host OOM).
healthCheckTimeout: 1800
logLevel: info
models:
  qwen3.8-27b:
    cmd: /home/max/llm-stack/bin/launch-vllm-qwen38-prod.sh
    proxy: http://127.0.0.1:9030
    checkEndpoint: /v1/models
    ttl: 0
  deepseek-v4-flash-0731:
    # PARKED (rollback target). Real launcher: bin/launch-ds4-entrpi.sh —
    # restored ONLY by rollback-to-ds4.sh after qwen is stopped.
    cmd: /home/max/llm-stack/bin/eval-window-blocked.sh
    proxy: http://127.0.0.1:9010
    checkEndpoint: /v1/models
    ttl: 0
groups:
  resident:
    swap: false
    exclusive: false
    persistent: true
    members:
    - qwen3.8-27b

hooks:
  on_startup:
    preload:
    - qwen3.8-27b
YAML
log "llama-swap.yaml written (live-watched -> deploying now)"

# 3. cron
log "installing qwen38 keepalive cron"
( crontab -l 2>/dev/null | grep -v 'ds4-keepalive\|ds4-degraded-watchdog\|qwen38-keepalive' ; \
  echo '*/5 * * * * /home/max/llm-stack/bin/qwen38-keepalive.sh' ) | crontab -

# 4. wake + verify
log "waking the lane (cold start ~8 min)"
curl -sf -m 1500 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"ping"}],"max_tokens":8}' >/dev/null 2>&1 || true
health=$(curl -sf -m 5 http://127.0.0.1:9030/v1/models >/dev/null 2>&1 && echo OK || echo DOWN)
log "lane health: $health"
free -g | sed -n 2p

cat <<'EOF'

REMAINING MANUAL STEPS (gateway, on .7 via deployed.yaml merge):
  1. deployed.yaml: point qwen3.6-27b / qwen3.6-35b-a3b / laguna-s-2.1 /
     deepseek-v4-flash-* / nemotron-3-puzzle-75b routes at openai/qwen3.8-27b,
     add a first-class qwen3.8-27b route. Set pricing (default: standard tier).
  2. litellm-merge.py + docker compose up -d --force-recreate litellm
     (NEVER docker restart litellm — stale-cmd crash-loop gotcha).
  3. Verify: spend>0 with prompt_tokens>0 on a test request (pricing-vanish
     gotcha of 2026-06/08); smoke one request per route name.
  4. Watch logs/qwen38-keepalive.log for NVRM warnings the first day.
EOF
