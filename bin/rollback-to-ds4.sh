#!/bin/bash
# Roll production back from qwen3.8-27b to ds4 (Entrpi). Staged 2026-08-16.
# Safe order is the whole point of this script:
#   1. STOP qwen first (frees ~85 GB)  2. only THEN restore ds4's real cmd
# Doing it the other way risks both engines resident = GB10 hard-hang.
set -euo pipefail
STACK=/home/max/llm-stack
CFG=$STACK/config/llama-swap.yaml
STAMP=$(date +%Y%m%d-%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "backing up $CFG -> .bak.$STAMP-prerollback"
cp "$CFG" "$CFG.bak.$STAMP-prerollback"

log "stopping qwen engine FIRST"
docker rm -f vllm-qwen38-27b 2>/dev/null || true
for _ in $(seq 1 30); do
  [ "$(free -g | awk '/^Mem:/{print $7}')" -gt 90 ] && break; sleep 2
done
log "available: $(free -g | awk '/^Mem:/{print $7}') GB"

log "restoring ds4 as resident (config write deploys — live-watched)"
python3 - "$CFG" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
# ds4 lane gets its real launcher back
s = s.replace("/home/max/llm-stack/bin/eval-window-blocked.sh",
              "/home/max/llm-stack/bin/launch-ds4-entrpi.sh")
# qwen lane becomes the blocked one (mirror of the forward cutover)
s = s.replace("/home/max/llm-stack/bin/launch-vllm-qwen38-prod.sh",
              "/home/max/llm-stack/bin/eval-window-blocked.sh")
# resident + preload flip back to ds4
s = re.sub(r"members:\s*\n\s*- qwen3\.8-27b", "members:\n    - deepseek-v4-flash-0731", s)
s = re.sub(r"preload:\s*\n\s*- qwen3\.8-27b", "preload:\n    - deepseek-v4-flash-0731", s)
open(p, "w").write(s)
EOF
grep -q 'launch-ds4-entrpi.sh' "$CFG" || { echo "ABORT: rewrite failed, check $CFG" >&2; exit 1; }

log "restoring ds4 keepalive + watchdog cron"
( crontab -l 2>/dev/null | grep -v 'qwen38-keepalive\|ds4-keepalive\|ds4-degraded-watchdog' ; \
  echo '*/5 * * * * /home/max/llm-stack/bin/ds4-keepalive.sh' ; \
  echo '*/2 * * * * /home/max/llm-stack/bin/ds4-degraded-watchdog.sh' ) | crontab -

log "waking ds4 (cold start ~10 min; disk-KV cache kv-cache-entrpi is intact)"
curl -sf -m 1800 http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' >/dev/null 2>&1 || true
log "health: $(curl -sf -m 5 http://127.0.0.1:9010/v1/models >/dev/null 2>&1 && echo OK || echo STILL-DOWN)"
echo "Gateway: if routes were remapped to qwen3.8-27b, point them back at"
echo "openai/deepseek-v4-flash-0731 in deployed.yaml + merge (the pre-cutover"
echo "state); .7 backups from the 2026-08-01 pattern are *.bak-*."
