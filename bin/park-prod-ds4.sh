#!/bin/bash
# Park the prod ds4 lane for an eval window. Pairs with restore-prod-ds4.sh.
#
# Parks by (a) pointing the model's cmd at eval-window-blocked.sh and clearing
# on_startup preload, so NO request and NO llama-swap reload can spawn a second
# ~90 GB engine next to a ~100 GB eval model, and (b) removing the keepalive
# cron so it stops poking port 9010 every 5 minutes.
#
# This is what saved the box on the 2026-08-02 power cycle: with preload
# cleared, the machine came back up without trying to auto-load 90 GB.
set -eu
log(){ echo "[$(date +%H:%M:%S)] $*"; }
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
CFG=/home/max/llm-stack/config/llama-swap.yaml

log "backing up config + crontab (stamp $STAMP)"
mkdir -p /home/max/llm-stack/etc
cp "$CFG" "$CFG.bak.$STAMP-prepark"
crontab -l > "/home/max/llm-stack/etc/crontab.bak.$STAMP-prepark" 2>/dev/null || true
# restore-prod-ds4.sh reads these fixed names:
cp "$CFG" /home/max/llm-stack/config/llama-swap.yaml.bak.20260802-preeval
crontab -l > /home/max/llm-stack/etc/crontab.bak.20260802-preeval 2>/dev/null || true

log "parking model cmd + clearing preload"
# Match ANY ds4 launcher, not just launch-ds4-server.sh: prod moved to
# launch-ds4-entrpi.sh on 2026-08-10, and a sed that silently matches nothing
# would leave the lane loadable during an eval window — the exact thing this
# script exists to prevent. Hence the verify-or-abort below.
sed -i 's#^\( *cmd: \)/home/max/llm-stack/bin/launch-ds4-[A-Za-z0-9._-]*\.sh#\1/home/max/llm-stack/bin/eval-window-blocked.sh#' "$CFG"
grep -q 'cmd: /home/max/llm-stack/bin/eval-window-blocked.sh' "$CFG" || {
  log "ABORT: cmd: line did not get parked — restoring $CFG and doing nothing"
  cp "$CFG.bak.$STAMP-prepark" "$CFG"
  exit 1
}
python3 - "$CFG" <<'EOF'
import re,sys
p=sys.argv[1]; s=open(p).read()
s=re.sub(r"(hooks:\s*\n\s*on_startup:\s*\n\s*preload:)\s*\n\s*- *deepseek-v4-flash-0731",
         r"\1 []", s)
open(p,'w').write(s)
EOF

log "removing keepalive cron"
crontab -l 2>/dev/null | grep -v "ds4-keepalive.sh" | crontab - || true

log "stopping ds4-server"
for p in $(ps -eo pid,cmd | awk '/[d]s4-server --cuda/ {print $1}'); do kill "$p" 2>/dev/null || true; done
for i in $(seq 1 30); do ps -eo pid,cmd | grep -q "[d]s4-server --cuda" || break; sleep 2; done

log "parked. cmd=$(grep -m1 'cmd:' "$CFG" | tr -s ' '), cron lines=$(crontab -l 2>/dev/null | grep -c . || echo 0)"
free -g | sed -n 2p
