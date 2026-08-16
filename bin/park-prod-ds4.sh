#!/bin/bash
# Park the prod ds4 lane for an eval window. Pairs with restore-prod-ds4.sh.
#
# 2026-08-14: merged with the short-lived bin/park-ds4-for-eval.sh, which was
# written without noticing this script already existed. Everything below is the
# union of the two; park-ds4-for-eval.sh is now a deprecation shim.
#
# Parks by:
#   (a) pointing the model's cmd at eval-window-blocked.sh AND clearing the
#       on_startup preload, so neither a request nor a llama-swap reload nor a
#       REBOOT can spawn a second ~90 GB engine next to a large eval model.
#       Clearing preload is what saved the box on the 2026-08-02 power cycle:
#       it came back up without trying to auto-load 90 GB.
#   (b) removing BOTH cron entries that poke the lane — ds4-keepalive.sh (*/5)
#       and ds4-degraded-watchdog.sh (*/2, added 2026-08-10). Removing only the
#       keepalive leaves the watchdog to resurrect the lane mid-eval.
#   (c) writing etc/eval-window.current so restore-prod-ds4.sh knows exactly
#       which backups to roll back to, instead of hardcoding dated filenames.
#   (d) refusing to report success until the unified pool has actually been
#       released — a "parked" lane that still holds 100 GB is how you OOM-hang
#       the box on the very next launch.
#
# THIS TAKES THE PAID LANE OFFLINE. Requests fail loudly, by design.
set -eu
log(){ echo "[$(date +%H:%M:%S)] $*"; }
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
STACK=/home/max/llm-stack
CFG=$STACK/config/llama-swap.yaml
ETC=$STACK/etc
NEED_FREE_GB="${NEED_FREE_GB:-100}"

log "backing up config + crontab (stamp $STAMP)"
mkdir -p "$ETC"
BAK_CFG="$CFG.bak.$STAMP-prepark"
BAK_CRON="$ETC/crontab.bak.$STAMP-prepark"
cp "$CFG" "$BAK_CFG"
crontab -l > "$BAK_CRON" 2>/dev/null || : > "$BAK_CRON"
printf 'cfg=%s\ncron=%s\nparked_at=%s\n' "$BAK_CFG" "$BAK_CRON" "$(date -Is)" \
  > "$ETC/eval-window.current"

log "parking model cmd + clearing preload"
# Match ANY ds4 launcher, not just launch-ds4-server.sh: prod moved to
# launch-ds4-entrpi.sh on 2026-08-10, and a sed that silently matches nothing
# would leave the lane loadable during an eval window — the exact thing this
# script exists to prevent. Hence the verify-or-abort below.
sed -i 's#^\( *cmd: \)/home/max/llm-stack/bin/launch-ds4-[A-Za-z0-9._-]*\.sh#\1/home/max/llm-stack/bin/eval-window-blocked.sh#' "$CFG"
grep -q 'cmd: /home/max/llm-stack/bin/eval-window-blocked.sh' "$CFG" || {
  log "ABORT: cmd: line did not get parked — restoring $CFG and doing nothing"
  cp "$BAK_CFG" "$CFG"
  exit 1
}
python3 - "$CFG" <<'EOF'
import re,sys
p=sys.argv[1]; s=open(p).read()
s=re.sub(r"(hooks:\s*\n\s*on_startup:\s*\n\s*preload:)\s*\n\s*- *deepseek-v4-flash-0731",
         r"\1 []", s)
open(p,'w').write(s)
EOF

log "removing keepalive + watchdog cron"
crontab -l 2>/dev/null | grep -v 'ds4-keepalive\|ds4-degraded-watchdog' | crontab - || true

log "stopping ds4-server"
for p in $(ps -eo pid,cmd | awk '/[d]s4-server --cuda/ {print $1}'); do kill "$p" 2>/dev/null || true; done
for _ in $(seq 1 30); do ps -eo pid,cmd | grep -q "[d]s4-server --cuda" || break; sleep 2; done

# Do not claim success until the pool is actually back.
for _ in $(seq 1 60); do
  avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "$avail" -ge "$NEED_FREE_GB" ] && break
  sleep 2
done
avail=$(free -g | awk '/^Mem:/{print $7}')
log "available memory: ${avail} GB"
if [ "$avail" -lt "$NEED_FREE_GB" ]; then
  log "WARNING: only ${avail} GB free (< ${NEED_FREE_GB}). Something still holds the pool."
  log "Do NOT start a large engine. Check: ps -eo pid,cmd | grep ds4-server; docker ps"
  exit 1
fi

log "parked. cmd=$(grep -m1 'cmd:' "$CFG" | tr -s ' '), cron lines=$(crontab -l 2>/dev/null | grep -c . || echo 0)"
log "restore with: $STACK/bin/restore-prod-ds4.sh"
free -g | sed -n 2p
