#!/bin/bash
# Open / close a test window for the ds4 lane by parking the Laguna resident.
#
# WHY THIS EXISTS: ds4 needs ~86 GB and laguna holds ~113 GB of the 121 GB
# unified pool, so they cannot co-exist — the test REQUIRES taking the prod
# coding lane down. Three things will otherwise fight us and bring laguna back
# mid-test (the "standalone runs and llama-swap FIGHT" failure mode: both
# docker rm -f the same container, engines die at exit 137):
#     1. llama-swap's `resident` group is persistent:true / ttl:0
#     2. hooks.on_startup.preload relaunches laguna on every config reload
#     3. bin/laguna-keepalive.sh runs from cron every 5 minutes
# So we disarm all three, and restoring re-arms all three.
#
# Deliberately NOT using systemctl: there is no passwordless sudo on this box,
# and llama-swap live-watches its config anyway, so swapping the config is both
# sufficient and the mechanism llama-swap is designed around.
#
# Usage:
#   ds4-test-window.sh open    # park laguna, free the pool
#   ds4-test-window.sh close   # restore prod exactly as it was
#   ds4-test-window.sh status
set -uo pipefail

CFG=/home/max/llm-stack/config/llama-swap.yaml
BAK=/home/max/llm-stack/config/llama-swap.yaml.bak.ds4-test-window
CRONBAK=/home/max/llm-stack/config/crontab.bak.ds4-test-window
PARKED=/home/max/llm-stack/config/llama-swap.yaml.parked

want_free_gb() { free -g | awk '/^Mem:/{print $7}'; }

case "${1:-status}" in
open)
  if [ -f "$BAK" ]; then
    echo "!! $BAK already exists — a window is already open. Close it first."
    exit 1
  fi
  cp -a "$CFG" "$BAK"
  crontab -l > "$CRONBAK" 2>/dev/null || true

  # 1. Disarm the keepalive cron (comment it, do not delete — restore re-arms).
  crontab -l 2>/dev/null | sed 's|^\(\*/5 .*laguna-keepalive.sh\)|#PARKED# \1|' | crontab -
  echo "cron: $(crontab -l | grep -c '^#PARKED#') keepalive entry parked"

  # 2. Config with NO models and NO preload. llama-swap reconciles on the
  #    live-watch and stops what it is no longer asked to keep resident.
  cat > "$PARKED" <<'YAML'
# PARKED for a ds4-0731 test window — see bin/ds4-test-window.sh.
# Restore with: ds4-test-window.sh close   (copies the .bak back over)
healthCheckTimeout: 1200
logLevel: info
models: {}
groups: {}
YAML
  cp -a "$PARKED" "$CFG"
  echo "config: laguna removed from llama-swap (live-watched, reconciling)"

  # 3. Belt and braces: if llama-swap does not reap the container itself,
  #    stop it directly. Safe now that nothing is configured to restart it.
  for i in $(seq 1 60); do
    if ! docker ps --format '{{.Names}}' | grep -q '^vllm-laguna-s21$'; then
      echo "laguna container gone after ${i}0s"
      break
    fi
    if [ "$i" = 12 ]; then
      echo "llama-swap did not reap it in 120s; stopping the container directly"
      docker stop vllm-laguna-s21 >/dev/null 2>&1
    fi
    sleep 10
  done
  docker rm -f vllm-laguna-s21 >/dev/null 2>&1 || true

  echo "waiting for the unified pool to drain..."
  for i in $(seq 1 30); do
    f=$(want_free_gb)
    [ "${f:-0}" -ge 90 ] && break
    sleep 5
  done
  echo "MemAvailable now: $(want_free_gb) GB  (need >= ~90 for ds4)"
  free -g | sed -n '1,2p'
  ;;

close)
  if [ ! -f "$BAK" ]; then
    echo "!! no $BAK — nothing to restore"
    exit 1
  fi
  # Stop any test engine still holding the pool, or laguna will OOM on load.
  pkill -f 'ds4-server .*--port 909' 2>/dev/null && echo "stopped test ds4-server"
  sleep 5
  cp -a "$BAK" "$CFG"
  rm -f "$BAK" "$PARKED"
  echo "config: restored (preload will bring laguna back, ~10 min cold start)"
  if [ -f "$CRONBAK" ]; then
    crontab "$CRONBAK" && rm -f "$CRONBAK"
    echo "cron: keepalive re-armed"
  fi
  echo "watch: docker logs -f vllm-laguna-s21"
  ;;

status)
  echo "window open:   $([ -f "$BAK" ] && echo YES || echo no)"
  echo "laguna up:     $(docker ps --format '{{.Names}}' | grep -q '^vllm-laguna-s21$' && echo YES || echo no)"
  echo "keepalive cron:$(crontab -l 2>/dev/null | grep -q '^#PARKED#' && echo ' PARKED' || echo ' armed')"
  echo "MemAvailable:  $(want_free_gb) GB"
  ;;
*)
  echo "usage: $0 {open|close|status}"; exit 1;;
esac
