#!/bin/bash
# DEPRECATED 2026-08-14 — use bin/park-prod-ds4.sh.
#
# This script was written without noticing park-prod-ds4.sh already existed
# (AGENTS.md names that one as canonical). It was missing the preload-clearing
# step, which is the part that lets the box survive a reboot during an eval
# window without auto-loading 90 GB. Everything useful it did — removing the
# watchdog cron as well as the keepalive, writing etc/eval-window.current for
# restore, and refusing to report success until the pool is actually free — has
# been merged into park-prod-ds4.sh.
#
# Kept as a shim so anything referencing this path still does the right thing.
echo "park-ds4-for-eval.sh is deprecated; running bin/park-prod-ds4.sh instead." >&2
exec /home/max/llm-stack/bin/park-prod-ds4.sh "$@"
