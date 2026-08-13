#!/bin/bash
# Placeholder cmd used ONLY during the 2026-08-02 quant eval window.
# llama-swap points the deepseek-v4-flash-0731 model here so that a stray
# request CANNOT spawn a second ds4-server while a ~100 GB eval model is
# resident — that would exceed the 121.6 GB unified pool and GB10 hard-hangs
# the whole box on host OOM (memory project_coresident_split_20260724).
# Fail fast and loudly instead.
echo "ds4 lane is parked for the quant eval window (see bin/restore-prod-ds4.sh)" >&2
exit 1
