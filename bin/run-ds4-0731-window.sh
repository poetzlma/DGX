#!/bin/bash
# One-shot driver for the ds4-0731 test window: load -> smoke -> bench -> stop.
#
# Runs unattended so the parked-prod window is spent on measurement instead of
# on round-trips. Everything lands in logs/ds4-0731-window-<stamp>/.
#
# ORDER IS DELIBERATE: smoke runs BEFORE the bench. It doubles as the warm-up,
# and our 20.8 tok/s baseline was itself recorded on steady-state queries after
# warm-up (the first decode after load populates the Q4 lazy cache — a cold
# first query measured 11.75 tok/s against 20.8 warm), so benching a cold engine
# would understate 0731 and invent a regression that isn't there. It also gates:
# if correctness hard-fails there is no point spending 35 min on numbers.
#
# Assumes the window is already open (bin/ds4-test-window.sh open) — this script
# does NOT park laguna itself, so it can never take prod down by accident.
set -uo pipefail

PORT="${DS4_PORT:-9099}"
MODEL="${DS4_MODEL:-/home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-0731.gguf}"
STAMP=$(date +%Y%m%d-%H%M%S)
DIR=/home/max/llm-stack/logs/ds4-0731-window-$STAMP
mkdir -p "$DIR"
echo "logging to $DIR"

if [ ! -f "$MODEL" ]; then echo "!! model missing: $MODEL"; exit 1; fi
free_gb=$(free -g | awk '/^Mem:/{print $7}')
if [ "${free_gb:-0}" -lt 90 ]; then
  echo "!! only ${free_gb} GB available — ds4 needs ~86 GB. Is the window open?"
  echo "   run: bin/ds4-test-window.sh open"
  exit 1
fi

echo "=== launching ds4-server on :$PORT with $(basename "$MODEL") ==="
DS4_PORT="$PORT" DS4_MODEL="$MODEL" DS4_BIN="${DS4_BIN:-/home/max/ds4-q4/ds4-server}" \
  /home/max/llm-stack/bin/launch-ds4-server-0731.sh > "$DIR/server.log" 2>&1 &
echo "binary: ${DS4_BIN:-/home/max/ds4-q4/ds4-server}" | tee "$DIR/binary.txt"
SRV=$!
cleanup() {
  echo "=== stopping ds4-server ==="
  kill "$SRV" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 2; done
  kill -9 "$SRV" 2>/dev/null
}
trap cleanup EXIT

echo "waiting for load (cold start touches 80.76 GiB of tensor pages, ~2 min)..."
up=0
for i in $(seq 1 300); do
  if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" -o /dev/null; then up=1; echo "up after $((i*3))s"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "!! SERVER EXITED DURING LOAD — this is the gating question answered NO."
    echo "--- last 40 lines of server.log ---"; tail -40 "$DIR/server.log"
    exit 2
  fi
  sleep 3
done
[ "$up" = 1 ] || { echo "!! never came up in 900s"; tail -40 "$DIR/server.log"; exit 3; }

echo
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  echo "=== SMOKE skipped (SKIP_SMOKE=1) — bench does its own 3-query warm-up ==="
  SMOKE=0
else
echo "=== SMOKE (correctness gate + warm-up) ==="
DS4_PORT="$PORT" python3 /home/max/llm-stack/bin/smoke-ds4-0731.py 2>&1 | tee "$DIR/smoke.log"
SMOKE=${PIPESTATUS[0]}
echo "smoke exit=$SMOKE"
fi
cp -f /home/max/llm-stack/logs/smoke-ds4-0731.json "$DIR/" 2>/dev/null

if [ "$SMOKE" != 0 ]; then
  echo
  echo "!! smoke had HARD failures — benching anyway (numbers still tell us"
  echo "   whether the loader path is sane), but do NOT promote on this run."
fi

echo
echo "=== BENCH (ctx sweep + concurrency) ==="
# DS4_SERVER_LOG: ds4 emits no `usage` on the streaming SSE, so the bench reads
# the engine's own `chat ctx=A..B:N prompt start` line to learn how many tokens
# were actually prefilled (N is 0 on a disk-KV cache hit — which is exactly the
# distinction we need to keep prefill numbers honest).
DS4_PORT="$PORT" DS4_SERVER_LOG="$DIR/server.log" \
  python3 /home/max/llm-stack/bin/bench-ds4-0731.py 2>&1 | tee "$DIR/bench.log"
cp -f /home/max/llm-stack/logs/bench-ds4-0731.json "$DIR/" 2>/dev/null

echo
echo "=== engine log tail (watch for q8 fallback / cache-reserve warnings) ==="
grep -iE "q8|fallback|reserve|host registration|cuda|warn|error" "$DIR/server.log" \
  | tail -25 | tee "$DIR/server-warnings.log"

echo
echo "DONE — artifacts in $DIR"
