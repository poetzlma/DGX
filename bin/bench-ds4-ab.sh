#!/bin/bash
# ds4 A/B: custom-Q4 (our q4-only fork) vs mainline native-Q4.
# Runs each binary DIRECTLY on a test port (bypasses llama-swap), 3 measured
# decode runs per arm with idle between (GB10 allocator-state confounder:
# decode t/s is ~44x dominated by allocator/page-cache state — idle 2-3 min
# between runs + take the median, or the numbers are noise).
#
# Usage: edit ARM_B_* below with mainline's correct Q4 invocation (from the
# ds4-repo research), then: bash bench-ds4-ab.sh 2>&1 | tee ~/ds4-ab.log
set -uo pipefail

MODEL="/home/max/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
PORT=9099
CTX=131072
IDLE_BETWEEN_RUNS=170     # seconds; >=2 min to settle the allocator
RUNS=3
GENTOK=200
PROMPT='Write a Python function quicksort(arr) with an explanatory docstring and a usage example. Then explain its time complexity in 3 sentences.'

# ─── ARM A: our custom-Q4 fork (control) ──────────────────────────────────
ARM_A_ENV=(DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=1024 DS4_CUDA_Q4_DECODE=1)
ARM_A_BIN="$HOME/ds4-q4/ds4-server"
ARM_A_ARGS=(--cuda --host 127.0.0.1 --port "$PORT" --model "$MODEL" --ctx "$CTX" --kv-disk-dir /tmp/ds4ab-A --kv-disk-space-mb 8192 --warm-weights)

# ─── ARM B: mainline native-Q4 (candidate) — FILL FROM RESEARCH ───────────
# Placeholder; replace ARM_B_ENV / ARM_B_ARGS with mainline's correct
# Q4-routed-MoE / expert-cache invocation before running arm B.
ARM_B_ENV=()
ARM_B_BIN="$HOME/ds4-mainline/ds4-server"
ARM_B_ARGS=(--cuda --host 127.0.0.1 --port "$PORT" --model "$MODEL" --ctx "$CTX" --kv-disk-dir /tmp/ds4ab-B --kv-disk-space-mb 8192 --warm-weights)

measure() {  # one decode run -> prints "tok/s tokens seconds"
  local t0 t1 resp comp
  resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"ds4\",\"messages\":[{\"role\":\"user\",\"content\":$(python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$PROMPT")}],\"max_tokens\":$GENTOK,\"temperature\":0,\"stream\":false}" \
    -w '\n%{time_total}')
  local tt=$(echo "$resp" | tail -1)
  local body=$(echo "$resp" | sed '$d')
  echo "$body" > /tmp/ds4ab-lastresp.json
  comp=$(python3 -c "import json;d=json.load(open('/tmp/ds4ab-lastresp.json'));print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
  python3 -c "print(f'{($comp/$tt) if $tt>0 else 0:.2f} {$comp} {$tt:.2f}')"
}

run_arm() {
  local label="$1"; shift
  local -n ENV=$1; local BIN=$2; local -n ARGS=$3
  echo; echo "############ ARM: $label ($BIN) ############"
  rm -rf /tmp/ds4ab-A /tmp/ds4ab-B 2>/dev/null
  env "${ENV[@]}" "$BIN" "${ARGS[@]}" > "/tmp/ds4ab-server-$label.log" 2>&1 &
  local SPID=$!
  echo "  server PID $SPID; waiting for /v1/models ..."
  local up=0
  for i in $(seq 1 150); do
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200 && { up=1; echo "  up after ~$((i*2))s"; break; }
    sleep 2
  done
  if [ $up -eq 0 ]; then echo "  ✗ never came up; tail:"; tail -20 "/tmp/ds4ab-server-$label.log"; kill $SPID 2>/dev/null; return 1; fi
  echo "  warm-up gen ..."; measure >/dev/null
  echo "  --- $RUNS measured runs (idle ${IDLE_BETWEEN_RUNS}s between) ---"
  local vals=()
  for r in $(seq 1 $RUNS); do
    local m=$(measure); local tps=$(echo "$m" | awk '{print $1}')
    echo "    run $r: $m  (tok/s tokens s)"; vals+=("$tps")
    [ "$r" -lt "$RUNS" ] && { echo "    idle ${IDLE_BETWEEN_RUNS}s ..."; sleep "$IDLE_BETWEEN_RUNS"; }
  done
  local median=$(printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  echo "  >>> $label MEDIAN decode: $median tok/s"
  echo "  sample output:"; python3 -c "import json;d=json.load(open('/tmp/ds4ab-lastresp.json'));m=d['choices'][0]['message'];print((m.get('content') or m.get('reasoning_content') or '')[:400])" 2>/dev/null
  kill $SPID 2>/dev/null; sleep 5
  echo "  (cooldown 60s before next arm)"; sleep 60
}

echo "ds4 A/B start $(date -u +%H:%M:%S)  model=$(basename "$MODEL")"
run_arm A ARM_A_ENV "$ARM_A_BIN" ARM_A_ARGS
if [ -n "${ONLY_A:-}" ]; then echo "(ONLY_A set — skipping arm B until mainline flags confirmed)"; else
run_arm B ARM_B_ENV "$ARM_B_BIN" ARM_B_ARGS
fi
echo; echo "ds4 A/B done $(date -u +%H:%M:%S)"
