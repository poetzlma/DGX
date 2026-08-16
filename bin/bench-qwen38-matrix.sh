#!/bin/bash
# Full Qwen3.8-27B evaluation matrix. Written 2026-08-14.
#
# USAGE:  bench-qwen38-matrix.sh PORT LABEL [REPEATS]
#   e.g.  bench-qwen38-matrix.sh 9030 nvfp4-mtp 3
#
# WHY THE REPEATS AND THE IDLE GAPS — do not "optimise" these away:
# On GB10 decode tok/s is dominated by ALLOCATOR STATE, not by kernels; we have
# measured the same config differing by ~44x across back-to-back runs. A single
# run of anything on this box is noise. Every point here is a 3-run median with
# a >=150 s idle gap between runs so the allocator settles. This is the same
# discipline that exposed the 2026-06-10 "ngc/main regression" as an artefact.
#
# max_tokens: OUT env, default 1024.
# These are thinking-on-by-default models, and on ds4 a 1024 budget was consumed
# ENTIRELY by the <think> block — think_chunks 1023/1024, first_content_ms null,
# zero visible output. So 1024 would be too small there.
# Qwen3.8 is a different animal: it answered a coding prompt in 124 tokens and
# stopped (finish=stop), so 1024 leaves ample headroom for thinking AND the
# answer while roughly halving wall time per level. Raise OUT to 2048+ if you
# ever bench a model that rambles, or first_content_ms will come back null and
# the decode window will be measured over thinking alone.
#
# CONTEXT PLAN:
#   120k : full c=1,2,3,4 sweep. This is the user's minimum requirement.
#   250k : c=1 and c=2 only. 250k x 4 streams = 1M tokens of KV, which will not
#          fit; our Qwen3.6 notes put the practical ceiling at ~150k per stream
#          at c=4. Pushing c=4 at 250k is how you OOM the box, not how you
#          measure it.
set -u
PORT="${1:?usage: bench-qwen38-matrix.sh PORT LABEL [REPEATS]}"
LABEL="${2:?usage: bench-qwen38-matrix.sh PORT LABEL [REPEATS]}"
REPEATS="${3:-3}"
IDLE="${IDLE:-150}"
MODEL="${MODEL:-qwen3.8-27b}"
BIN=/home/max/llm-stack/bin
OUTDIR=/home/max/llm-stack/logs
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# SINGLE-INSTANCE LOCK. On 2026-08-14 two copies of this script ran against the
# same engine for ~2.5 h. Their request batches interleaved, so every level was
# contending with an unrelated batch: TTFT inflated to 183-535 s at 120k and
# decode fell to 2-8 tok/s, while the engine logged "Running: 4, Waiting: 2"
# during what was supposed to be a c=2 level. Every number was garbage and the
# whole set had to be thrown away. A concurrency benchmark MUST own the engine.
LOCK=/tmp/bench-qwen38-matrix.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "REFUSING: another bench-qwen38-matrix.sh already holds $LOCK." >&2
  echo "A second instance would contend for the engine and invalidate both runs." >&2
  ps -eo pid,etime,args --no-headers | grep '[b]ench-qwen38-matrix.sh' >&2
  exit 1
fi
echo "$$" >&9

# Refuse to start if the engine already has traffic — same contamination risk
# from any other source (a stray smoke test, a manual curl loop).
busy=$(curl -sf -m 5 "http://127.0.0.1:${1}/metrics" 2>/dev/null \
       | awk -F' ' '/^vllm:num_requests_running/{print int($2)}' | tail -1)
if [ -n "${busy:-}" ] && [ "${busy:-0}" -gt 0 ]; then
  echo "REFUSING: engine on :$1 already has $busy request(s) in flight." >&2
  echo "Wait for it to drain, or the measurements will be contended." >&2
  exit 1
fi

run_set () {          # $1=ctx tokens  $2=comma levels
  local ctx="$1" levels="$2" i
  for i in $(seq 1 "$REPEATS"); do
    log "=== ctx=${ctx} levels=${levels} run ${i}/${REPEATS} ==="
    BENCH_GATEWAY_HOST=127.0.0.1 BENCH_GATEWAY_PORT="$PORT" \
    CONC_LEVELS="$levels" CONC_INPUT="$ctx" CONC_OUT="${OUT:-1024}" \
      /home/max/llm-stack/venv/bin/python3 "$BIN/bench-coding-conc.py" \
        "$MODEL" "${LABEL}-ctx${ctx}-run${i}" || log "run ${i} FAILED (continuing)"
    if [ "$i" -lt "$REPEATS" ]; then
      log "idle ${IDLE}s so the allocator settles"
      sleep "$IDLE"
    fi
  done
}

log "waiting for :$PORT to answer /v1/models"
for _ in $(seq 1 240); do
  curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
done
curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { log "engine never came up"; exit 1; }
log "engine is up"

run_set 120000 "1,2,3,4"
log "idle ${IDLE}s before the long-context set"
sleep "$IDLE"
run_set 250000 "1,2"

log "=== MEDIANS ==="
/home/max/llm-stack/venv/bin/python3 - "$LABEL" <<'PY'
import glob, json, statistics, sys
label = sys.argv[1]
rows = {}
for p in glob.glob(f"/home/max/llm-stack/logs/bench-coding-conc-{label}-ctx*-run*.json"):
    d = json.load(open(p))
    for r in d["rows"]:
        rows.setdefault((d["target_input"], r["conc"]), []).append(r)
print(f"{'ctx':>8} {'c':>2} {'runs':>4} {'dec/req':>8} {'dec_agg':>8} "
      f"{'ttft_s':>8} {'vis_s':>8} {'err':>4}")
for (ctx, c) in sorted(rows):
    rs = rows[(ctx, c)]
    def med(k, scale=1.0):
        vals = [r[k] for r in rs if r.get(k) is not None]
        return statistics.median(vals) * scale if vals else float("nan")
    print(f"{ctx:>8} {c:>2} {len(rs):>4} {med('decode_per_req_mean'):>8.1f} "
          f"{med('decode_agg'):>8.1f} {med('ttft_ms_mean', 1e-3):>8.1f} "
          f"{med('first_content_ms_mean', 1e-3):>8.1f} "
          f"{sum(r['errors'] for r in rs):>4}")
PY
log "done"
