#!/usr/bin/env python3
"""Ad-hoc concurrency sweep on an already-loaded model. Skips cold start."""
import json
import statistics
import sys
import time
import urllib.request
sys.path.insert(0, "/home/max/llm-stack/bin")
from importlib import import_module
m = import_module("bench-deep")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nemotron-3-nano-omni"
LEVELS = [int(x) for x in (sys.argv[2:] or ["1", "3", "5", "8", "12", "16"])]

# Read available KV blocks (best signal for "could we fit more seqs?")
def kv_info():
    try:
        with urllib.request.urlopen(f"{m.GATEWAY}/v1/models", timeout=10) as r:
            json.loads(r.read())  # just confirms gateway warm
    except Exception:
        pass

print(f"Sweep: {MODEL}  levels={LEVELS}")
print(f"{'c':>3} {'agg_tok/s':>10} {'per_req':>8} {'ttft_ms':>8} {'lat_s':>6} {'wall_s':>6} {'errs':>4}")
print("-" * 60)

results = []
for n in LEVELS:
    cr = m.run_concurrent(MODEL, m.SHORT_MSGS, 512, n)
    results.append({"n": n, **cr})
    print(f"{n:>3} {cr['agg_tok_s']:>10.1f} {cr['mean_tok_s']:>8.1f} "
          f"{cr['mean_ttft_ms']:>8.0f} {cr['mean_latency_s']:>6.1f} "
          f"{cr['wall_time_s']:>6.1f} {cr['errors']:>4}")
    time.sleep(2)

# Find peak
ok = [r for r in results if r["errors"] == 0]
if ok:
    peak = max(ok, key=lambda r: r["agg_tok_s"])
    print(f"\nPeak agg throughput: c={peak['n']}  →  {peak['agg_tok_s']:.1f} tok/s")

with open("/home/max/llm-stack/logs/bench-concurrency-sweep.json", "w") as f:
    json.dump(results, f, indent=2)
