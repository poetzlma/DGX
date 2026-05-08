#!/home/max/llm-stack/venv/bin/python3
"""
Big-context concurrency sweep on an already-loaded model.

For each concurrency level n in LEVELS, fire n requests in parallel, each with
a ~TARGET_INPUT_TOKENS prompt with a unique nonce so the prefix cache cannot
mask the prefill cost. Reports aggregate tok/s, per-request decode tok/s,
TTFT, end-to-end latency, and KV usage.

Usage:
    bench-bigctx-concurrency.py MODEL [TARGET_INPUT_TOKENS] [LEVELS...]

Defaults:
    MODEL                 = qwen3.6-27b-mtp
    TARGET_INPUT_TOKENS   = 100000
    LEVELS                = 1 3 5 8
"""
import json
import sys
import time
from importlib import import_module

sys.path.insert(0, "/home/max/llm-stack/bin")
m = import_module("bench-deep")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-27b-mtp"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
LEVELS = [int(x) for x in (sys.argv[3:] or ["1", "3", "5", "8"])]
OUT_TOKENS = 256

# Each filler block is ~370 tokens; tune repeats to hit TARGET.
TOKENS_PER_FILLER = 370
REPEATS = max(1, TARGET // TOKENS_PER_FILLER)


def make_msgs(nonce):
    prompt = (
        f"[NONCE-{nonce}] "
        + (m._FILLER * REPEATS)
        + "\n\nSummarize the above text in exactly 3 short bullets."
    )
    return [{"role": "user", "content": prompt}]


def run_level(n):
    """Build n distinct prompts (unique nonces) and fire them concurrently."""
    import threading
    results = [None] * n
    errors = [None] * n
    msgs_list = [make_msgs(f"{time.time_ns()}-{i}") for i in range(n)]

    def worker(i):
        try:
            results[i] = m.chat_streaming(MODEL, msgs_list[i], OUT_TOKENS, timeout=900)
        except Exception as e:
            errors[i] = str(e)[:200]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r]
    err_count = sum(1 for e in errors if e)
    if not ok:
        return {
            "n": n, "wall_s": wall, "errors": err_count,
            "first_error": next((e for e in errors if e), None),
        }
    ttfts = [r["ttft_ms"] for r in ok]
    out_toks = [r["completion_tokens"] for r in ok]
    e2e = [r["total_s"] for r in ok]
    decode = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    agg_tok_s = sum(out_toks) / wall
    return {
        "n": n,
        "wall_s": wall,
        "errors": err_count,
        "agg_tok_s": agg_tok_s,
        "mean_per_req_decode_tok_s": (sum(decode) / len(decode)) if decode else None,
        "mean_ttft_ms": sum(ttfts) / len(ttfts),
        "mean_latency_s": sum(e2e) / len(e2e),
        "max_latency_s": max(e2e),
    }


def main():
    print(
        f"Big-context concurrency sweep: {MODEL}  "
        f"target_input≈{TARGET} tok ({REPEATS}× filler)  "
        f"out={OUT_TOKENS}  levels={LEVELS}"
    )
    print(
        f"{'c':>3} {'agg_t/s':>8} {'per_req_t/s':>11} "
        f"{'ttft_s':>7} {'mean_lat':>9} {'max_lat':>8} {'wall_s':>7} {'errs':>4}"
    )
    print("-" * 70)
    results = []
    for n in LEVELS:
        r = run_level(n)
        results.append(r)
        if r.get("agg_tok_s") is None:
            print(f"{n:>3}  ALL FAILED ({r['errors']}): {r.get('first_error')}")
        else:
            print(
                f"{n:>3} {r['agg_tok_s']:>8.1f} "
                f"{(r['mean_per_req_decode_tok_s'] or 0):>11.1f} "
                f"{r['mean_ttft_ms']/1000:>7.1f} "
                f"{r['mean_latency_s']:>9.1f} {r['max_latency_s']:>8.1f} "
                f"{r['wall_s']:>7.1f} {r['errors']:>4}"
            )
        time.sleep(3)

    out = "/home/max/llm-stack/logs/bench-bigctx-concurrency.json"
    with open(out, "w") as f:
        json.dump({"model": MODEL, "target_input": TARGET, "results": results},
                  f, indent=2)
    print(f"\nResults: {out}")


if __name__ == "__main__":
    main()
