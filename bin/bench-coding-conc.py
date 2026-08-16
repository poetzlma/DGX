#!/home/max/llm-stack/venv/bin/python3
"""Realistic-content concurrency bench (companion to bench-coding-realistic).

Fires N concurrent chat requests with code-shaped ~target-token prompts, each
with a unique nonce (no cross-request prefix cache). Reports per-request decode
tok/s, TTFT, and aggregate tok/s per concurrency level.

Usage:
    BENCH_GATEWAY_PORT=9030 bench-coding-conc.py MODEL [LABEL]

Env:
    CONC_LEVELS   comma list, default "1,2,4"
    CONC_INPUT    target input tokens per request, default 60000
    CONC_OUT      max output tokens per request, default 1024
"""
import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import import_module

sys.path.insert(0, "/home/max/llm-stack/bin")
deep = import_module("bench-deep")
real = import_module("bench-coding-realistic")


def bucket_for(target_input: int, max_tokens: int) -> dict:
    # reuse make_prompt sizing; name only feeds the task lookup, use closest
    name = min(real.BUCKETS,
               key=lambda b: abs(b["target_input"] - target_input))["name"]
    return {"name": name, "target_input": target_input, "max_tokens": max_tokens}


def main():
    if len(sys.argv) < 2:
        print("Usage: bench-coding-conc.py MODEL [LABEL]", file=sys.stderr)
        sys.exit(2)
    model = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else model.replace("/", "_")
    levels = [int(x) for x in os.environ.get("CONC_LEVELS", "1,2,4").split(",")]
    target_input = int(os.environ.get("CONC_INPUT", "60000"))
    max_tokens = int(os.environ.get("CONC_OUT", "1024"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = f"/home/max/llm-stack/logs/bench-coding-conc-{label}-{ts}.json"

    print(f"=== bench-coding-conc ===  model={model} input~{target_input} "
          f"out={max_tokens} levels={levels}")
    rows = []
    for n in levels:
        bucket = bucket_for(target_input, max_tokens)
        prompts = [real.make_prompt(bucket) for _ in range(n)]
        print(f"[c={n}] firing {n} unique ~{target_input}-tok prompts ...")
        wall0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [
                pool.submit(deep.chat_streaming, model,
                            [{"role": "user", "content": p}], max_tokens, 1800)
                for p in prompts
            ]
            results = []
            for f in concurrent.futures.as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append({"error": str(e)[:200]})
        wall = time.perf_counter() - wall0
        ok = [r for r in results if not r.get("error")]
        total_out = sum(r.get("completion_tokens") or 0 for r in ok)
        row = {
            "conc": n,
            "ok": len(ok),
            "errors": len(results) - len(ok),
            "wall_s": round(wall, 1),
            "ttft_ms_mean": round(statistics.mean(r["ttft_ms"] for r in ok), 0) if ok else None,
            "ttft_ms_max": round(max(r["ttft_ms"] for r in ok), 0) if ok else None,
            # Thinking models: first VISIBLE token, and how much of the stream
            # was reasoning. ttft_ms above is first-token-of-any-kind.
            "first_content_ms_mean": round(statistics.mean(
                r["first_content_ms"] for r in ok
                if r.get("first_content_ms") is not None), 0)
                if any(r.get("first_content_ms") is not None for r in ok) else None,
            "think_chunks_mean": round(statistics.mean(
                r.get("think_chunks", 0) for r in ok), 0) if ok else None,
            "decode_per_req_mean": round(statistics.mean(r["decode_tok_s"] for r in ok), 1) if ok else None,
            "decode_agg": round(sum(r["decode_tok_s"] for r in ok), 1) if ok else None,
            "out_tokens_total": total_out,
        }
        rows.append(row)
        print("  " + json.dumps(row))
        time.sleep(10)  # let engine metrics/scheduler settle between levels

    with open(out_path, "w") as f:
        json.dump({"model": model, "label": label, "target_input": target_input,
                   "max_tokens": max_tokens, "rows": rows}, f, indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
