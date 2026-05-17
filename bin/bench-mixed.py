#!/home/max/llm-stack/venv/bin/python3
"""Mixed-engine parallel bench.

Fires concurrent requests to two endpoints simultaneously and measures
per-engine decode tok/s without interference.

Usage: bench-mixed.py
"""
import concurrent.futures
import json
import sys
import time
import urllib.request

# Targets: (model_name, url, concurrency)
TARGETS = [
    ("qwen3.6-27b-int4-dflash", "http://127.0.0.1:9018/v1/chat/completions", 1),
    ("qwen3.6-35b-a3b-nvfp4",   "http://127.0.0.1:9019/v1/chat/completions", 2),
]
MAX_TOKS = 1024
PROMPT = (
    "Write a complete production-grade Python implementation of a thread-safe "
    f"LRU cache with TTL support. Aim for ~{MAX_TOKS} tokens of output. "
    "Include docstrings and at least 5 unit tests."
)


def fire(model: str, url: str, nonce: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"[NONCE-{nonce}] {PROMPT}"}],
        "max_tokens": MAX_TOKS,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            u = obj.get("usage")
            if u:
                completion_tokens = u.get("completion_tokens", completion_tokens)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            chunk = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
                or ""
            )
            if chunk and ttft is None:
                ttft = time.perf_counter() - t0
            if chunk:
                n_chunks += 1
    t1 = time.perf_counter()
    total_s = t1 - t0
    decode_s = max(0.001, total_s - (ttft or 0))
    out = completion_tokens or n_chunks
    return {
        "model": model,
        "ttft_s": round(ttft or 0, 3),
        "total_s": round(total_s, 3),
        "decode_s": round(decode_s, 3),
        "completion_tokens": out,
        "decode_tok_s": round(out / decode_s, 2),
    }


def main():
    print("# mixed-engine parallel bench")
    print(f"# targets: {[(m, c) for m, _, c in TARGETS]}")
    print(f"# max_tokens: {MAX_TOKS}")

    futures = []
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        for model, url, c in TARGETS:
            for i in range(c):
                futures.append(ex.submit(fire, model, url, time.time_ns() + i))
        results = []
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            print(json.dumps(r), flush=True)
    wall = time.perf_counter() - t_start

    # Per-model aggregates
    print("\n# Summary:")
    by_model = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        total_out = sum(r["completion_tokens"] for r in rs)
        mean_decode = sum(r["decode_tok_s"] for r in rs) / len(rs)
        # aggregate using wall time of slowest in this group
        max_total = max(r["total_s"] for r in rs)
        min_decode_start = min(r["total_s"] - r["decode_s"] for r in rs)
        window = max_total - min_decode_start
        agg = total_out / window if window > 0 else 0
        print(json.dumps({
            "model": model,
            "concurrency": len(rs),
            "total_completion_tokens": total_out,
            "mean_per_req_decode_tok_s": round(mean_decode, 2),
            "agg_decode_tok_s": round(agg, 2),
            "mean_ttft_s": round(sum(r["ttft_s"] for r in rs) / len(rs), 3),
        }, indent=2))
    print(f"\n# wall: {wall:.2f}s")

if __name__ == "__main__":
    main()
