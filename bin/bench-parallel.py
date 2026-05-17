#!/home/max/llm-stack/venv/bin/python3
"""Parallel-streams decode-throughput bench.

Fires N concurrent requests via SSE streaming and measures:
  - per-request decode tok/s (from completion_tokens / (total_s - ttft))
  - aggregate decode tok/s (sum completion_tokens across reqs / parallel-window decode time)
  - TTFT per request

Counts both `delta.reasoning` and `delta.content` chunks (vLLM emits one or the
other when --reasoning-parser qwen3 is active). Uses stream_options.include_usage
so completion_tokens is authoritative.

Usage: bench-parallel.py MODEL CONCURRENCY [PROMPT_TOKENS] [MAX_TOKENS]
"""
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

MODEL = sys.argv[1]
CONCURRENCY = int(sys.argv[2])
PROMPT_TOKS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MAX_TOKS = int(sys.argv[4]) if len(sys.argv) > 4 else 1024

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8080/v1/chat/completions")
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer dummy"}

# Build a prompt approximately PROMPT_TOKS long via filler repeats.
_BLOCK = (
    "The quick brown fox jumps over the lazy dog while a bright sun rises "
    "over the mountains, casting long shadows across the green valley below. "
    "Birds sing in the morning air and a gentle breeze rustles the leaves. "
    "Children play in the meadow, their laughter echoing off the rocks. "
)  # ~60 tokens
REPEATS = max(1, PROMPT_TOKS // 60)


def make_body(nonce: int) -> bytes:
    prompt = (
        f"[NONCE-{nonce}] " + (_BLOCK * REPEATS)
        + "\n\nWrite a complete production-grade Python implementation of a "
        f"thread-safe LRU cache with TTL support. Aim for ~{MAX_TOKS} tokens "
        "of output. Include docstrings and at least 5 unit tests."
    )
    return json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKS,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()


def one_run(idx: int) -> dict:
    body = make_body(time.time_ns() + idx)
    req = urllib.request.Request(URL, data=body, headers=HEADERS, method="POST")
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
        "idx": idx,
        "ttft_s": round(ttft or 0, 3),
        "total_s": round(total_s, 3),
        "decode_s": round(decode_s, 3),
        "completion_tokens": out,
        "decode_tok_s": round(out / decode_s, 2),
    }


def main():
    print(f"# bench {MODEL}  c={CONCURRENCY}  prompt~{PROMPT_TOKS}  max={MAX_TOKS}",
          flush=True)
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(one_run, i) for i in range(CONCURRENCY)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    wall = time.perf_counter() - t_start
    results.sort(key=lambda r: r["idx"])
    for r in results:
        print(json.dumps(r), flush=True)

    total_out = sum(r["completion_tokens"] for r in results)
    decode_starts = [r["total_s"] - r["decode_s"] for r in results]
    decode_ends = [r["total_s"] for r in results]
    # decode window: from first req's first content to last req's end
    window_s = wall - min(decode_starts) if decode_starts else wall
    agg_decode_tok_s = total_out / window_s if window_s > 0 else 0
    per_req = [r["decode_tok_s"] for r in results]
    mean_ttft = sum(r["ttft_s"] for r in results) / len(results)

    summary = {
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "wall_s": round(wall, 2),
        "agg_decode_tok_s": round(agg_decode_tok_s, 2),
        "mean_per_req_decode_tok_s": round(sum(per_req) / len(per_req), 2),
        "mean_ttft_s": round(mean_ttft, 3),
        "total_completion_tokens": total_out,
    }
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
