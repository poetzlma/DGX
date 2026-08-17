#!/usr/bin/env python3
"""Concurrency bench, same generation shape as bench-fresh-gen.py.

Reports BOTH numbers, because they answer different questions:
  aggregate tok/s  = total completion tokens / wall time of the whole batch
                     -> throughput. What a fleet operator cares about.
  per-request tok/s = median of each stream's own decode rate
                     -> latency. What one interactive user feels.

A config can win one and lose the other; DSpark draft depth is exactly such a
knob (0xBakeer §3: "k=14 is not simply better" once concurrency rises).

Env: BENCH_BASE, BENCH_MODEL, BENCH_C (default 4), BENCH_PROMPT=prose|code,
     BENCH_THINK=1|0, BENCH_RUNS (default 2), BENCH_IDLE (default 60)
"""
import json, os, time, random, statistics, urllib.request
import concurrent.futures as cf

BASE  = os.environ.get("BENCH_BASE", "http://127.0.0.1:8080")
URL   = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-27b")
C     = int(os.environ.get("BENCH_C", "4"))
RUNS  = int(os.environ.get("BENCH_RUNS", "2"))
IDLE  = int(os.environ.get("BENCH_IDLE", "60"))
THINK = os.environ.get("BENCH_THINK", "1") == "1"
MAX_TOK = 400

PROMPTS = {
    "prose": ("Write a concise technical explanation of how a write-ahead log "
              "guarantees durability in a database. Aim for about 300 words."),
    "code":  ("Write a complete Python module implementing an LRU cache with a "
              "capacity limit, get/put methods, and type hints. Include "
              "docstrings and a small __main__ demo. Output only code."),
}
PROMPT = PROMPTS[os.environ.get("BENCH_PROMPT", "code")]


def one(idx):
    # unique prefix per stream AND per run: no prefix-cache sharing between
    # concurrent streams, which would otherwise inflate the aggregate.
    tag = f"{random.randint(10**11,10**12)}-{idx}"
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": f"[req {tag}] {PROMPT}"}],
            "max_tokens": MAX_TOK, "stream": True,
            "stream_options": {"include_usage": True}}
    if not THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    first = last = None
    usage = None
    for raw in urllib.request.urlopen(req, timeout=900):
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        p = line[6:]
        if p == "[DONE]":
            break
        try:
            ch = json.loads(p)
        except json.JSONDecodeError:
            continue
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now
                last = now
    ct = usage["completion_tokens"] if usage else 0
    return {"ct": ct, "first": first, "last": last,
            "tok_s": ct / (last - first) if last and first and last > first else 0.0}


print(f"model={MODEL} c={C} prompt={os.environ.get('BENCH_PROMPT','code')} "
      f"thinking={THINK} runs={RUNS}")
aggs, pers = [], []
for r in range(RUNS):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=C) as ex:
        res = list(ex.map(one, range(C)))
    wall = time.time() - t0
    total = sum(x["ct"] for x in res)
    # aggregate measured over the batch's own decode span, not including the
    # initial connect, so it is comparable to the c=1 decode figure.
    span = max(x["last"] for x in res) - min(x["first"] for x in res)
    agg = total / span if span > 0 else 0.0
    per = statistics.median(x["tok_s"] for x in res)
    aggs.append(agg); pers.append(per)
    print(f"  run{r+1}: aggregate {agg:6.2f} tok/s | per-request median "
          f"{per:5.2f} tok/s | {total} tok, wall {wall:.1f}s", flush=True)
    if r < RUNS - 1:
        time.sleep(IDLE)

print(f"==> c={C} AGGREGATE median {statistics.median(aggs):.2f} tok/s")
print(f"==> c={C} PER-REQUEST median {statistics.median(pers):.2f} tok/s")
