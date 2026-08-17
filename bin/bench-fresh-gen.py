#!/usr/bin/env python3
"""Replicate MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark bench.sh methodology.

Their claim (single stream, 400 tok, MTP): 16.9 tok/s thinking / 21.0 non-thinking.

Methodology notes:
- decode tok/s = completion_tokens / (last_token_wall - first_token_wall).
  completion_tokens INCLUDES reasoning tokens; first token = first delta
  carrying EITHER `reasoning` or `content`. Getting this wrong is the exact
  harness bug called out in decisions.md §42 (it inflates tok/s on thinking
  models by collapsing the decode window).
- unique prefix per run so prefill is never a prefix-cache replay.
- 3 runs per mode, median reported; idle IDLE_S between runs because GB10
  decode swings on allocator state alone (AGENTS.md hard rule).
"""
import json, time, statistics, urllib.request, random, sys

import os
BASE  = os.environ.get("BENCH_BASE", "http://127.0.0.1:8080")
URL   = BASE.rstrip("/") + "/v1/chat/completions"
MODEL = os.environ.get("BENCH_MODEL", "qwen3.8-27b")
IDLE_OVERRIDE = os.environ.get("BENCH_IDLE")
MAX_TOK = 400
RUNS = 3
IDLE_S = 150  # overridden below by BENCH_IDLE if set

if IDLE_OVERRIDE:
    IDLE_S = int(IDLE_OVERRIDE)

# BENCH_PROMPT=prose|code. Speculative-decode acceptance is highly sensitive to
# output predictability, so the prompt is part of the measurement, not decoration.
# `code` matches the 0xBakeer "fresh generation" shape (write a Python module
# from a one-line spec); `prose` is deliberately harder to draft.
PROMPTS = {
    "prose": ("Write a concise technical explanation of how a write-ahead log "
              "guarantees durability in a database. Aim for about 300 words."),
    "code":  ("Write a complete Python module implementing an LRU cache with a "
              "capacity limit, get/put methods, and type hints. Include "
              "docstrings and a small __main__ demo. Output only code."),
}
PROMPT = PROMPTS[os.environ.get("BENCH_PROMPT", "prose")]


def run(thinking: bool):
    tag = random.randint(10**11, 10**12)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"[req {tag}] {PROMPT}"}],
        "max_tokens": MAX_TOK,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    last = None
    usage = None
    n_deltas = 0
    for raw in urllib.request.urlopen(req, timeout=600):
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            ch = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            # count a token-bearing delta from EITHER field
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now
                last = now
                n_deltas += 1

    ttft = first - t0
    window = last - first
    ct = usage["completion_tokens"] if usage else n_deltas
    return {"ttft": ttft, "window": window, "ct": ct,
            "tok_s": ct / window if window > 0 else 0.0,
            "deltas": n_deltas}


for mode, thinking in (("thinking", True), ("non-thinking", False)):
    rows = []
    for i in range(RUNS):
        r = run(thinking)
        rows.append(r)
        print(f"  {mode:12} run{i+1}: {r['tok_s']:6.2f} tok/s  "
              f"({r['ct']} tok in {r['window']:.2f}s, ttft {r['ttft']:.2f}s, "
              f"{r['deltas']} deltas)", flush=True)
        if i < RUNS - 1:
            time.sleep(IDLE_S)
    med = statistics.median(x["tok_s"] for x in rows)
    runs_str = ", ".join(f"{x['tok_s']:.2f}" for x in rows)
    print(f"==> {mode:12} MEDIAN {med:.2f} tok/s (runs: {runs_str})", flush=True)
    print(flush=True)
    if mode == "thinking":
        time.sleep(IDLE_S)
