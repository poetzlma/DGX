#!/usr/bin/env python3
"""Concurrency bench for an OpenAI-compatible lane. Written 2026-08-02.

WHY A SEPARATE HARNESS: bench-ds4-0731.py's concurrency block was written for
the ds4 engine, which serializes prefills, so it reports an "aggregate" over the
decode window only. For a lane that genuinely batches (llama.cpp with -np N),
the number an operator actually cares about is TOTAL COMPLETION TOKENS / WALL
CLOCK for the whole batch — that is what resale throughput is. Both are printed
here so the two engines can still be compared honestly.

Every stream gets a DISTINCT prompt with a unique token-0 prefix, so no stream
can win by hitting the prompt cache.

Usage: bench-concurrency.py [--port 9098] [--ctx-tokens 8192] [--gen 200]
                            [--levels 1,2,4,8]
"""
import argparse, json, random, statistics, threading, time, urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="9098")
ap.add_argument("--ctx-tokens", type=int, default=8192)
ap.add_argument("--gen", type=int, default=200)
ap.add_argument("--levels", default="1,2,4,8")
ap.add_argument("--model", default="deepseek-v4-flash")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

_chunks = []
for root in (Path("/home/max/ds4-upstream"), Path("/home/max/llama.cpp/src"), Path("/home/max/ds4-q4")):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in {".c", ".h", ".cu", ".cpp", ".py"}:
            try: _chunks.append(p.read_text(errors="ignore"))
            except Exception: pass
CORPUS = "".join(_chunks)


def make_prompt(tokens):
    need = int(tokens * 3.6)
    txt = CORPUS
    while len(txt) < need:
        txt = txt * 2
    # unique at token 0 so prefix caching cannot short-circuit prefill
    return f"// conc {random.getrandbits(48):012x}\n" + txt[:need] + "\n\nName one function above."


def stream_once(prompt, gen):
    body = json.dumps({
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); tf = None; n = 0; usage = {}
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try: d = json.loads(p)
                except Exception: continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    delta = ch.get("delta") or {}
                    # MUST read all three: with --jinja this is a thinking model,
                    # so a small max_tokens can be spent entirely inside <think>
                    # and `content` stays empty while tokens ARE being produced.
                    # Laguna emits `reasoning`; others `reasoning_content`.
                    piece = (delta.get("content") or delta.get("reasoning")
                             or delta.get("reasoning_content") or "")
                    if piece:
                        if tf is None: tf = time.perf_counter()
                        n += 1
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}
    te = time.perf_counter()
    if tf is None:
        return {"err": "no tokens"}
    return {
        "t0": t0, "tf": tf, "te": te,
        "ptok": usage.get("prompt_tokens", 0),
        "ctok": usage.get("completion_tokens", n),
        "ttft": tf - t0,
        "decode_tps": (n - 1) / (te - tf) if te > tf and n > 1 else 0.0,
    }


def run_level(c):
    prompts = [make_prompt(a.ctx_tokens) for _ in range(c)]
    res = {}
    def w(i): res[i] = stream_once(prompts[i], a.gen)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(c)]
    wall0 = time.perf_counter()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.perf_counter() - wall0
    ok = [r for r in res.values() if not r.get("err")]
    if not ok:
        errs = [r.get("err") for r in res.values()][:2]
        print(f"  c={c:2d}  ALL FAILED: {errs}")
        return
    ctok = sum(r["ctok"] for r in ok)
    ptok = sum(r["ptok"] for r in ok)
    # what an operator actually gets: completions per wall second for the batch
    print(f"  c={c:2d}  n_ok={len(ok)}/{c}  wall={wall:6.1f}s  "
          f"batch_tps={ctok/wall:6.2f}  per_stream_med={statistics.median(r['decode_tps'] for r in ok):5.2f}  "
          f"ttft_med={statistics.median(r['ttft'] for r in ok):6.1f}s  "
          f"ttft_max={max(r['ttft'] for r in ok):6.1f}s  prefill_agg={ptok/wall:7.1f}tok/s")


print(f"concurrency bench -> {BASE} | prompt ~{a.ctx_tokens} tok, gen {a.gen}, distinct+uncached per stream")
for lv in [int(x) for x in a.levels.split(",")]:
    run_level(lv)
    time.sleep(10)
