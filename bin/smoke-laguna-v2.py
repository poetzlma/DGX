#!/usr/bin/env python3
"""Smoke test for Laguna S-2.1 v2 (spinquantless) cutover — looping + speed check.

Runs the two repro classes from HF discussions #6/#10 (long creative gen,
arithmetic verification) plus a coding prompt against the llama-swap endpoint,
then reports decode tok/s (DFlash acceptance proxy vs ~33 tok/s @100k baseline)
and flags degenerate repetition (n-gram loops, "Let me reconsider" pileups).

Usage: smoke-laguna-v2.py [--base http://127.0.0.1:8080/v1] [--model laguna-s-2.1]
Real prompts on purpose — spec-decode must never be benched with random text.
"""
import argparse, json, re, sys, time, urllib.request
from collections import Counter

PROMPTS = [
    ("story-5k", "Write a complete ~4000-word science fiction short story about a "
     "lighthouse keeper on Europa who discovers a signal in the ice. Full prose, "
     "beginning to end, no outline.", 8192),
    ("arith-verify", "Verify step by step whether 7919 * 6841 = 54173679, showing "
     "your working, then state clearly TRUE or FALSE.", 4096),
    ("code-task", "Write a Python function that merges overlapping intervals, with "
     "type hints, docstring, and 5 pytest cases covering edge cases.", 4096),
]

def ngram_loop_score(text, n=12):
    """Max repeat count of any n-word window — >3 means degenerate looping."""
    words = text.split()
    if len(words) < n * 2:
        return 0
    grams = Counter(tuple(words[i:i+n]) for i in range(len(words) - n))
    return max(grams.values()) if grams else 0

def run(base, model, name, prompt, max_tok):
    body = json.dumps({
        "model": model, "max_tokens": max_tok, "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    ch = resp["choices"][0]
    content = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    usage = resp.get("usage", {})
    ctok = usage.get("completion_tokens", 0)
    full = reasoning + "\n" + content
    reconsider = len(re.findall(r"(?i)let me (re|think again|reconsider)", full))
    loops = ngram_loop_score(full)
    hit_cap = ch.get("finish_reason") == "length"
    print(f"[{name}] {ctok} tok in {dt:.0f}s = {ctok/dt:.1f} tok/s | "
          f"finish={ch.get('finish_reason')} | content_len={len(content)} | "
          f"reconsider-phrases={reconsider} | max-ngram-repeat={loops}"
          f"{'  <-- LOOPING?' if loops > 3 or reconsider > 8 else ''}"
          f"{'  <-- hit token cap, content may be null' if hit_cap and not content else ''}")
    return loops <= 3 and reconsider <= 8 and bool(content)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="laguna-s-2.1")
    args = ap.parse_args()
    ok = all([run(args.base, args.model, *p) for p in PROMPTS])
    print("SMOKE:", "PASS" if ok else "FAIL — inspect above; consider LAG_THINK=0 A/B or rollback")
    sys.exit(0 if ok else 1)
