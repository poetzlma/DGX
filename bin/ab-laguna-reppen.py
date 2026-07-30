#!/usr/bin/env python3
"""A/B repetition_penalty as a looping mitigation for Laguna S-2.1, thinking ON.

Context (2026-07-30): laguna still loops upstream — HF NVFP4 discussion #16
("Reproducible Laguna S 2.1 NVFP4 runaway reasoning", open, no poolside reply)
reproduces it with DFlash DISABLED, so it is not a drafter or spec-decode bug.
In that thread a temperature sweep (0.0/0.3/0.7/1.0) failed 3/3 at every value
and output budgets 2k->16k all exhausted 3/3, while the two things that helped
were enable_thinking=false and repetition_penalty=1.15.

We keep thinking ON (Laguna profits from it), so repetition_penalty is the only
mitigation left that does not cost us the reasoning channel. This measures it
before anything gets baked into --override-generation-config.

Reuses the repro prompt classes and n-gram loop scoring from smoke-laguna-v2.py
so results are comparable. Real prompts on purpose: spec-decode must never be
benched with random text.

Usage: ab-laguna-reppen.py [--base http://127.0.0.1:8080/v1] [--model laguna-s-2.1]
                          [--penalties 1.0,1.15] [--reps 2]
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter

# The two classes that actually run away (#16 = reasoning-channel overrun,
# #6/#10 = long creative gen). Budgets match smoke-laguna-v2.py.
PROMPTS = [
    ("story-5k", "Write a complete ~4000-word science fiction short story about a "
     "lighthouse keeper on Europa who discovers a signal in the ice. Full prose, "
     "beginning to end, no outline.", 8192),
    ("arith-verify", "Verify step by step whether 7919 * 6841 = 54173679, showing "
     "your working, then state clearly TRUE or FALSE.", 4096),
    # #16's own repro: a single-turn concurrency debugging task.
    ("lru-debug", "This LRU cache is losing entries under concurrent access and "
     "occasionally returns stale values. Find the root cause and fix it:\n\n"
     "class LRUCache:\n"
     "    def __init__(self, cap):\n"
     "        self.cap = cap\n"
     "        self.d = {}\n"
     "        self.order = []\n"
     "        self.lock = threading.Lock()\n\n"
     "    def get(self, k):\n"
     "        if k in self.d:\n"
     "            self.order.remove(k)\n"
     "            self.order.append(k)\n"
     "            return self.d[k]\n"
     "        return None\n\n"
     "    def put(self, k, v):\n"
     "        with self.lock:\n"
     "            if k in self.d:\n"
     "                self.order.remove(k)\n"
     "            elif len(self.d) >= self.cap:\n"
     "                del self.d[self.order.pop(0)]\n"
     "            self.d[k] = v\n"
     "            self.order.append(k)\n", 4096),
]


def ngram_loop_score(text, n=12):
    """Max repeat count of any n-word window — >3 means degenerate looping."""
    words = text.split()
    if len(words) < n * 2:
        return 0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n))
    return max(grams.values()) if grams else 0


def reconsider_pileup(text):
    """Count hedging restarts — the tell for reasoning-channel runaway."""
    pats = [r"let me reconsider", r"wait,? (?:but|no|actually)", r"let me re-?check",
            r"hmm,? (?:but|wait|actually)", r"let me try again", r"actually,? wait"]
    return sum(len(re.findall(p, text, re.I)) for p in pats)


def run(base, model, prompt, max_tok, rep_pen):
    body = {
        "model": model, "max_tokens": max_tok, "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    # 1.0 is the no-op identity — send nothing so we exercise the real default path.
    if rep_pen != 1.0:
        body["repetition_penalty"] = rep_pen
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
    except Exception as e:  # noqa: BLE001 - report and keep the sweep going
        return {"error": f"{type(e).__name__}: {e}", "elapsed_s": time.time() - t0}
    dt = time.time() - t0
    ch = resp["choices"][0]
    msg = ch.get("message", {})
    content = msg.get("content") or ""
    # Laguna + --reasoning-parser poolside_v1 on vLLM 0.25.1 emits the thinking
    # block as "reasoning", NOT the "reasoning_content" you would expect (true
    # for streaming deltas too — verified 2026-07-30). Reading only the latter
    # makes the whole reasoning channel invisible: every think= column reads 0c
    # and reasoning-side loop scores silently score the empty string. Check both
    # so this keeps working if upstream renames it back.
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    usage = resp.get("usage", {})
    out_tok = usage.get("completion_tokens") or 0
    return {
        "finish_reason": ch.get("finish_reason"),
        "completion_tokens": out_tok,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "loop_score_content": ngram_loop_score(content),
        "loop_score_reasoning": ngram_loop_score(reasoning),
        "reconsider_reasoning": reconsider_pileup(reasoning),
        "elapsed_s": round(dt, 1),
        "decode_tok_s": round(out_tok / dt, 1) if dt > 0 else None,
        # length + nothing in the visible channel = ran away inside <think>
        "runaway": ch.get("finish_reason") == "length" and len(content) < 200,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="laguna-s-2.1")
    ap.add_argument("--penalties", default="1.0,1.15")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    pens = [float(x) for x in a.penalties.split(",")]

    print(f"=== ab-laguna-reppen  model={a.model}  penalties={pens}  reps={a.reps}")
    print("    thinking stays ON (server default); only repetition_penalty varies\n")
    out = {"model": a.model, "penalties": pens, "reps": a.reps, "runs": []}
    for name, prompt, max_tok in PROMPTS:
        for pen in pens:
            for i in range(a.reps):
                r = run(a.base, a.model, prompt, max_tok, pen)
                r.update(prompt=name, rep_pen=pen, rep=i)
                out["runs"].append(r)
                if "error" in r:
                    print(f"  {name:13} rp={pen:<5} #{i}  ERROR {r['error']}")
                    continue
                flag = "  <<< RUNAWAY" if r["runaway"] else ""
                loop = max(r["loop_score_content"], r["loop_score_reasoning"])
                if loop > 3 and not flag:
                    flag = "  <<< LOOP"
                print(f"  {name:13} rp={pen:<5} #{i}  fin={r['finish_reason']:6} "
                      f"out={r['completion_tokens']:5}  think={r['reasoning_chars']:6}c "
                      f"vis={r['content_chars']:5}c  loop={loop:2}  "
                      f"recon={r['reconsider_reasoning']:2}  {r['decode_tok_s']}t/s{flag}")

    print("\n=== verdict by penalty ===")
    for pen in pens:
        rs = [r for r in out["runs"] if r.get("rep_pen") == pen and "error" not in r]
        if not rs:
            continue
        n = len(rs)
        run_away = sum(r["runaway"] for r in rs)
        looped = sum(max(r["loop_score_content"], r["loop_score_reasoning"]) > 3 for r in rs)
        hit_len = sum(r["finish_reason"] == "length" for r in rs)
        tps = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
        print(f"  rp={pen:<5} runaway {run_away}/{n}  looped {looped}/{n}  "
              f"hit-length {hit_len}/{n}  mean {sum(tps)/len(tps):.1f} t/s")

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = f"/home/max/llm-stack/logs/ab-reppen-{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    sys.exit(main())
