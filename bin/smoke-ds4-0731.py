#!/usr/bin/env python3
"""Correctness smoke for the DeepSeek-V4-Flash-0731 weights swap.

Speed is the easy half. These are the things that can be silently WRONG after
a weights swap even when the server answers 200 OK:

  1. CHAT TEMPLATE — 0731 upstream ships NO Jinja template (a Python encoding/
     folder instead of tokenizer_config chat_template). If antirez's conversion
     embedded a stale/preview template, turns get framed wrong and the model
     rambles, answers as the wrong role, or leaks control tokens. Checked by
     looking for role bleed and stray special tokens in a multi-turn exchange.
  2. IMATRIX MISMATCH — the quant used the PREVIEW imatrix. Damage from that
     shows up as degraded factual/arithmetic reliability and broken code, not
     as a crash. Checked with deterministic tasks that have exact answers.
  3. reasoning_effort — new in 0731 (low/high/max). If ds4 passes it through,
     output length should move with it; if it 400s or is ignored, we want to
     know before wiring it at the gateway.
  4. LONG-CONTEXT COHERENCE — needle retrieval at depth, since this lane's
     whole reason to exist is long-context planning.

Exit code is nonzero if any HARD check fails, so this can gate a promotion.

Usage: DS4_PORT=9099 python3 smoke-ds4-0731.py
"""
import json, os, re, sys, time, urllib.request

PORT = int(os.environ.get("DS4_PORT", "9099"))
BASE = f"http://127.0.0.1:{PORT}"
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<｜begin▁of▁sentence｜>",
            "<｜User｜>", "<｜Assistant｜>", "<|assistant|>", "</assistant>"]

results = []


def chat(messages, max_tokens=512, temperature=0, **extra):
    body = {"model": "deepseek-v4-flash", "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    body.update(extra)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "detail": e.read().decode("utf-8", "ignore")[:300]}
    except Exception as e:
        return {"error": str(e)[:200]}
    m = d["choices"][0]["message"]
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
        "finish": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage", {}),
        "elapsed": round(time.perf_counter() - t0, 1),
    }


def record(name, hard, ok, detail):
    results.append({"name": name, "hard": hard, "ok": ok, "detail": detail})
    mark = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{mark}] {name}: {detail}", flush=True)


# ── 1. basic liveness + template sanity ────────────────────────────────────
r = chat([{"role": "user", "content": "Reply with exactly the word: ready"}], max_tokens=16)
if r.get("error") or r.get("http_error"):
    record("liveness", True, False, str(r)[:200])
    print("\nserver not answering — aborting smoke")
    sys.exit(1)
record("liveness", True, "ready" in r["content"].lower(),
       f"{r['content'].strip()[:60]!r} in {r['elapsed']}s")

# ── 2. no special-token leakage ────────────────────────────────────────────
r = chat([{"role": "user", "content": "Write two sentences about tail latency."}],
         max_tokens=200)
leaked = [s for s in SPECIALS if s in r["content"]]
record("no special-token leak", True, not leaked,
       f"leaked {leaked}" if leaked else "clean")

# ── 3. multi-turn: no role bleed (template framing) ────────────────────────
r = chat([
    {"role": "user", "content": "My favourite number is 41. Acknowledge in three words."},
    {"role": "assistant", "content": "Noted, forty-one."},
    {"role": "user", "content": "What is my favourite number plus one? Answer with the digits only."},
], max_tokens=64)
body = r["content"]
bled = bool(re.search(r"\b(user|assistant)\s*:", body, re.I)) or any(s in body for s in SPECIALS)
record("multi-turn no role bleed", True, not bled, f"{body.strip()[:80]!r}")
record("multi-turn recall (41+1=42)", True, "42" in body, f"{body.strip()[:80]!r}")

# ── 4. deterministic arithmetic (imatrix damage canary) ────────────────────
r = chat([{"role": "user", "content":
           "Compute 1234 * 5678. Then compute 9999 - 4321. "
           "Answer as exactly two lines: 'A=<value>' and 'B=<value>'."}],
         max_tokens=2048)
txt = r["content"] + " " + r["reasoning"]
a_ok = "7006652" in txt.replace(",", "")
b_ok = "5678" in txt.replace(",", "")
record("arithmetic A (1234*5678=7006652)", False, a_ok, "found" if a_ok else f"missing — {r['content'][:100]!r}")
record("arithmetic B (9999-4321=5678)", False, b_ok, "found" if b_ok else "missing")

# ── 5. code generation actually executes ───────────────────────────────────
r = chat([{"role": "user", "content":
           "Write a Python function `is_balanced(s)` returning True iff brackets "
           "()[]{} in s are balanced and correctly nested. Output ONLY the function "
           "in a ```python code block, no explanation."}], max_tokens=2048)
code = ""
m = re.search(r"```(?:python)?\n(.*?)```", r["content"], re.S)
if m:
    code = m.group(1)
exec_ok, exec_detail = False, "no code block found"
if code:
    ns = {}
    try:
        exec(code, ns)
        f = ns.get("is_balanced")
        cases = [("()", True), ("([{}])", True), ("(]", False), ("((", False), ("", True),
                 ("{[()()]}", True), (")(", False)]
        bad = [c for c, want in cases if bool(f(c)) != want]
        exec_ok = not bad
        exec_detail = "all 7 cases pass" if exec_ok else f"wrong on {bad}"
    except Exception as e:
        exec_detail = f"exec failed: {str(e)[:120]}"
record("generated code is correct", True, exec_ok, exec_detail)

# ── 6. reasoning_effort passthrough (new in 0731) ──────────────────────────
lens = {}
for eff in ("low", "high"):
    r = chat([{"role": "user", "content":
               "How many distinct ways can you make 37 cents from US coins? "
               "Think it through, then give the number."}],
             max_tokens=1500, reasoning_effort=eff)
    if r.get("http_error"):
        lens[eff] = f"HTTP {r['http_error']}"
    else:
        lens[eff] = r["usage"].get("completion_tokens", 0)
if all(isinstance(v, int) for v in lens.values()):
    moved = lens["high"] != lens["low"]
    record("reasoning_effort accepted", False, True, f"low={lens['low']} tok, high={lens['high']} tok")
    record("reasoning_effort changes output", False, moved,
           "budget responds" if moved else "identical length — likely ignored by ds4")
else:
    record("reasoning_effort accepted", False, False, f"{lens} — not supported by this build")

# ── 7. long-context needle at depth ────────────────────────────────────────
filler_src = ("The build system compiles each translation unit independently and "
              "then links the resulting objects into a single static binary. ")
NEEDLE = "The deployment passphrase for the Ramsau cluster is quartz-heron-8814."
approx_tokens = 30000
filler = filler_src * (int(approx_tokens * 3.6) // len(filler_src))
half = len(filler) // 2
haystack = filler[:half] + "\n\n" + NEEDLE + "\n\n" + filler[half:]
r = chat([{"role": "user", "content":
           haystack + "\n\nQuestion: what is the deployment passphrase for the "
           "Ramsau cluster? Answer with just the passphrase."}], max_tokens=256)
found = "quartz-heron-8814" in (r["content"] + r["reasoning"])
record("needle @~30k ctx", True, found,
       f"prompt {r['usage'].get('prompt_tokens')} tok, {r['elapsed']}s, "
       f"got {r['content'].strip()[:60]!r}")

# ── summary ────────────────────────────────────────────────────────────────
hard_fail = [x for x in results if x["hard"] and not x["ok"]]
soft_fail = [x for x in results if not x["hard"] and not x["ok"]]
print("\n" + "=" * 70)
print(f"{len([x for x in results if x['ok']])}/{len(results)} checks passed | "
      f"{len(hard_fail)} hard failures, {len(soft_fail)} warnings")
if hard_fail:
    print("HARD FAILURES: " + ", ".join(x["name"] for x in hard_fail))
print("=" * 70)
with open("/home/max/llm-stack/logs/smoke-ds4-0731.json", "w") as fh:
    json.dump(results, fh, indent=2)
sys.exit(1 if hard_fail else 0)
