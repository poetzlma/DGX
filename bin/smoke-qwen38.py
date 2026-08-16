#!/usr/bin/env python3
"""Correctness smoke for the Qwen3.8-27B lane. Gate a promotion on this.

Staged 2026-08-14 before the weights were public, adapted from smoke-ds4-0731.py.
Speed is the easy half to measure; these are the things that can be silently
WRONG after a model swap even when the server answers 200 OK:

  1. CHAT TEMPLATE / ROLE BLEED — new generation, possibly a new template. If
     framing is wrong the model answers as the wrong role or leaks control
     tokens. Qwen3.6 needed a hand-picked template (etc/qwen3.6-chat-template-
     froggeric.jinja); check whether 3.8 needs the same treatment.
  2. THINKING EATS THE BUDGET — thinking is ON by default on this model. On
     Qwen3.6 at max_tokens 1024-1536 the <think> block consumed the entire
     budget and the user got an empty/truncated answer. This is the single most
     likely day-one complaint from real clients, so it is a HARD check here.
  3. reasoning_effort — 3.8 advertises {xhigh, medium, low}, which is NOT the
     same vocabulary as ds4 0731's {low, high, max}. If we wire the gateway
     assuming ds4's names, requests 400 or silently degrade.
  4. REASONING FIELD NAME — Laguna emitted `reasoning`, ds4 emits thinking
     inline in `content`, vLLM's qwen3 parser emits `reasoning_content`. Getting
     this wrong silently broke log-proxy ttft_s and every bench's think column
     once already. We report which key this lane actually uses.
  5. TOOL CALLS — served with --tool-call-parser qwen3_coder. A parser mismatch
     shows up as tool calls arriving as prose inside content, which agentic
     clients (opencode) will not recover from.
  6. LONG-CONTEXT RETRIEVAL — this lane's headline claim over ds4 is 262k vs
     131k. Verify retrieval actually works at depth before selling it.

Exit code is nonzero if any HARD check fails.

Usage: Q38_PORT=9030 python3 smoke-qwen38.py
"""
import json, os, re, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("Q38_PORT", "9030"))
MODEL = os.environ.get("Q38_MODEL", "qwen3.8-27b")
DEEP_TOKENS = int(os.environ.get("Q38_DEEP_TOKENS", "30000"))
BASE = f"http://127.0.0.1:{PORT}"
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|assistant|>",
            "</assistant>", "<tool_call>", "<|tool_call|>", "<think>", "</think>"]

results = []


def chat(messages, max_tokens=512, temperature=0, **extra):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
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
    # Which key carries thinking is exactly what we are trying to learn.
    think_key = ("reasoning_content" if m.get("reasoning_content")
                 else "reasoning" if m.get("reasoning") else None)
    return {
        "content": m.get("content") or "",
        "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
        "think_key": think_key,
        "tool_calls": m.get("tool_calls") or [],
        "finish": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage", {}),
        "elapsed": round(time.perf_counter() - t0, 1),
    }


def record(name, hard, ok, detail):
    results.append({"name": name, "hard": hard, "ok": ok, "detail": detail})
    print(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name}: {detail}", flush=True)


# ── 1. liveness ────────────────────────────────────────────────────────────
r = chat([{"role": "user", "content": "Reply with exactly the word: ready"}], max_tokens=2048)
if r.get("error") or r.get("http_error"):
    record("liveness", True, False, str(r)[:200])
    print("\nserver not answering — aborting smoke")
    sys.exit(1)
record("liveness", True, "ready" in r["content"].lower(),
       f"{r['content'].strip()[:60]!r} in {r['elapsed']}s")
record("thinking field name", False, r["think_key"] is not None,
       f"lane uses {r['think_key']!r} "
       f"(wire log-proxy and benches to THIS key)" if r["think_key"]
       else "no separate thinking field — thinking is inline in content")

# ── 2. no special-token leakage ────────────────────────────────────────────
r = chat([{"role": "user", "content": "Write two sentences about tail latency."}],
         max_tokens=2048)
leaked = [s for s in SPECIALS if s in r["content"]]
record("no special-token leak", True, not leaked,
       f"leaked {leaked}" if leaked else "clean")

# ── 3. multi-turn: no role bleed ───────────────────────────────────────────
r = chat([
    {"role": "user", "content": "My favourite number is 41. Acknowledge in three words."},
    {"role": "assistant", "content": "Noted, forty-one."},
    {"role": "user", "content": "What is my favourite number plus one? Digits only."},
], max_tokens=2048)
body = r["content"]
bled = bool(re.search(r"\b(user|assistant)\s*:", body, re.I)) or any(s in body for s in SPECIALS)
record("multi-turn no role bleed", True, not bled, f"{body.strip()[:80]!r}")
record("multi-turn recall (41+1=42)", True, "42" in body, f"{body.strip()[:80]!r}")

# ── 4. THINKING BUDGET — the Qwen3.6 regression, as a hard gate ────────────
# Thinking is on by default. If a modest budget yields an empty answer, real
# clients with max_tokens=1024 get blank replies and we must NOT promote.
budget_rows = []
for mt in (1024, 1536, 4096):
    rr = chat([{"role": "user", "content":
                "Write a Python one-liner that reverses a string. Just the code."}],
              max_tokens=mt)
    visible = rr["content"].strip()
    budget_rows.append((mt, len(visible), rr["finish"],
                        rr["usage"].get("completion_tokens", 0)))
    if mt == 1024:
        record("answer survives max_tokens=1024", True, bool(visible),
               f"visible={len(visible)} chars, finish={rr['finish']}, "
               f"completion_tokens={rr['usage'].get('completion_tokens')}"
               + ("" if visible else "  <-- thinking ate the whole budget"))
print("       budget sweep (max_tokens, visible_chars, finish, completion_tok):")
for row in budget_rows:
    print(f"         {row}")

# ── 5. deterministic arithmetic ────────────────────────────────────────────
r = chat([{"role": "user", "content":
           "Compute 1234 * 5678. Then compute 9999 - 4321. "
           "Answer as exactly two lines: 'A=<value>' and 'B=<value>'."}],
         max_tokens=4096)
txt = (r["content"] + " " + r["reasoning"]).replace(",", "")
record("arithmetic A (1234*5678=7006652)", False, "7006652" in txt,
       "found" if "7006652" in txt else f"missing — {r['content'][:100]!r}")
record("arithmetic B (9999-4321=5678)", False, "5678" in txt, "found" if "5678" in txt else "missing")

# ── 6. generated code actually executes ────────────────────────────────────
r = chat([{"role": "user", "content":
           "Write a Python function `is_balanced(s)` returning True iff brackets "
           "()[]{} in s are balanced and correctly nested. Output ONLY the function "
           "in a ```python code block, no explanation."}], max_tokens=4096)
m = re.search(r"```(?:python)?\n(.*?)```", r["content"], re.S)
exec_ok, exec_detail = False, "no code block found"
if m:
    ns = {}
    try:
        exec(m.group(1), ns)
        f = ns["is_balanced"]
        cases = [("()", True), ("([{}])", True), ("(]", False), ("((", False),
                 ("", True), ("{[()()]}", True), (")(", False)]
        bad = [c for c, want in cases if bool(f(c)) != want]
        exec_ok = not bad
        exec_detail = "all 7 cases pass" if exec_ok else f"wrong on {bad}"
    except Exception as e:
        exec_detail = f"exec failed: {str(e)[:120]}"
record("generated code is correct", True, exec_ok, exec_detail)

# ── 7. tool calling through the qwen3_coder parser ─────────────────────────
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current temperature for a city.",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
r = chat([{"role": "user", "content": "What is the temperature in Ramsau right now?"}],
         max_tokens=4096, tools=TOOLS, tool_choice="auto")
if r.get("http_error"):
    record("tool call emitted", True, False, f"HTTP {r['http_error']}: {r['detail'][:120]}")
else:
    tc = r["tool_calls"]
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather"
    args_ok = False
    if ok:
        try:
            args_ok = "ramsau" in json.loads(tc[0]["function"]["arguments"]).get("city", "").lower()
        except Exception:
            args_ok = False
    record("tool call emitted", True, ok,
           f"{len(tc)} call(s)" if tc else
           f"NO tool_calls — parser mismatch? content={r['content'][:120]!r}")
    record("tool call arguments parse", True, args_ok,
           tc[0]["function"]["arguments"][:120] if ok else "n/a")

# ── 8. reasoning_effort vocabulary (xhigh/medium/low, NOT ds4's low/high/max) ─
lens = {}
for eff in ("low", "medium", "xhigh"):
    rr = chat([{"role": "user", "content":
                "How many distinct ways can you make 37 cents from US coins? "
                "Think it through, then give the number."}],
              max_tokens=8192, reasoning_effort=eff)
    lens[eff] = f"HTTP {rr['http_error']}" if rr.get("http_error") else \
        rr["usage"].get("completion_tokens", 0)
ints = [v for v in lens.values() if isinstance(v, int)]
record("reasoning_effort accepted", False, len(ints) == 3, str(lens))
record("reasoning_effort changes budget", False, len(set(ints)) > 1,
       "budget responds" if len(set(ints)) > 1 else "identical lengths — likely ignored")

# ── 9. long-context needle ─────────────────────────────────────────────────
filler_src = ("The build system compiles each translation unit independently and "
              "then links the resulting objects into a single static binary. ")
NEEDLE = "The deployment passphrase for the Ramsau cluster is quartz-heron-8814."
filler = filler_src * (int(DEEP_TOKENS * 3.6) // len(filler_src))
half = len(filler) // 2
haystack = filler[:half] + "\n\n" + NEEDLE + "\n\n" + filler[half:]
r = chat([{"role": "user", "content": haystack +
           "\n\nQuestion: what is the deployment passphrase for the Ramsau "
           "cluster? Answer with just the passphrase."}], max_tokens=2048)
found = "quartz-heron-8814" in (r["content"] + r["reasoning"])
record(f"needle @~{DEEP_TOKENS//1000}k ctx", True, found,
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
with open("/home/max/llm-stack/logs/smoke-qwen38.json", "w") as fh:
    json.dump(results, fh, indent=2)
sys.exit(1 if hard_fail else 0)
