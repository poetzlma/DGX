#!/usr/bin/env python3
"""Does the lane COMMIT TO AN ACTION, or plan forever?

Written 2026-08-01 after laguna-s-2.1 was pulled from prod for generating
~2000 plans for one task and never acting (HF Laguna-S-2.1 discussion #16,
"runaway reasoning", open, no vendor reply). Ordinary benchmarks do not catch
this: the model is fluent, on-topic and confidently wrong about being finished.

What this measures, per task:
  committed      did a concrete action appear (tool call / final answer)?
  plan_ratio     share of output spent deliberating vs. acting
  restarts       count of "let me reconsider / actually / wait / alternatively"
                 pivots — the tell for plan-loop rather than plan-then-act
  finish_reason  'length' means it ran out of budget still thinking = the
                 laguna failure signature

Deliberately given a tight max_tokens: a model that only commits when handed an
unlimited budget has not solved the problem, it has hidden it.

Usage: MODEL=deepseek-v4-flash-0731 python3 probe-commits-to-action.py
       BASE=http://192.168.1.7:4000 KEY=... python3 probe-commits-to-action.py
"""
import json, os, re, sys, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8079")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-0731")
KEY = os.environ.get("KEY", "not-needed")
BUDGET = int(os.environ.get("BUDGET", "1200"))

TOOLS_SYS = (
    "You are a coding agent with these tools:\n"
    "  read_file(path)\n  write_file(path, content)\n  run(cmd)\n"
    "To act, emit EXACTLY one line: TOOL: <name>(<args>)\n"
    "Do not plan more than two sentences before your first TOOL line."
)

TASKS = [
    ("first-action",
     TOOLS_SYS,
     "The test suite fails with 'ModuleNotFoundError: no module named utils' "
     "in tests/test_api.py. Begin fixing it."),
    ("ambiguous-scope",
     TOOLS_SYS,
     "Make the codebase faster. Start."),          # bait for endless scoping
    ("decide-under-uncertainty",
     "You are a decisive engineer. Give a recommendation, not a survey.",
     "Our 100k-token prefill takes 6 minutes. We can (a) rewrite the dequant "
     "kernels, (b) add a prefix cache, or (c) buy a second box. Pick ONE and "
     "give the first concrete step."),
    ("terminate-early",
     "You are a coding agent.",
     "Print the numbers 1 to 5. Then stop and say DONE."),
]

PIVOT = re.compile(r"\b(wait|actually|let me reconsider|on second thought|"
                   r"alternatively|but hold on|hmm,? let me|rethink|"
                   r"let me re-?examine|scratch that)\b", re.I)
ACTION = re.compile(r"(TOOL:\s*\w+\s*\(|^\s*DONE\b|\bI (will|'ll) (now )?"
                    r"(use|call|run|read|write)\b)", re.I | re.M)


def ask(system, user):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": BUDGET, "temperature": 0.3, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    m = ch["message"]
    return {"text": (m.get("content") or "") + (m.get("reasoning") or m.get("reasoning_content") or ""),
            "finish": ch.get("finish_reason"),
            "tok": d.get("usage", {}).get("completion_tokens", 0),
            "s": round(time.perf_counter() - t0, 1)}


print(f"probe → {BASE} model={MODEL} budget={BUDGET} tok\n" + "=" * 74)
rows = []
for name, sys_p, user_p in TASKS:
    r = ask(sys_p, user_p)
    txt = r["text"]
    act = ACTION.search(txt)
    pivots = len(PIVOT.findall(txt))
    # everything before the first concrete action is deliberation
    plan_chars = act.start() if act else len(txt)
    ratio = plan_chars / max(len(txt), 1)
    verdict = "COMMITTED" if act and r["finish"] != "length" else (
        "RAN OUT THINKING" if r["finish"] == "length" else "NO ACTION")
    rows.append((name, verdict, pivots, ratio, r))
    print(f"\n--- {name}")
    print(f"  {verdict:17} finish={r['finish']:7} {r['tok']:5} tok  {r['s']:6}s  "
          f"pivots={pivots}  plan_ratio={ratio:.2f}")
    first = act.group(0).strip()[:60] if act else "(none)"
    print(f"  first action: {first!r}")
    print(f"  opens: {txt.strip()[:150]!r}")

print("\n" + "=" * 74)
ok = sum(1 for _, v, *_ in rows if v == "COMMITTED")
print(f"{ok}/{len(rows)} committed to an action within {BUDGET} tokens")
worst = max(rows, key=lambda x: x[2])
print(f"most plan-pivots: {worst[0]} ({worst[2]})")
if any(v == "RAN OUT THINKING" for _, v, *_ in rows):
    print("!! at least one task exhausted its budget still deliberating —")
    print("   that is the laguna failure signature; do not call this fixed.")
json.dump([{"task": n, "verdict": v, "pivots": p, "plan_ratio": round(rt, 3),
            "finish": r["finish"], "tokens": r["tok"]} for n, v, p, rt, r in rows],
          open("/home/max/llm-stack/logs/probe-commits-to-action.json", "w"), indent=2)
