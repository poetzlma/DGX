#!/usr/bin/env python3
"""Context-length sweep: TTFT (prefill rate), decode rate, KV usage at each size.

Each prompt has a unique nonce so prefix cache doesn't mask prefill cost.
"""
import json
import sys
import time
import urllib.request
sys.path.insert(0, "/home/max/llm-stack/bin")
from importlib import import_module
m = import_module("bench-deep")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nemotron-3-nano-omni"
# (target_input_tokens, filler_repeats) — repeats tuned to ~360 tok per filler block
PLAN = [
    (1_000,   3),
    (4_000,  11),
    (16_000, 45),
    (64_000, 178),
    (100_000, 278),
]
OUT_TOKENS = 256

def kv_pct():
    try:
        with urllib.request.urlopen("http://192.168.1.12:9012/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None

print(f"Context sweep: {MODEL}  output={OUT_TOKENS} tok/req")
print(f"{'tgt_ctx':>8} {'in_tok':>7} {'TTFT_s':>7} {'prefill_t/s':>11} {'decode_t/s':>10} {'e2e_t/s':>8} {'KV%_post':>9}")
print("-" * 75)

results = []
for target, reps in PLAN:
    nonce = f"[NONCE-{time.time_ns()}] "
    prompt = nonce + (m._FILLER * reps) + "\nSummarize the above in 3 short bullets."
    msgs = [{"role": "user", "content": prompt}]
    try:
        r = m.chat_streaming(MODEL, msgs, OUT_TOKENS, timeout=600)
    except Exception as e:
        print(f"{target:>8} ERROR: {str(e)[:60]}")
        results.append({"target": target, "error": str(e)[:200]})
        continue

    # Read prompt_tokens via a non-streaming call (cheap, prefix-cached now)
    body = {
        "model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0.0,
    }
    try:
        data = m.post_json("/v1/chat/completions", body, timeout=120)
        in_tok = data.get("usage", {}).get("prompt_tokens", 0)
    except Exception:
        in_tok = 0

    ttft_s = r["ttft_ms"] / 1000
    prefill_tps = in_tok / ttft_s if ttft_s > 0 else 0
    kvp = kv_pct()

    row = {
        "target": target, "in_tok": in_tok,
        "ttft_s": round(ttft_s, 2),
        "prefill_tok_s": round(prefill_tps, 1),
        "decode_tok_s": r["decode_tok_s"],
        "e2e_tok_s": r["e2e_tok_s"],
        "completion_tokens": r["completion_tokens"],
        "kv_pct_post": round(kvp * 100, 2) if kvp is not None else None,
    }
    results.append(row)
    print(f"{target:>8} {in_tok:>7} {ttft_s:>7.2f} {prefill_tps:>11.0f} "
          f"{r['decode_tok_s']:>10.1f} {r['e2e_tok_s']:>8.1f} "
          f"{(kvp*100 if kvp is not None else 0):>9.2f}")

with open("/home/max/llm-stack/logs/bench-context-sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: /home/max/llm-stack/logs/bench-context-sweep.json")
