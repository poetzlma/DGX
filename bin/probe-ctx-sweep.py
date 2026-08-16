#!/home/max/llm-stack/venv/bin/python3
"""Fast single-request context sweep: real prefill and decode tok/s per context.

USAGE:  probe-ctx-sweep.py PORT MODEL [LABEL]

WHY THIS EXISTS: the full concurrency matrix takes hours per configuration. This
answers "is config A faster than config B" in ~5 minutes, so you can pick the
engine/quant combination FIRST and spend the hours only on the winner.

WHY IT REPORTS prompt_tokens: never trust the requested context size. The
bench prompt generator OVERSHOOTS at large targets — target=120000 produces
141,674 real tokens (~18% over). Always report what the engine actually saw.

WHY IT COMPUTES PREFILL FROM WALL-CLOCK: vLLM's own `Avg prompt throughput`
line is unreliable on the AEON 0.23.0 build — it reported a constant
14,166.x tok/s in every interval while wall-clock showed ~800 tok/s at 141k.
prompt_tokens / measured_TTFT is the number that survives scrutiny.

Engine must be IDLE. A concurrent request invalidates every row.
"""
import http.client, json, sys, time
from importlib import import_module

sys.path.insert(0, "/home/max/llm-stack/bin")
real = import_module("bench-coding-realistic")

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9030
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
LABEL = sys.argv[3] if len(sys.argv) > 3 else MODEL
TARGETS = [int(x) for x in
           (sys.argv[4].split(",") if len(sys.argv) > 4 else
            ["2000", "16000", "60000", "120000", "220000"])]


def probe(target, out=128):
    bucket = {"name": "100k", "target_input": target, "max_tokens": out}
    prompt = real.make_prompt(bucket)
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": out, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft = None; n = 0; usage = {}; first_text = ""
    while True:
        line = resp.readline()
        if not line:
            break
        s = line.decode().strip()
        if not s.startswith("data: "):
            continue
        p = s[6:]
        if p == "[DONE]":
            break
        try:
            ch = json.loads(p)
        except json.JSONDecodeError:
            continue
        for c in ch.get("choices") or []:
            d = c.get("delta", {})
            tok = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
            if tok:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if len(first_text) < 80:
                    first_text += tok
                n += 1
        if ch.get("usage"):
            usage = ch["usage"]
    total = time.perf_counter() - t0
    conn.close()
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", n)
    dec_window = (total - ttft) if ttft else total
    return {"target": target, "prompt_tok": pt, "ttft_s": round(ttft or total, 1),
            "prefill_tok_s": round(pt / ttft, 0) if ttft else 0,
            "decode_tok_s": round(ct / dec_window, 1) if dec_window > 0.01 else 0,
            "total_s": round(total, 1), "sample": first_text[:60]}


def main():
    print(f"=== ctx sweep: {LABEL}  (:{PORT}, model={MODEL}) ===")
    print(f"{'target':>8} {'prompt_tok':>11} {'ttft_s':>8} {'prefill/s':>10} "
          f"{'decode/s':>9} {'total_s':>8}")
    rows = []
    for t in TARGETS:
        try:
            r = probe(t)
        except Exception as e:
            print(f"{t:>8}  FAILED: {str(e)[:70]}")
            continue
        rows.append(r)
        print(f"{r['target']:>8} {r['prompt_tok']:>11,} {r['ttft_s']:>8.1f} "
              f"{r['prefill_tok_s']:>10,.0f} {r['decode_tok_s']:>9.1f} {r['total_s']:>8.1f}")
        # Garbage canary: NVFP4 on a wrong kernel emits "!!!!" / "d d d" while
        # still reporting excellent throughput. Show what it actually said.
        print(f"         sample: {r['sample']!r}")
        time.sleep(20)
    out = f"/home/max/llm-stack/logs/probe-ctx-{LABEL}.json"
    with open(out, "w") as fh:
        json.dump({"label": LABEL, "model": MODEL, "port": PORT, "rows": rows}, fh, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
