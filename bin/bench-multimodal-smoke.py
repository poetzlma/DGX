#!/usr/bin/env python3
"""Multimodal smoke test: send one image and one audio request, measure
latency and peak host memory delta (Spark unified mem ≈ GPU mem)."""
import json
import sys
import threading
import time
sys.path.insert(0, "/home/max/llm-stack/bin")
from importlib import import_module
m = import_module("bench-deep")

MODEL = "nemotron-3-nano-omni"
IMG = "file:///home/max/llama.cpp-gx10-dsv4/media/llama1-banner.png"
AUD = "file:///home/max/llama.cpp-gx10-dsv4/tools/mtmd/test-2.mp3"

CASES = [
    ("image", [
        {"type": "image_url", "image_url": {"url": IMG}},
        {"type": "text", "text": "Describe this image in two sentences."}
    ], 256),
    ("audio", [
        {"type": "audio_url", "audio_url": {"url": AUD}},
        {"type": "text", "text": "Transcribe this audio."}
    ], 256),
]


def sample(stop, peaks):
    while not stop["v"]:
        peaks.append(m.meminfo_used_gb())
        time.sleep(0.5)


def run_one(name, content, max_toks):
    print(f"\n=== {name} ===", flush=True)
    base = m.meminfo_used_gb()
    print(f"  baseline mem: {base:.2f} GB", flush=True)
    stop, peaks = {"v": False}, [base]
    th = threading.Thread(target=sample, args=(stop, peaks))
    th.start()

    msgs = [{"role": "user", "content": content}]
    try:
        r = m.chat_streaming(MODEL, msgs, max_toks, timeout=600)
        ok = True
    except Exception as e:
        r = {"error": str(e)[:300]}
        ok = False

    stop["v"] = True
    th.join()
    peak = max(peaks)
    delta = peak - base
    print(f"  peak mem: {peak:.2f} GB  Δ={delta:+.2f} GB", flush=True)
    if ok:
        print(f"  TTFT {r['ttft_ms']:.0f} ms  decode {r['decode_tok_s']:.1f} tok/s  "
              f"e2e {r['e2e_tok_s']:.1f} tok/s  ({r['completion_tokens']} tok)", flush=True)
    else:
        print(f"  FAILED: {r['error']}", flush=True)

    # Pull a non-stream response too so we can show what the model said
    body = {"model": MODEL, "messages": msgs, "max_tokens": 200, "temperature": 0.0}
    try:
        data = m.post_json("/v1/chat/completions", body, timeout=600)
        msg = data["choices"][0]["message"]
        # Reasoning parser puts post-think answer in .content, thinking in .reasoning
        text = msg.get("content") or msg.get("reasoning") or ""
        print(f"  output: {text.strip()[:300]}", flush=True)
    except Exception as e:
        print(f"  output FAILED: {str(e)[:200]}", flush=True)

    return {"case": name, "base_gb": base, "peak_gb": peak, "delta_gb": delta,
            "ok": ok, "result": r}


results = [run_one(*c) for c in CASES]
with open("/home/max/llm-stack/logs/bench-multimodal-smoke.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved: /home/max/llm-stack/logs/bench-multimodal-smoke.json")
