#!/usr/bin/env python3
"""Larger multimodal sweep: scaled image / audio / video sizes.

Samples peak host memory at 200 ms (Spark unified mem ≈ GPU mem).
"""
import json
import sys
import threading
import time
sys.path.insert(0, "/home/max/llm-stack/bin")
from importlib import import_module
m = import_module("bench-deep")

MODEL = "nemotron-3-nano-omni"

CASES = [
    # (label, content_blocks, max_tokens)
    ("image-small (640x1280 PNG, 33KB)", [
        {"type": "image_url", "image_url": {
            "url": "file:///home/max/llama.cpp-gx10-dsv4/media/llama1-banner.png"}},
        {"type": "text", "text": "Describe this image in two sentences."},
    ], 256),
    ("image-1080p (1920x1080 JPEG, 512KB)", [
        {"type": "image_url", "image_url": {
            "url": "file:///home/max/lingbot/data/runs/20260420-160102/input_frames/000158.jpg"}},
        {"type": "text", "text": "Describe what is happening in this image."},
    ], 256),
    ("audio-tiny (1.5s mp3, 137KB)", [
        {"type": "audio_url", "audio_url": {
            "url": "file:///home/max/llama.cpp-gx10-dsv4/tools/mtmd/test-2.mp3"}},
        {"type": "text", "text": "Transcribe this audio."},
    ], 256),
    ("audio-23s (16kHz mono wav, 720KB)", [
        {"type": "audio_url", "audio_url": {
            "url": "file:///home/max/lingbot/data/_bench_media/audio_23s.wav"}},
        {"type": "text", "text": "Transcribe this audio."},
    ], 384),
    ("audio-5min (16kHz mono wav, 9.2MB)", [
        {"type": "audio_url", "audio_url": {
            "url": "file:///home/max/lingbot/data/_bench_media/audio_5min.wav"}},
        {"type": "text", "text": "Summarize what is being said in this audio."},
    ], 384),
    ("video-low (518x294 15fps 15.8s, 2.1MB)", [
        {"type": "video_url", "video_url": {
            "url": "file:///home/max/lingbot/data/smoke.mp4"}},
        {"type": "text", "text": "Describe this video briefly."},
    ], 384),
    ("video-1080p (1920x1080 30fps 23s, 56MB)", [
        {"type": "video_url", "video_url": {
            "url": "file:///home/max/lingbot/data/runs/20260420-160102/input.mp4"}},
        {"type": "text", "text": "Describe what is happening in this video."},
    ], 384),
]


def sample(stop, peaks):
    while not stop["v"]:
        peaks.append(m.meminfo_used_gb())
        time.sleep(0.2)


def kv_pct():
    import urllib.request
    try:
        with urllib.request.urlopen("http://192.168.1.12:9012/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None


def run_one(label, content, max_toks):
    print(f"\n=== {label} ===", flush=True)
    base = m.meminfo_used_gb()
    print(f"  baseline mem: {base:.2f} GB  KV%: {(kv_pct() or 0)*100:.1f}", flush=True)
    stop, peaks = {"v": False}, [base]
    th = threading.Thread(target=sample, args=(stop, peaks))
    th.start()

    msgs = [{"role": "user", "content": content}]
    err = None
    try:
        r = m.chat_streaming(MODEL, msgs, max_toks, timeout=600)
        ok = True
    except Exception as e:
        r = {"error": str(e)[:300]}
        ok = False
        err = str(e)[:300]

    stop["v"] = True
    th.join()
    peak = max(peaks)
    delta_mb = (peak - base) * 1024
    print(f"  peak mem: {peak:.2f} GB  Δ={delta_mb:+.0f} MB", flush=True)
    if ok:
        print(f"  TTFT {r['ttft_ms']:.0f} ms  decode {r['decode_tok_s']:.1f} tok/s  "
              f"({r['completion_tokens']} tok)", flush=True)
    else:
        print(f"  FAILED: {err}", flush=True)

    # Get usage to measure prompt tokens (= visual tokens too for multimodal)
    in_tok = None
    if ok:
        body = {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0.0}
        try:
            data = m.post_json("/v1/chat/completions", body, timeout=300)
            in_tok = data.get("usage", {}).get("prompt_tokens")
            print(f"  prompt_tokens: {in_tok}", flush=True)
        except Exception:
            pass

    return {
        "case": label, "ok": ok,
        "base_gb": round(base, 2), "peak_gb": round(peak, 2),
        "delta_mb": round(delta_mb, 0),
        "ttft_ms": r.get("ttft_ms") if ok else None,
        "decode_tok_s": r.get("decode_tok_s") if ok else None,
        "completion_tokens": r.get("completion_tokens") if ok else None,
        "prompt_tokens": in_tok,
        "error": err,
    }


results = [run_one(*c) for c in CASES]

print("\n" + "=" * 96)
print(f"{'case':<48} {'in_tok':>7} {'TTFT_s':>7} {'dec_t/s':>8} {'Δmem_MB':>8}  status")
print("-" * 96)
for r in results:
    s = "OK" if r["ok"] else "FAIL"
    print(f"{r['case']:<48} "
          f"{str(r.get('prompt_tokens') or '-'):>7} "
          f"{(r['ttft_ms']/1000 if r['ttft_ms'] else 0):>7.2f} "
          f"{(r['decode_tok_s'] or 0):>8.1f} "
          f"{r['delta_mb']:>8.0f}  {s}")

with open("/home/max/llm-stack/logs/bench-multimodal-large.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: /home/max/llm-stack/logs/bench-multimodal-large.json")
