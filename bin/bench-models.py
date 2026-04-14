#!/usr/bin/env python3
"""Benchmark each llama-swap model: cold-load time, memory footprint, tok/s."""
import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

GATEWAY = "http://192.168.1.12:8080"
MODELS = [
    "qwen3.5-35b-a3b",
    "gemma-4-26b-a4b",
    "minimax-m2.7",
    "supergemma-4-26b",
    "qwen3.5-35b-distill",
    "qwen3.5-122b-nvfp4",
    "gemma-4-e4b",
]
COLD_TIMEOUT_S = 900
WARM_TIMEOUT_S = 180
WARM_MAX_TOKENS = 256
PROMPT = "Write the numbers from 1 to 30 separated by commas, then stop."


def meminfo_used_gb():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k.strip()] = int(v.strip().split()[0])
    return (d["MemTotal"] - d["MemAvailable"]) / 1024 / 1024


def sample_peak(stop, out):
    peak = meminfo_used_gb()
    while not stop["v"]:
        peak = max(peak, meminfo_used_gb())
        time.sleep(0.5)
    out["peak"] = peak


def post_json(path, body, timeout):
    req = urllib.request.Request(
        GATEWAY + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def try_unload():
    for path in ("/unload", "/upstream/unload"):
        try:
            urllib.request.urlopen(GATEWAY + path, timeout=60).read()
            return True
        except Exception:
            pass
    return False


def chat(model, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    data = post_json("/v1/chat/completions", body, timeout)
    return data, time.perf_counter() - t0


def bench(model):
    print(f"\n=== {model} ===", flush=True)
    try_unload()
    time.sleep(3)
    base = meminfo_used_gb()
    print(f"  baseline mem: {base:.1f} GB", flush=True)

    stop = {"v": False}
    out = {}
    mon = threading.Thread(target=sample_peak, args=(stop, out))
    mon.start()

    cold_s = None
    peak = base
    model_gb = None
    try:
        _, cold_s = chat(model, max_tokens=8, timeout=COLD_TIMEOUT_S)
        stop["v"] = True
        mon.join()
        peak = out["peak"]
        model_gb = peak - base
        print(f"  cold swap+load+gen: {cold_s:.1f} s", flush=True)
        print(f"  peak mem: {peak:.1f} GB  (≈{model_gb:.1f} GB for this model)", flush=True)
    except Exception as e:
        stop["v"] = True
        mon.join()
        print(f"  FAILED cold: {e}", flush=True)
        return {"model": model, "status": "fail_cold", "error": str(e)[:200]}

    try:
        data, warm_s = chat(model, max_tokens=WARM_MAX_TOKENS, timeout=WARM_TIMEOUT_S)
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        prompt_tok = usage.get("prompt_tokens", 0)
        tps = comp / warm_s if warm_s > 0 else 0
        print(f"  warm gen: {comp} tok in {warm_s:.2f} s → {tps:.1f} tok/s", flush=True)
        return {
            "model": model,
            "status": "ok",
            "cold_s": round(cold_s, 1),
            "peak_mem_gb": round(peak, 1),
            "model_gb": round(model_gb, 1),
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp,
            "warm_s": round(warm_s, 2),
            "tok_per_s": round(tps, 1),
        }
    except Exception as e:
        print(f"  FAILED warm: {e}", flush=True)
        return {
            "model": model,
            "status": "fail_warm",
            "cold_s": round(cold_s, 1),
            "peak_mem_gb": round(peak, 1),
            "model_gb": round(model_gb, 1) if model_gb is not None else None,
            "error": str(e)[:200],
        }


def main():
    results = [bench(m) for m in MODELS]
    print("\n" + "=" * 88)
    print(f"{'model':<28} {'cold_s':>8} {'peak_GB':>8} {'model_GB':>9} {'tok/s':>7}  {'status'}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['model']:<28} "
            f"{str(r.get('cold_s','-')):>8} "
            f"{str(r.get('peak_mem_gb','-')):>8} "
            f"{str(r.get('model_gb','-')):>9} "
            f"{str(r.get('tok_per_s','-')):>7}  "
            f"{r['status']}"
        )
    out = Path("/home/max/llm-stack/logs/bench-results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
