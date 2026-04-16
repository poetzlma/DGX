#!/usr/bin/env python3
"""Deep benchmark: TTFT, decode tok/s, multi-prompt, concurrency scaling."""

import concurrent.futures
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

GATEWAY_HOST = "192.168.1.12"
GATEWAY_PORT = 8080
GATEWAY = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

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
WARM_TIMEOUT_S = 300
WARM_RUNS = 3
CONCURRENCY_LEVELS = [3, 5]

SHORT_MSGS = [
    {"role": "user",
     "content": "Write the numbers from 1 to 30 separated by commas, then stop."}
]

MEDIUM_MSGS = [
    {"role": "user",
     "content": (
         "The history of artificial intelligence spans several decades, beginning "
         "with Alan Turing's foundational work in the 1950s. Early AI research "
         "focused on symbolic reasoning, expert systems, and game playing. The "
         "field experienced several 'AI winters' — periods of reduced funding and "
         "interest — before the deep learning revolution of the 2010s reignited "
         "progress. Modern large language models represent the latest paradigm, "
         "trained on vast internet-scale datasets using transformer architectures. "
         "These models demonstrate emergent capabilities including reasoning, code "
         "generation, and creative writing, though they also exhibit limitations "
         "such as hallucination and lack of grounding in physical reality.\n\n"
         "Based on the above passage, write a detailed 500-word essay analyzing "
         "the key turning points in AI history and their implications for the "
         "future of technology and society."
     )}
]

_FILLER = (
    "In software engineering, the design of distributed systems presents unique "
    "challenges that do not exist in single-machine architectures. Network "
    "partitions, variable latency, and partial failures require fundamentally "
    "different approaches to consistency, availability, and fault tolerance. The "
    "CAP theorem, formalized by Eric Brewer in 2000, establishes that a distributed "
    "system cannot simultaneously guarantee all three of consistency, availability, "
    "and partition tolerance — at most two can be achieved. This has profound "
    "implications for system design: databases like DynamoDB choose availability "
    "and partition tolerance (AP), accepting eventual consistency, while systems "
    "like Google Spanner use synchronized clocks (TrueTime) to approach all three "
    "simultaneously at the cost of additional hardware complexity. Modern "
    "microservice architectures must navigate these tradeoffs at every layer, from "
    "the database tier through message queues to API gateways. Patterns like saga "
    "orchestration, event sourcing, and CQRS (Command Query Responsibility "
    "Segregation) have emerged to manage distributed state without two-phase "
    "commit, which is too slow for internet-scale workloads. Circuit breakers, "
    "bulkheads, and retry policies with exponential backoff provide resilience at "
    "the network boundary. Observability — the combination of structured logging, "
    "distributed tracing, and metrics — has become essential for understanding "
    "behavior in systems where no single node has a complete view of the overall "
    "state.\n\n"
)

LONG_MSGS = [
    {"role": "user",
     "content": _FILLER * 15 + "Summarize the above text in exactly 3 bullet points."}
]

PROMPTS = {
    "short":        {"messages": SHORT_MSGS,  "max_tokens": 512},
    "medium":       {"messages": MEDIUM_MSGS, "max_tokens": 1024},
    "long_prefill": {"messages": LONG_MSGS,   "max_tokens": 256},
}


# ---------------------------------------------------------------------------
# Helpers (meminfo, unload) — same pattern as bench-models.py
# ---------------------------------------------------------------------------

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


def try_unload():
    for path in ("/unload", "/upstream/unload"):
        try:
            urllib.request.urlopen(GATEWAY + path, timeout=60).read()
            return True
        except Exception:
            pass
    return False


def post_json(path, body, timeout):
    req = urllib.request.Request(
        GATEWAY + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Streaming chat — measures TTFT and decode rate separately
# ---------------------------------------------------------------------------

def chat_streaming(model, messages, max_tokens, timeout=WARM_TIMEOUT_S):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    conn = http.client.HTTPConnection(GATEWAY_HOST, GATEWAY_PORT, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()

    ttft = None
    token_chunks = 0
    usage = {}

    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                token_chunks += 1

        if chunk.get("usage"):
            usage = chunk["usage"]

    total_s = time.perf_counter() - t0
    conn.close()

    comp = usage.get("completion_tokens", token_chunks)
    ttft_ms = ttft if ttft is not None else total_s * 1000
    decode_time = total_s - (ttft_ms / 1000) if ttft is not None else total_s
    decode_tok_s = comp / decode_time if decode_time > 0.001 else 0
    e2e_tok_s = comp / total_s if total_s > 0.001 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "decode_tok_s": round(decode_tok_s, 1),
        "e2e_tok_s": round(e2e_tok_s, 1),
        "completion_tokens": comp,
        "total_s": round(total_s, 2),
    }


# ---------------------------------------------------------------------------
# Concurrent requests
# ---------------------------------------------------------------------------

def run_concurrent(model, messages, max_tokens, n, timeout=WARM_TIMEOUT_S):
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [
            pool.submit(chat_streaming, model, messages, max_tokens, timeout)
            for _ in range(n)
        ]
        results = []
        for f in concurrent.futures.as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                results.append({
                    "error": str(e)[:200],
                    "e2e_tok_s": 0, "ttft_ms": 0,
                    "total_s": 0, "completion_tokens": 0,
                })
    wall_time = time.perf_counter() - wall_start

    ok = [r for r in results if "error" not in r]
    total_tokens = sum(r["completion_tokens"] for r in results)
    agg_tok_s = total_tokens / wall_time if wall_time > 0.001 else 0

    return {
        "n_concurrent": n,
        "wall_time_s": round(wall_time, 2),
        "agg_tok_s": round(agg_tok_s, 1),
        "mean_tok_s": round(statistics.mean(r["e2e_tok_s"] for r in ok), 1) if ok else 0,
        "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in ok), 1) if ok else 0,
        "mean_latency_s": round(statistics.mean(r["total_s"] for r in ok), 2) if ok else 0,
        "errors": len(results) - len(ok),
    }


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def compute_stats(values):
    if not values:
        return {"mean": 0, "std": 0, "values": []}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0
    return {
        "mean": round(m, 1),
        "std": round(s, 1),
        "values": [round(v, 1) for v in values],
    }


# ---------------------------------------------------------------------------
# Benchmark one model
# ---------------------------------------------------------------------------

def bench_model(model):
    print(f"\n{'='*70}\n  {model}\n{'='*70}", flush=True)

    try_unload()
    time.sleep(3)
    base_mem = meminfo_used_gb()
    print(f"  baseline mem: {base_mem:.1f} GB", flush=True)

    stop = {"v": False}
    out = {}
    mon = threading.Thread(target=sample_peak, args=(stop, out))
    mon.start()

    result = {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        body = {
            "model": model,
            "messages": SHORT_MSGS,
            "max_tokens": 8,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        post_json("/v1/chat/completions", body, COLD_TIMEOUT_S)
        cold_s = time.perf_counter() - t0
        stop["v"] = True
        mon.join()
        peak = out["peak"]
        model_gb = peak - base_mem
        result.update({
            "status": "ok",
            "cold_s": round(cold_s, 1),
            "peak_mem_gb": round(peak, 1),
            "model_gb": round(model_gb, 1),
        })
        print(f"  cold start: {cold_s:.1f} s", flush=True)
        print(f"  peak mem: {peak:.1f} GB (model ~{model_gb:.1f} GB)", flush=True)
    except Exception as e:
        stop["v"] = True
        mon.join()
        print(f"  FAILED cold start: {e}", flush=True)
        result.update({"status": "fail_cold", "error": str(e)[:200]})
        return result

    # Warm passes: each prompt tier × WARM_RUNS
    result["prompts"] = {}
    for tier, cfg in PROMPTS.items():
        ttfts, decode_rates, e2e_rates, completions = [], [], [], []
        print(f"\n  --- {tier} (x{WARM_RUNS}) ---", flush=True)
        for i in range(WARM_RUNS):
            try:
                r = chat_streaming(model, cfg["messages"], cfg["max_tokens"])
                ttfts.append(r["ttft_ms"])
                decode_rates.append(r["decode_tok_s"])
                e2e_rates.append(r["e2e_tok_s"])
                completions.append(r["completion_tokens"])
                print(f"    run {i+1}: {r['e2e_tok_s']:.1f} tok/s  "
                      f"TTFT {r['ttft_ms']:.0f}ms  "
                      f"decode {r['decode_tok_s']:.1f} tok/s  "
                      f"({r['completion_tokens']} tok)", flush=True)
            except Exception as e:
                print(f"    run {i+1}: FAILED - {e}", flush=True)

        result["prompts"][tier] = {
            "ttft_ms": compute_stats(ttfts),
            "decode_tok_s": compute_stats(decode_rates),
            "e2e_tok_s": compute_stats(e2e_rates),
            "completion_tokens": compute_stats(completions),
        }

    # Concurrency passes (short prompt only)
    result["concurrency"] = {}
    for n in CONCURRENCY_LEVELS:
        print(f"\n  --- concurrency={n} (short prompt) ---", flush=True)
        try:
            cr = run_concurrent(model, SHORT_MSGS, 512, n)
            result["concurrency"][str(n)] = cr
            print(f"    agg: {cr['agg_tok_s']:.1f} tok/s  "
                  f"per-req: {cr['mean_tok_s']:.1f} tok/s  "
                  f"TTFT: {cr['mean_ttft_ms']:.0f}ms  "
                  f"latency: {cr['mean_latency_s']:.1f}s  "
                  f"wall: {cr['wall_time_s']:.1f}s", flush=True)
            if cr["errors"]:
                print(f"    errors: {cr['errors']}/{n}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            result["concurrency"][str(n)] = {"error": str(e)[:200]}

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    print(f"Deep benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Gateway: {GATEWAY}")
    print(f"Models: {len(models)} | Warm runs: {WARM_RUNS} | "
          f"Concurrency: {CONCURRENCY_LEVELS}")
    if models != MODELS:
        print(f"  Filter: {', '.join(models)}")

    results = [bench_model(m) for m in models]

    # Summary table
    print(f"\n\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    hdr = f"{'model':<28} {'cold_s':>7} {'mem_GB':>7} {'tok/s':>7} {'TTFT_ms':>8} {'decode':>7}"
    print(hdr)
    print("-" * 90)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['model']:<28} {'FAILED':>7}")
            continue
        sp = r.get("prompts", {}).get("short", {})
        e2e = sp.get("e2e_tok_s", {}).get("mean", 0)
        ttft = sp.get("ttft_ms", {}).get("mean", 0)
        dec = sp.get("decode_tok_s", {}).get("mean", 0)
        print(f"{r['model']:<28} {r['cold_s']:>7.1f} {r['model_gb']:>7.1f} "
              f"{e2e:>7.1f} {ttft:>8.1f} {dec:>7.1f}")

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_path = Path(f"/home/max/llm-stack/logs/bench-deep-{ts}.json")
    out_path.write_text(json.dumps(results, indent=2))
    latest = Path("/home/max/llm-stack/logs/bench-deep-latest.json")
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_path.name)
    print(f"\nSaved: {out_path}")
    print(f"Symlink: {latest} -> {out_path.name}")


if __name__ == "__main__":
    main()
