#!/usr/bin/env python3
"""ds4 config A/B/C with concurrency sweep (c=1,2,4).

Each arm runs DIRECTLY (bypasses llama-swap) on a test port. For each
concurrency level we fire c simultaneous chat requests and report aggregate
tok/s (sum completion_tokens / wall span) and median per-stream tok/s, over
RUNS runs with idle between (GB10 allocator-state confounder).

Usage:  python3 bench-ds4-matrix.py A B        # run arms A and B
        python3 bench-ds4-matrix.py C          # run arm C (needs q2-q4 gguf)
"""
import json, os, subprocess, sys, threading, time, urllib.request, statistics
from pathlib import Path

PORT = 9099
CTX = 131072
GENTOK = 200
RUNS = 3
IDLE = 90              # s between runs (allocator settle)
CONCURRENCY = [1, 2, 4]
GGUF = "/home/max/ds4/gguf"
Q2 = f"{GGUF}/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
Q2Q4 = f"{GGUF}/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf"
MTP = f"{GGUF}/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
PROMPT = ("Write a Python function quicksort(arr) with a docstring and a usage "
          "example, then explain its average and worst-case time complexity.")

COMMON = ["--cuda", "--host", "127.0.0.1", "--port", str(PORT), "--ctx", str(CTX),
          "--kv-disk-dir", "/tmp/ds4mx-kv", "--kv-disk-space-mb", "8192"]
ARMS = {
    "A": {"desc": "fork q2 + custom Q4 decode (control)",
          "bin": "/home/max/ds4-q4/ds4-server",
          "env": {"DS4_CUDA_Q8_F16_CACHE_RESERVE_MB": "1024", "DS4_CUDA_Q4_DECODE": "1"},
          "args": ["-m", Q2] + COMMON},
    "B": {"desc": "mainline q2 + MTP draft=4",
          "bin": "/home/max/ds4-mainline/ds4-server",
          "env": {"DS4_CUDA_Q8_F16_CACHE_RESERVE_MB": "1024"},
          "args": ["-m", Q2, "--mtp", MTP, "--mtp-draft", "4", "--mtp-margin", "3"] + COMMON},
    "C": {"desc": "mainline q2-q4 (native Q4_K experts) + MTP draft=4",
          "bin": "/home/max/ds4-mainline/ds4-server",
          "env": {"DS4_CUDA_Q8_F16_CACHE_RESERVE_MB": "1024"},
          "args": ["-m", Q2Q4, "--mtp", MTP, "--mtp-draft", "4", "--mtp-margin", "3"] + COMMON},
}


def http_up():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def one_request(out, idx):
    body = json.dumps({"model": "deepseek-v4-flash",
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": GENTOK, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        t1 = time.perf_counter()
        u = d.get("usage", {})
        comp = u.get("completion_tokens", 0)
        m = d["choices"][0]["message"]
        sample = (m.get("content") or m.get("reasoning_content") or "")[:300]
        out[idx] = {"t0": t0, "t1": t1, "tok": comp, "tps": comp/(t1-t0) if t1 > t0 else 0, "sample": sample}
    except Exception as e:
        out[idx] = {"t0": t0, "t1": time.perf_counter(), "tok": 0, "tps": 0, "err": str(e)[:120]}


def run_concurrency(c):
    out = {}
    threads = [threading.Thread(target=one_request, args=(out, i)) for i in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    res = [out[i] for i in range(c)]
    span = max(r["t1"] for r in res) - min(r["t0"] for r in res)
    tot = sum(r["tok"] for r in res)
    agg = tot/span if span > 0 else 0
    per = statistics.median([r["tps"] for r in res]) if res else 0
    errs = [r.get("err") for r in res if r.get("err")]
    return {"agg_tps": round(agg, 1), "per_stream_tps": round(per, 1),
            "tot_tok": tot, "span_s": round(span, 2), "errs": errs,
            "sample": res[0].get("sample", "")}


def bench_arm(key):
    a = ARMS[key]
    print(f"\n{'='*78}\n### ARM {key}: {a['desc']}\n### {a['bin']}\n{'='*78}", flush=True)
    if not os.path.exists(a["args"][a["args"].index("-m")+1]):
        print(f"  ✗ model file missing: {a['args'][a['args'].index('-m')+1]}", flush=True)
        return {"arm": key, "status": "missing_model"}
    os.system("rm -rf /tmp/ds4mx-kv")
    env = {**os.environ, **a["env"]}
    log = open(f"/tmp/ds4mx-server-{key}.log", "w")
    t0 = time.perf_counter()
    proc = subprocess.Popen([a["bin"]] + a["args"], env=env, stdout=log, stderr=subprocess.STDOUT)
    print("  loading model...", flush=True)
    up = False
    for _ in range(180):
        if http_up():
            up = True
            break
        if proc.poll() is not None:
            print(f"  ✗ server exited early (code {proc.returncode}); see /tmp/ds4mx-server-{key}.log", flush=True)
            return {"arm": key, "status": "server_died"}
        time.sleep(2)
    if not up:
        print("  ✗ never came up", flush=True)
        proc.terminate()
        return {"arm": key, "status": "no_start"}
    load_s = round(time.perf_counter()-t0, 1)
    print(f"  up after {load_s}s; warm-up...", flush=True)
    run_concurrency(1)  # warm
    result = {"arm": key, "desc": a["desc"], "status": "ok", "load_s": load_s, "by_c": {}}
    for c in CONCURRENCY:
        print(f"  --- c={c} ({RUNS} runs, idle {IDLE}s) ---", flush=True)
        aggs, pers = [], []
        last = None
        for r in range(RUNS):
            m = run_concurrency(c)
            last = m
            aggs.append(m["agg_tps"])
            pers.append(m["per_stream_tps"])
            print(f"    run {r+1}: agg {m['agg_tps']} tok/s | per-stream {m['per_stream_tps']} | {m['tot_tok']} tok / {m['span_s']}s"
                  + (f" | ERR {m['errs']}" if m["errs"] else ""), flush=True)
            if r < RUNS-1:
                time.sleep(IDLE)
        result["by_c"][c] = {"agg_med": statistics.median(aggs),
                             "per_stream_med": statistics.median(pers),
                             "aggs": aggs}
        print(f"    >>> c={c} MEDIAN agg {statistics.median(aggs)} tok/s | per-stream {statistics.median(pers)}", flush=True)
    print(f"  sample @c=1: {last['sample'][:200] if last else ''}", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    print("  cooldown 45s...", flush=True)
    time.sleep(45)
    return result


def main():
    keys = [k for k in sys.argv[1:] if k in ARMS] or ["A", "B", "C"]
    print(f"ds4 matrix bench — arms {keys}, concurrency {CONCURRENCY}, {RUNS} runs/median", flush=True)
    results = [bench_arm(k) for k in keys]
    print(f"\n{'='*78}\nSUMMARY (median tok/s)\n{'='*78}")
    print(f"{'arm':<4} {'load_s':>7} | " + " | ".join(f"c={c} agg / per" for c in CONCURRENCY))
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['arm']:<4} {r.get('status')}")
            continue
        cells = " | ".join(f"{r['by_c'][c]['agg_med']:>5} / {r['by_c'][c]['per_stream_med']:<5}" for c in CONCURRENCY)
        print(f"{r['arm']:<4} {r['load_s']:>7} | {cells}")
    outp = Path("/home/max/llm-stack/logs/bench-ds4-matrix.json")
    prev = json.loads(outp.read_text()) if outp.exists() else []
    outp.write_text(json.dumps({"runs": prev + results}, indent=2) if False else json.dumps(prev + results, indent=2))
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
