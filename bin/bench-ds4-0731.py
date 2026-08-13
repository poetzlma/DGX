#!/usr/bin/env python3
"""ds4 prefill+decode bench, streaming, context-swept — for the 0731 weights swap.

WHY A NEW HARNESS: bench-ds4-matrix.py only measures aggregate decode on a tiny
prompt, but the two numbers we have a recorded baseline for are PREFILL (~396
tok/s @2048 ctx, ngc-shj host-register fallback) and DECODE (20.8 tok/s), and
prefill is where this lane's wall clock actually lives on ~100k:4k traffic.
This one streams, so TTFT separates prefill from decode cleanly.

PROTOCOL (do not shorten): GB10 decode t/s is ~44x dominated by allocator/page-
cache state, not kernels — see memory project_ds4_gb10_bench_confounder. Every
point is RUNS runs with IDLE seconds between, reported as a median. A single
run is noise, and a fast first run after load is especially untrustworthy.

Prompts are real source text (not random tokens) so prefill work is realistic.

Usage:
  python3 bench-ds4-0731.py                 # full sweep against port 9099
  DS4_PORT=9010 python3 bench-ds4-0731.py   # against the live lane
  python3 bench-ds4-0731.py --quick         # 1 run/point, short ctx only
"""
import json, os, random, re, statistics, sys, threading, time, urllib.request
from pathlib import Path

PORT = int(os.environ.get("DS4_PORT", "9099"))
BASE = f"http://127.0.0.1:{PORT}"
# ds4's persistent disk-KV cache makes a repeated prompt a CACHE HIT, not a
# prefill: the engine logs `kv cache hit ... load=28.8 ms` then
# `chat ctx=1859..1859:0 prompt start` — zero tokens to prefill. A naive
# repeated-prompt bench therefore measures cache replay (~0.36 s TTFT) and
# never touches the prefill path at all. Every run below gets a unique prefix
# at token 0 so the prefix match fails and the full prompt is prefilled.
# The cache-hit path IS separately interesting for this lane's ~100k:4k
# prefix-heavy traffic, so it is measured deliberately as its own point.
#
# ds4 also emits no `usage` block on the streaming SSE, so prompt_tokens has to
# come from the engine log line `chat ctx=A..B:N prompt start`, where N is the
# number of tokens actually prefilled. SERVER_LOG points at it.
SERVER_LOG = os.environ.get("DS4_SERVER_LOG", "")
QUICK = "--quick" in sys.argv
CTX_TARGETS = [2048, 8192, 32768] if QUICK else [2048, 8192, 32768, 100000]
# Per-point run counts. The short points are cheap, so they get the full 3-run
# median; the deep ones are prefill-dominated (100k at ~400 tok/s is ~4 min of
# wall clock per run) and prefill is far less allocator-sensitive than decode,
# so 2 runs there buys most of the confidence at half the cost. Sized to fit a
# ~1 h window with the prod lane parked.
RUNS_BY_CTX = {2048: 3, 8192: 3, 32768: 2, 100000: 2}
RUNS = 3
IDLE = 20 if QUICK else 150          # allocator settle between runs
GENTOK = 200
OUT = Path("/home/max/llm-stack/logs/bench-ds4-0731.json")

TASK = ("\n\nRead the source above. Name the single hottest function for decode "
        "latency and explain in three sentences why it dominates. Be specific.")


def corpus():
    """Real text, deterministic order, big enough for the largest target."""
    roots = [Path("/home/max/ds4-q4"), Path("/home/max/llm-stack/bin")]
    exts = {".c", ".h", ".cu", ".py", ".sh", ".md"}
    chunks = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in exts:
                try:
                    chunks.append(f"\n\n===== {p} =====\n" + p.read_text(errors="ignore"))
                except Exception:
                    pass
    text = "".join(chunks)
    if not text:
        sys.exit("no corpus found — cannot build realistic prompts")
    return text


CORPUS = corpus()


def make_prompt(target_tokens, unique=True):
    """~3.6 chars/token for source code; trimmed to target by measured feedback.

    `unique` puts a fresh marker at the very START so ds4's prefix match fails
    and we measure real prefill. Pass unique=False to deliberately re-use a
    prompt and measure the cache-hit path instead.
    """
    need = int(target_tokens * 3.6)
    while len(CORPUS) < need:
        globals()["CORPUS"] = CORPUS * 2
    head = f"// bench-run {random.getrandbits(48):012x}\n" if unique else ""
    return head + CORPUS[:need] + TASK


def log_size():
    try:
        return os.path.getsize(SERVER_LOG) if SERVER_LOG else 0
    except OSError:
        return 0


def prefilled_tokens_since(offset):
    """Read the engine's own count of tokens prefilled for the last request."""
    if not SERVER_LOG:
        return None, None
    try:
        with open(SERVER_LOG, "r", errors="ignore") as fh:
            fh.seek(offset)
            tail = fh.read()
    except OSError:
        return None, None
    hits = re.findall(r"chat ctx=(\d+)\.\.(\d+):(\d+) prompt start", tail)
    if not hits:
        return None, None
    a, b, n = hits[-1]
    return int(n), int(b)      # tokens prefilled, total context after prefill


def stream_once(prompt, gentok=GENTOK):
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gentok, "temperature": 0, "stream": True,
        # 2026-08-02: ask for usage so this harness also works against
        # llama.cpp (used for the unsloth UD-* quants, which cannot load on
        # the ds4 engine). ds4 ignores unknown body fields and keeps reporting
        # prompt_tokens via SERVER_LOG, so this is safe for both lanes.
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    log0 = log_size()
    t0 = time.perf_counter()
    t_first = None
    ntok = 0
    text = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning") \
                        or delta.get("reasoning_content") or ""
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        ntok += 1
                        text.append(piece)
    except Exception as e:
        return {"err": str(e)[:160]}
    t_end = time.perf_counter()
    if t_first is None:
        return {"err": "no tokens streamed"}
    n_prefilled, total_ctx = prefilled_tokens_since(log0)
    ptok = usage.get("prompt_tokens") or n_prefilled
    ctok = usage.get("completion_tokens") or ntok
    ttft = t_first - t0
    dec_s = t_end - t_first
    return {
        "t_first": t_first, "t_end": t_end,
        "prompt_tokens": ptok, "tokens_prefilled": n_prefilled, "total_ctx": total_ctx,
        "completion_tokens": ctok,
        "ttft_s": round(ttft, 2),
        "prefill_tps": round(ptok / ttft, 1) if ptok and ttft > 0 else None,
        "decode_tps": round((ctok - 1) / dec_s, 2) if dec_s > 0 and ctok > 1 else 0,
        "total_s": round(t_end - t0, 2),
        "sample": "".join(text)[:200],
    }


def concurrent(prompts, c, gentok=GENTOK):
    """`prompts` may be one string (shared) or a list of c distinct prompts."""
    if isinstance(prompts, str):
        prompts = [prompts] * c
    res = {}

    def w(i):
        res[i] = stream_once(prompts[i], gentok)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(c)]
    span0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    span = time.perf_counter() - span0
    ok = [r for r in res.values() if not r.get("err")]
    if not ok:
        return {"err": [r.get("err") for r in res.values()][:2]}
    return {
        # Aggregate over the DECODE window (first token seen -> last stream
        # done), not the request span. ds4 serializes prefills, so a span that
        # includes them reports an "aggregate" below per-stream — arithmetic
        # nonsense (measured: agg 6.6 vs per-stream 20.0 at c=2). The prefill
        # serialization itself is visible in ttft_med, where it belongs.
        "agg_decode_tps": round(sum(r["completion_tokens"] for r in ok)
                                / (max(r["t_end"] for r in ok)
                                   - min(r["t_first"] for r in ok)), 1),
        "req_span_tps": round(sum(r["completion_tokens"] for r in ok) / span, 1),
        "per_stream_med": round(statistics.median(r["decode_tps"] for r in ok), 2),
        "ttft_med": round(statistics.median(r["ttft_s"] for r in ok), 2),
        "n_ok": len(ok), "span_s": round(span, 1),
    }


def wait_up(timeout=1200):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    print(f"ds4 0731 bench → {BASE} | ctx {CTX_TARGETS} | {RUNS} runs/point, idle {IDLE}s",
          flush=True)
    if not wait_up():
        sys.exit(f"server not up on {BASE}")
    # THREE warm-ups, not one. The Q4 lazy cache populates per-block on first
    # decode access, and it takes ~3 queries to fully warm: a cold relaunch
    # measured 11.75 tok/s against 20.8 steady-state on the preview weights.
    # Benching after a single warm-up would invent a regression.
    print("server up; 3 warm-up requests (Q4 lazy cache populates over ~3)...", flush=True)
    for i in range(3):
        w = stream_once(make_prompt(2048, unique=True), gentok=48)
        print(f"  warm-up {i+1}: decode {w.get('decode_tps')} tok/s "
              f"TTFT {w.get('ttft_s')}s", flush=True)

    results = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "port": PORT,
               "runs_per_point": RUNS_BY_CTX, "points": [], "concurrency": {}}

    for target in CTX_TARGETS:
        runs = 1 if QUICK else RUNS_BY_CTX.get(target, RUNS)
        print(f"\n--- ctx target {target} ({runs} runs, unique prefix per run) ---", flush=True)
        rows = []
        for i in range(runs):
            # Fresh prefix every run, or ds4 replays the stored KV and we
            # measure a 29 ms cache load instead of prefill.
            r = stream_once(make_prompt(target, unique=True))
            rows.append(r)
            if r.get("err"):
                print(f"  run {i+1}: ERROR {r['err']}", flush=True)
            else:
                print(f"  run {i+1}: prefilled {r['tokens_prefilled']} tok "
                      f"(ctx {r['total_ctx']}) | TTFT {r['ttft_s']}s "
                      f"| prefill {r['prefill_tps']} tok/s | decode {r['decode_tps']} tok/s",
                      flush=True)
            if i < runs - 1:
                time.sleep(IDLE)
        good = [r for r in rows if not r.get("err")]
        if good:
            pf = [r["prefill_tps"] for r in good if r.get("prefill_tps")]
            point = {
                "ctx_target": target,
                "prompt_tokens": good[0]["prompt_tokens"],
                "tokens_prefilled": good[0].get("tokens_prefilled"),
                "ttft_med": statistics.median(r["ttft_s"] for r in good),
                # None rather than a crash if the engine log was unreadable or
                # every run turned out to be a cache hit.
                "prefill_tps_med": statistics.median(pf) if pf else None,
                "decode_tps_med": statistics.median(r["decode_tps"] for r in good),
                "runs": [{k: r[k] for k in ("ttft_s", "prefill_tps", "decode_tps")} for r in good],
                "sample": good[-1]["sample"],
            }
            print(f"  >>> MEDIAN prefill {point['prefill_tps_med']} tok/s | "
                  f"decode {point['decode_tps_med']} tok/s", flush=True)
        else:
            point = {"ctx_target": target, "error": rows[-1].get("err")}
        results["points"].append(point)
        time.sleep(IDLE)

    # The cache-hit path, measured on purpose. This lane serves ~100k:4k
    # prefix-heavy coding traffic, so a warm prefix is the COMMON case in
    # production and its TTFT is the number real clients feel.
    print("\n--- disk-KV cache-hit path @32k (same prompt twice) ---", flush=True)
    warm = make_prompt(32768, unique=True)
    cold = stream_once(warm)
    time.sleep(5)
    hot = stream_once(warm)
    if not cold.get("err") and not hot.get("err"):
        results["cache_hit"] = {
            "cold_ttft_s": cold["ttft_s"], "cold_prefilled": cold["tokens_prefilled"],
            "hot_ttft_s": hot["ttft_s"], "hot_prefilled": hot["tokens_prefilled"],
            "speedup": round(cold["ttft_s"] / hot["ttft_s"], 1) if hot["ttft_s"] else None,
        }
        print(f"  cold: TTFT {cold['ttft_s']}s ({cold['tokens_prefilled']} tok prefilled) | "
              f"hot: TTFT {hot['ttft_s']}s ({hot['tokens_prefilled']} tok prefilled) | "
              f"{results['cache_hit']['speedup']}x", flush=True)
    time.sleep(IDLE)

    if not QUICK:
        print("\n--- concurrency @8k prompt (distinct prompt per stream) ---", flush=True)
        for c in (2, 4):
            # Distinct prompts, or stream 2..c would ride the first stream's
            # cached prefix and the aggregate would be fiction.
            m = concurrent([make_prompt(8192, unique=True) for _ in range(c)], c)
            print(f"  c={c}: {m}", flush=True)
            results["concurrency"][c] = m
            time.sleep(IDLE)

    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    prev.append(results)
    OUT.write_text(json.dumps(prev, indent=2))
    print(f"\nsaved → {OUT}", flush=True)

    print("\n" + "=" * 72)
    print(f"{'ctx':>8} {'prefilled':>11} {'TTFT s':>8} {'prefill t/s':>12} {'decode t/s':>11}")
    print("=" * 72)
    for p in results["points"]:
        if "error" in p:
            print(f"{p['ctx_target']:>8} {'ERROR: ' + str(p['error'])[:50]}")
        else:
            pf = p["prefill_tps_med"]
            pfs = f"{pf:>12.1f}" if pf is not None else f"{'n/a':>12}"
            print(f"{p['ctx_target']:>8} {str(p.get('tokens_prefilled')):>11} "
                  f"{p['ttft_med']:>8.2f} {pfs} {p['decode_tps_med']:>11.2f}")
    print("\nBASELINE (preview weights, same binary, recorded 2026-05-18):")
    print("   prefill ~396 tok/s @2048 ctx | decode 20.74-20.83 tok/s steady-state")


if __name__ == "__main__":
    main()
