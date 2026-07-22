#!/usr/bin/env python3
"""Prefill/decode A/B for Hy3 quants via llama-server's NATIVE /completion.

Reads server-reported `timings` (prompt_per_second / predicted_per_second) so
there is no OpenAI-compat layer in the measurement. Warmup + median-of-3 per
bucket to fight the GB10 allocator-state confounder (see ds4 memory).

Usage: bench-hy3-prefill.py --port 9028 --label iq1m [--buckets 2048,8192,16384]
"""
import argparse, json, statistics, time, urllib.request

CHUNK = (
    "def solve(nums, target):\n"
    "    seen = {}\n"
    "    for i, n in enumerate(nums):\n"
    "        if target - n in seen:\n"
    "            return [seen[target - n], i]\n"
    "        seen[n] = i\n"
    "    return []\n"
    "# The quick brown fox jumps over the lazy dog while refactoring the module.\n"
)

def make_prompt(approx_tokens):
    # ~0.3 tokens/char for this text; oversize then rely on server prompt_n.
    reps = max(1, int(approx_tokens / 60))
    return (CHUNK * reps)

def post(port, prompt, n_predict):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9028)
    ap.add_argument("--label", default="?")
    ap.add_argument("--buckets", default="2048,8192,16384")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--npredict", type=int, default=32)
    args = ap.parse_args()
    buckets = [int(x) for x in args.buckets.split(",")]

    print(f"\n=== {args.label} (port {args.port}) ===")
    print(f"{'bucket':>8} {'prompt_n':>9} {'prefill t/s':>13} {'decode t/s':>11}")
    results = {}
    for b in buckets:
        prompt = make_prompt(b)
        # warmup (populate allocator/caches), not counted
        post(args.port, prompt, args.npredict)
        time.sleep(2)
        pf, dc, pn = [], [], None
        for _ in range(args.runs):
            d, _wall = post(args.port, prompt, args.npredict)
            t = d.get("timings", {})
            pn = t.get("prompt_n")
            pf.append(t.get("prompt_per_second") or 0.0)
            dc.append(t.get("predicted_per_second") or 0.0)
            time.sleep(2)
        mpf, mdc = statistics.median(pf), statistics.median(dc)
        results[b] = {"prompt_n": pn, "prefill_tps": mpf, "decode_tps": mdc,
                      "prefill_runs": pf, "decode_runs": dc}
        print(f"{b:>8} {str(pn):>9} {mpf:>13.1f} {mdc:>11.1f}")
    print("\nJSON:", json.dumps({"label": args.label, "results": results}))

if __name__ == "__main__":
    main()
