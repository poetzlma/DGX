#!/usr/bin/env python3
"""Context x concurrency decode-throughput sweep against an OpenAI-compatible
vLLM endpoint. Stdlib only.

Per cell (ctx, concurrency) it fires `concurrency` simultaneous streaming
completion requests, each with a UNIQUE long prefix (so prefix-caching doesn't
make prefill free and inflate numbers), a fixed decode length, and greedy
sampling. It separates TTFT from decode and reports:
  - prompt_tokens (mean actual, read back from usage)
  - TTFT ms (mean)
  - per-request decode tok/s (mean of completion/(t_last-t_first))
  - aggregate decode tok/s (sum completion tokens / wall window)

Usage:
  bench-ctx-conc.py --base-url http://127.0.0.1:9019 --model qwopus3.6-27b \
      --ctx 8000 20000 60000 120000 --conc 1 2 4 8 \
      --out-tokens 256 --json /home/max/qwopus-bench.json
"""
import argparse, json, time, threading, urllib.request, urllib.error, random, string, sys

WORDS = None
def rand_text(n_words, rng):
    # Varied tokens (not a repeated string) so the tokenizer doesn't compress
    # and so each request has a unique prefix -> no cross-request prefix cache.
    out = []
    for _ in range(n_words):
        L = rng.randint(3, 9)
        out.append(''.join(rng.choice(string.ascii_lowercase) for _ in range(L)))
    return ' '.join(out)

def post_stream(base_url, payload, timeout):
    """POST /v1/completions with stream=true. Returns (ttft, t_last, t_start,
    completion_tokens, prompt_tokens, err)."""
    url = base_url.rstrip('/') + '/v1/completions'
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    t_start = time.perf_counter()
    ttft = None; t_last = None; n_chunks = 0
    prompt_tokens = None; completion_tokens = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode('utf-8', 'ignore').strip()
                if not line or not line.startswith('data:'):
                    continue
                body = line[5:].strip()
                if body == '[DONE]':
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                ch = obj.get('choices') or [{}]
                txt = ch[0].get('text', '')
                if txt:
                    if ttft is None:
                        ttft = now - t_start
                    t_last = now
                    n_chunks += 1
                if obj.get('usage'):
                    prompt_tokens = obj['usage'].get('prompt_tokens')
                    completion_tokens = obj['usage'].get('completion_tokens')
    except Exception as e:
        return None, None, t_start, None, None, repr(e)
    if completion_tokens is None:
        completion_tokens = n_chunks
    return ttft, t_last, t_start, completion_tokens, prompt_tokens, None

def calibrate_ratio(base_url, model, timeout):
    """chars-per-token using a probe request."""
    probe = rand_text(400, random.Random(1))
    payload = {"model": model, "prompt": probe, "max_tokens": 1,
               "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    _, _, _, _, pt, err = post_stream(base_url, payload, timeout)
    if err or not pt:
        raise RuntimeError(f"calibration failed: {err}, prompt_tokens={pt}")
    return len(probe) / pt

def run_cell(base_url, model, target_ctx, conc, out_tokens, cpt, timeout):
    results = [None] * conc
    barrier = threading.Barrier(conc)
    def worker(i):
        rng = random.Random(10000 * target_ctx + i + int(time.time()))
        n_words = max(1, int((target_ctx - out_tokens) * cpt / 6.0))  # ~6 chars/word avg incl space
        prompt = f"req{i}-{rng.random()} " + rand_text(n_words, rng)
        payload = {"model": model, "prompt": prompt, "max_tokens": out_tokens,
                   "temperature": 0.0, "stream": True,
                   "stream_options": {"include_usage": True},
                   "ignore_eos": True}
        barrier.wait()
        results[i] = post_stream(base_url, payload, timeout)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in threads: t.start()
    for t in threads: t.join()

    ok = [r for r in results if r and r[5] is None and r[0] is not None and r[1] is not None]
    errs = [r[5] for r in results if r and r[5] is not None]
    if not ok:
        return {"ctx_target": target_ctx, "conc": conc, "ok": 0, "errors": errs[:3]}
    ttfts = [r[0] for r in ok]
    per_req_tps = [(r[3] - 1) / (r[1] - r[0]) for r in ok if r[1] > r[0] and r[3] and r[3] > 1]
    starts = [r[2] for r in ok]; lasts = [r[1] for r in ok]
    total_completion = sum(r[3] for r in ok if r[3])
    window = max(lasts) - min(starts)
    agg_tps = total_completion / window if window > 0 else 0
    prompt_toks = [r[4] for r in ok if r[4]]
    return {
        "ctx_target": target_ctx, "conc": conc, "ok": len(ok), "failed": len(errs),
        "prompt_tokens_mean": round(sum(prompt_toks) / len(prompt_toks)) if prompt_toks else None,
        "ttft_ms_mean": round(1000 * sum(ttfts) / len(ttfts), 1),
        "ttft_ms_max": round(1000 * max(ttfts), 1),
        "decode_tps_per_req_mean": round(sum(per_req_tps) / len(per_req_tps), 2) if per_req_tps else None,
        "decode_tps_aggregate": round(agg_tps, 2),
        "errors": errs[:3],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--ctx', type=int, nargs='+', default=[8000, 20000, 60000, 120000])
    ap.add_argument('--conc', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--out-tokens', type=int, default=256)
    ap.add_argument('--timeout', type=int, default=1200)
    ap.add_argument('--json', default='/home/max/qwopus-bench.json')
    args = ap.parse_args()

    print(f"# calibrating chars/token against {args.model} ...", flush=True)
    cpt = calibrate_ratio(args.base_url, args.model, args.timeout)
    print(f"# chars/token = {cpt:.3f}", flush=True)

    cells = []
    for ctx in args.ctx:
        for conc in args.conc:
            print(f"# running ctx~{ctx} conc={conc} ...", flush=True)
            r = run_cell(args.base_url, args.model, ctx, conc, args.out_tokens, cpt, args.timeout)
            cells.append(r)
            print("  " + json.dumps(r), flush=True)
            json.dump({"model": args.model, "cells": cells}, open(args.json, 'w'), indent=2)

    # markdown table
    print("\n## Results: decode tok/s (per-request mean / aggregate)\n")
    hdr = "| ctx (actual) | " + " | ".join(f"c{c}" for c in args.conc) + " |"
    print(hdr); print("|" + "---|" * (len(args.conc) + 1))
    by_ctx = {}
    for r in cells:
        by_ctx.setdefault(r['ctx_target'], {})[r['conc']] = r
    for ctx in args.ctx:
        actual = by_ctx[ctx].get(args.conc[0], {}).get('prompt_tokens_mean') or ctx
        row = [f"~{actual}"]
        for c in args.conc:
            r = by_ctx[ctx].get(c, {})
            pr = r.get('decode_tps_per_req_mean'); ag = r.get('decode_tps_aggregate')
            row.append(f"{pr} / {ag}" if pr is not None else "FAIL")
        print("| " + " | ".join(str(x) for x in row) + " |")
    print(f"\nFull JSON: {args.json}")

if __name__ == '__main__':
    main()
