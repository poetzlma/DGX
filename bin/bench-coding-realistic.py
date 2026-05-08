#!/home/max/llm-stack/venv/bin/python3
"""Realistic coding A/B bench centered on the user's actual bell-curve.

Three buckets, c=1 each, single fire per bucket:
    13k input  -> max 1024 output  (start of session)
    60k input  -> max 1500 output  (bell-curve middle)
    100k input -> max 2048 output  (long-running session)

For each bucket captures: TTFT, decode tok/s, prompt/completion tokens, total_s,
plus spec-decode acceptance metrics scraped from the model's vLLM container log
within the bench window.

Usage:
    bench-coding-realistic.py MODEL CONTAINER [LABEL]

Defaults LABEL = MODEL. Output JSON: /tmp/bench-coding-LABEL-TIMESTAMP.json
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import import_module

sys.path.insert(0, "/home/max/llm-stack/bin")
m = import_module("bench-deep")

# ---- code-shaped filler, ~500 tokens per repeat ---------------------------
_CODE_FILLER = '''
class RequestProcessor:
    """Handles incoming HTTP requests with retry, circuit-break, and tracing."""

    def __init__(self, registry: ServiceRegistry, breaker: CircuitBreaker,
                 tracer: Tracer, retry: RetryPolicy):
        self.registry = registry
        self.breaker = breaker
        self.tracer = tracer
        self.retry = retry
        self._semaphore = asyncio.Semaphore(64)
        self._inflight: dict[str, asyncio.Task] = {}

    async def process(self, request: Request) -> Response:
        span = self.tracer.start_span("process_request",
                                       attributes={"path": request.path,
                                                   "method": request.method})
        try:
            target = self.registry.resolve(request.path)
            if target is None:
                return Response(status=404, body={"error": "not found"})
            async with self._semaphore:
                if not self.breaker.allow(target.name):
                    return Response(status=503, body={"error": "circuit open"})
                async for attempt in self.retry.attempts():
                    try:
                        upstream = await self._call_upstream(target, request)
                        if upstream.status < 500:
                            self.breaker.record_success(target.name)
                            return upstream
                        attempt.failure(upstream.status)
                    except (TimeoutError, ConnectionError) as e:
                        attempt.failure(str(e))
                        self.breaker.record_failure(target.name)
                return Response(status=502, body={"error": "upstream failed"})
        finally:
            span.finish()

    async def _call_upstream(self, target: Service, request: Request) -> Response:
        body = await request.body()
        async with httpx.AsyncClient(timeout=target.timeout) as client:
            r = await client.request(request.method, target.url + request.path,
                                      headers=self._strip_hop_headers(request.headers),
                                      content=body)
        return Response(status=r.status_code, body=r.content,
                         headers=dict(r.headers))

    @staticmethod
    def _strip_hop_headers(headers: Headers) -> dict[str, str]:
        hop = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade"}
        return {k: v for k, v in headers.items() if k.lower() not in hop}
'''

# Per-bucket task instructions. Same task ⇒ deterministic comparison across configs.
TASK_BY_BUCKET = {
    "13k": (
        "\n\nYou are reviewing the codebase above. Identify the THREE most likely "
        "bugs or subtle correctness issues. For each, cite the function, explain "
        "the failure mode, and propose a one-paragraph fix. Be specific."
    ),
    "60k": (
        "\n\nYou are extending the system above with a new feature: per-tenant "
        "rate limiting that survives process restarts. Design the data model, "
        "the integration point in the request pipeline, and the failure-recovery "
        "behavior. Produce concrete Python code (≥40 lines) for the core "
        "RateLimiter class, then explain how it integrates with RequestProcessor."
    ),
    "100k": (
        "\n\nYou are migrating the system above from blocking httpx to a "
        "connection-pooled gRPC client while keeping the public HTTP API "
        "unchanged. Plan the migration in 4 phases (interface compatibility, "
        "client pool, timeout/retry parity, observability). For each phase, "
        "produce concrete Python code (combined ≥80 lines) showing the key "
        "changes, then list 3 risks and how you'd mitigate each."
    ),
}

# Approximate tokens per filler repeat (Qwen tokenizer, code-shaped).
# Measured against vLLM at request time on 2026-05-08 — the _CODE_FILLER
# block tokenizes to ~446 tokens with Qwen3.6 BPE (snake_case + dot-paths
# split aggressively, hence higher than typical prose ratio). Final prompt
# size is reported by vLLM at request time; this constant only sizes the
# filler repetition.
TOKENS_PER_FILLER = 450

BUCKETS = [
    {"name": "13k",  "target_input": 13000,  "max_tokens": 1024},
    {"name": "60k",  "target_input": 60000,  "max_tokens": 1500},
    {"name": "100k", "target_input": 100000, "max_tokens": 2048},
]


def make_prompt(bucket: dict) -> str:
    repeats = max(1, (bucket["target_input"] - 200) // TOKENS_PER_FILLER)
    nonce = f"// bench-nonce {time.time_ns()}-{bucket['name']}\n"
    body = nonce + (_CODE_FILLER * repeats)
    task = TASK_BY_BUCKET[bucket["name"]]
    return body + task


def detect_acceptance(log_text: str) -> dict:
    """Average and per-position acceptance from a chunk of vLLM container log."""
    avg_rates = re.findall(r"Avg Draft acceptance rate:\s*([0-9.]+)%", log_text)
    accepted_lengths = re.findall(r"Mean acceptance length:\s*([0-9.]+)", log_text)
    pos_blocks = re.findall(
        r"Per-position acceptance rate:\s*([0-9., ]+?),\s*Avg Draft", log_text
    )
    decode_throughput = re.findall(
        r"generation throughput:\s*([0-9.]+)\s*tokens/s,\s*Running:\s*1", log_text
    )

    pos_sums = [0.0] * 15
    pos_counts = [0] * 15
    for block in pos_blocks:
        vals = [float(x.strip()) for x in block.split(",") if x.strip()]
        for i, v in enumerate(vals[:15]):
            pos_sums[i] += v
            pos_counts[i] += 1

    per_pos = [
        round(pos_sums[i] / pos_counts[i], 3) if pos_counts[i] else None
        for i in range(15)
    ]

    decode_vals = [float(x) for x in decode_throughput]
    decode_vals_sorted = sorted(decode_vals)
    decode_stats = {}
    if decode_vals_sorted:
        n = len(decode_vals_sorted)
        decode_stats = {
            "samples": n,
            "mean": round(sum(decode_vals_sorted) / n, 1),
            "median": round(decode_vals_sorted[n // 2], 1),
            "p90": round(decode_vals_sorted[int(n * 0.9)], 1),
            "max": round(decode_vals_sorted[-1], 1),
        }

    return {
        "n_accept_samples": len(avg_rates),
        "mean_accept_pct": round(sum(float(x) for x in avg_rates) / len(avg_rates), 2)
            if avg_rates else None,
        "median_accept_pct": round(sorted(float(x) for x in avg_rates)[len(avg_rates) // 2], 2)
            if avg_rates else None,
        "max_accept_pct": round(max(float(x) for x in avg_rates), 2) if avg_rates else None,
        "mean_accept_length": round(
            sum(float(x) for x in accepted_lengths) / len(accepted_lengths), 2
        ) if accepted_lengths else None,
        "per_position": per_pos,
        "engine_decode_throughput": decode_stats,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: bench-coding-realistic.py MODEL CONTAINER [LABEL]", file=sys.stderr)
        sys.exit(2)

    model = sys.argv[1]
    container = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else model.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = f"/home/max/llm-stack/logs/bench-coding-{label}-{ts}.json"
    log_path = f"/tmp/bench-coding-{label}-{ts}.vllm.log"

    print(f"=== bench-coding-realistic ===")
    print(f"  model:     {model}")
    print(f"  container: {container}")
    print(f"  label:     {label}")
    print(f"  out:       {out_path}")
    print(f"  vllm log:  {log_path}")
    print()

    # Start tailing container log to a file (background subprocess).
    log_f = open(log_path, "w")
    tail_proc = subprocess.Popen(
        ["docker", "logs", "-f", "--since", "0s", container],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )

    bench_t0 = time.time()
    results = []
    try:
        for bucket in BUCKETS:
            prompt_text = make_prompt(bucket)
            messages = [{"role": "user", "content": prompt_text}]

            print(f"[{bucket['name']}] firing  target={bucket['target_input']}  "
                  f"prompt_chars={len(prompt_text):,}  max_tokens={bucket['max_tokens']}")
            t_start = time.time()
            try:
                r = m.chat_streaming(model, messages, bucket["max_tokens"], timeout=900)
                r["error"] = None
            except Exception as e:
                r = {"error": str(e)[:300]}
            t_end = time.time()

            r["bucket"] = bucket["name"]
            r["target_input"] = bucket["target_input"]
            r["max_tokens_requested"] = bucket["max_tokens"]
            r["window_start_unix"] = t_start
            r["window_end_unix"] = t_end
            results.append(r)

            if r.get("error"):
                print(f"  ERROR: {r['error']}")
            else:
                print(
                    f"  ok  ttft={r['ttft_ms']:.0f}ms  decode={r['decode_tok_s']}t/s  "
                    f"out={r['completion_tokens']}tok  total={r['total_s']}s"
                )
            # tiny gap to let metrics flush
            time.sleep(4)
    finally:
        # Drain log for a couple seconds so the last metrics line lands.
        time.sleep(3)
        tail_proc.terminate()
        try:
            tail_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tail_proc.kill()
        log_f.close()

    # Parse acceptance per-bucket by slicing the vLLM log on its
    # "INFO MM-DD HH:MM:SS" timestamps against each bucket's window.
    full_log_lines = open(log_path).readlines()
    line_re = re.compile(r"^.*?INFO (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
    # Build (line_unix_ts, line_str) tuples for lines we can timestamp.
    bench_year = datetime.now(timezone.utc).year
    timed_lines = []
    for line in full_log_lines:
        mt = line_re.match(line)
        if not mt:
            continue
        try:
            mn, dy, hr, mi, sc = (int(x) for x in mt.groups())
            dt = datetime(bench_year, mn, dy, hr, mi, sc, tzinfo=timezone.utc)
            timed_lines.append((dt.timestamp(), line))
        except ValueError:
            continue

    for r in results:
        ws, we = r["window_start_unix"], r["window_end_unix"]
        # Generous +5s pad to capture metrics emitted just after request close.
        section_lines = [ln for ts, ln in timed_lines if ws - 1 <= ts <= we + 5]
        r["spec_metrics"] = detect_acceptance("".join(section_lines))
        r["spec_metrics"]["log_lines_in_window"] = len(section_lines)

    bench_total = time.time() - bench_t0
    payload = {
        "model": model,
        "label": label,
        "container": container,
        "timestamp_utc": ts,
        "bench_total_s": round(bench_total, 1),
        "buckets": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    print(f"=== summary ({label}) ===")
    print(f"{'bucket':>7}  {'ttft':>7}  {'decode':>7}  {'out_tok':>7}  "
          f"{'total':>6}  {'accept':>7}  {'acc_len':>7}")
    for r in results:
        if r.get("error"):
            print(f"{r['bucket']:>7}  ERROR: {r['error'][:60]}")
            continue
        sp = r.get("spec_metrics") or {}
        print(
            f"{r['bucket']:>7}  "
            f"{r['ttft_ms']:>5.0f}ms  "
            f"{r['decode_tok_s']:>5.1f}t/s  "
            f"{r['completion_tokens']:>7}  "
            f"{r['total_s']:>5.1f}s  "
            f"{(sp.get('mean_accept_pct') or 0):>6.1f}%  "
            f"{(sp.get('mean_accept_length') or 0):>6.2f}"
        )
    print()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
