#!/home/max/llm-stack/venv/bin/python3
"""
Quality A/B for the 27B coding slot.

Runs a fixed prompt suite against:
  - qwen3.6-27b      (prod: AEON-7 abliterated NVFP4 + DFlash drafter)
  - qwen3.6-27b-mtp  (dormant: AlphaOxO clean NVFP4 + MTP)

Both live in llama-swap's `main` exclusive group, so this WILL evict the
currently-loaded model and incur cold-start cost. See notes in README.

Usage:
    bench-quality-27b.py --prompts ~/prompts.json [--out report.md]

Prompt file format (JSON list):
    [
      {"name": "task1", "system": "optional system prompt",
       "user": "the user message", "max_tokens": 1024},
      ...
    ]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

GATEWAY = "http://192.168.1.12:8080"
MODELS = ["qwen3.6-27b", "qwen3.6-27b-mtp"]
COLD_TIMEOUT_S = 1200  # cold start can be ~10 min for DFlash autotuner
WARM_TIMEOUT_S = 600


async def stream_chat(session, model, prompt, timeout):
    messages = []
    if prompt.get("system"):
        messages.append({"role": "system", "content": prompt["system"]})
    messages.append({"role": "user", "content": prompt["user"]})

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": prompt.get("max_tokens", 1024),
        "temperature": prompt.get("temperature", 0.0),
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    chunks = []
    usage = None

    async with session.post(
        GATEWAY + "/v1/chat/completions",
        json=body,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks.append(delta)

    t_end = time.perf_counter()
    text = "".join(chunks)
    return {
        "text": text,
        "ttft_s": (t_first - t_start) if t_first else None,
        "total_s": t_end - t_start,
        "usage": usage,
    }


async def run_model(model, prompts, is_first):
    """Run all prompts against one model. is_first → use cold timeout for first call."""
    print(f"\n=== {model} ===", flush=True)
    results = []
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts):
            timeout = COLD_TIMEOUT_S if (is_first and i == 0) else WARM_TIMEOUT_S
            label = "(cold)" if (is_first and i == 0) else ""
            print(f"  [{i+1}/{len(prompts)}] {prompt['name']} {label}", flush=True)
            try:
                t0 = time.perf_counter()
                r = await stream_chat(session, model, prompt, timeout)
                wall = time.perf_counter() - t0
                completion_toks = (r["usage"] or {}).get("completion_tokens")
                tok_s = (
                    completion_toks / (r["total_s"] - (r["ttft_s"] or 0))
                    if completion_toks and r["ttft_s"] and r["total_s"] > r["ttft_s"]
                    else None
                )
                print(
                    f"      ttft={r['ttft_s']:.2f}s  total={r['total_s']:.2f}s  "
                    f"out_toks={completion_toks}  decode_tok/s={tok_s:.1f}"
                    if tok_s
                    else f"      ttft={r['ttft_s']}  total={r['total_s']:.2f}s  "
                         f"out_toks={completion_toks}",
                    flush=True,
                )
                results.append({"prompt": prompt, "ok": True, **r,
                                "decode_tok_s": tok_s, "wall_s": wall})
            except Exception as e:
                print(f"      ERROR: {e}", flush=True)
                results.append({"prompt": prompt, "ok": False, "error": str(e)})
    return results


MODEL_LABELS = {
    "qwen3.6-27b": "AEON-7 abliterated NVFP4 + DFlash drafter (current prod)",
    "qwen3.6-27b-mtp": "AlphaOxO clean NVFP4 + MTP (dormant rollback)",
}


def write_report(out_path, all_results):
    models = list(all_results.keys())
    lines = [f"# 27B Quality A/B — {datetime.now().isoformat(timespec='seconds')}\n"]
    lines.append("Models:\n")
    for m in models:
        lines.append(f"- **{m}** = {MODEL_LABELS.get(m, '(custom)')}\n")

    n_prompts = len(all_results[models[0]])
    for i in range(n_prompts):
        first = all_results[models[0]][i]
        name = first["prompt"]["name"]
        lines.append(f"\n---\n\n## Prompt: {name}\n")
        lines.append("### Input\n")
        if first["prompt"].get("system"):
            lines.append(f"_system_: {first['prompt']['system']}\n")
        lines.append(f"```\n{first['prompt']['user']}\n```\n")

        for m in models:
            res = all_results[m][i]
            lines.append(f"\n### {m}\n")
            if not res.get("ok"):
                lines.append(f"**ERROR:** {res.get('error')}\n")
                continue
            usage = res.get("usage") or {}
            if res.get("decode_tok_s"):
                lines.append(
                    f"- ttft: {res['ttft_s']:.2f}s · total: {res['total_s']:.2f}s · "
                    f"out_toks: {usage.get('completion_tokens')} · "
                    f"decode_tok/s: {res['decode_tok_s']:.1f}\n"
                )
            else:
                lines.append(
                    f"- total: {res['total_s']:.2f}s · "
                    f"out_toks: {usage.get('completion_tokens')}\n"
                )
            lines.append(f"\n```\n{res['text']}\n```\n")

    Path(out_path).write_text("".join(lines))
    print(f"\nReport written to {out_path}", flush=True)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True, help="JSON file with prompt list")
    p.add_argument("--out", default=None, help="Output markdown path")
    p.add_argument("--json-out", default=None, help="Optional raw JSON dump")
    p.add_argument("--models", nargs="+", default=MODELS,
                   help="Override the list of models to bench")
    p.add_argument("--rewarm", default="qwen3.6-27b",
                   help="Model to warm at the end (default: qwen3.6-27b prod). "
                        "Set to empty string to skip.")
    args = p.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    print(f"Loaded {len(prompts)} prompt(s) from {args.prompts}", flush=True)
    print(f"Bench order: {' -> '.join(args.models)}", flush=True)
    print("First call against each model uses cold timeout (1200s).", flush=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_out = args.json_out or f"/home/max/llm-stack/logs/bench-quality-27b-{ts}.json"

    all_results = {}
    for idx, model in enumerate(args.models):
        all_results[model] = await run_model(model, prompts, is_first=True)
        # Persist after every model so a later crash never loses completions.
        Path(json_out).write_text(json.dumps(all_results, indent=2, default=str))
        if idx < len(args.models) - 1:
            await asyncio.sleep(2)

    print(f"Raw JSON written to {json_out}", flush=True)
    out = args.out or f"/home/max/llm-stack/logs/bench-quality-27b-{ts}.md"
    try:
        write_report(out, all_results)
    except Exception as e:
        print(f"WARN: write_report failed: {e}. JSON dump is at {json_out}", flush=True)

    # Warm the production model back up so the next customer request is hot.
    if args.rewarm and args.rewarm not in args.models[-1:]:
        print(f"\n=== Re-warming {args.rewarm} for prod ===", flush=True)
        warm_prompt = {"name": "warmup", "user": "ping", "max_tokens": 1}
        async with aiohttp.ClientSession() as session:
            try:
                await stream_chat(session, args.rewarm, warm_prompt, COLD_TIMEOUT_S)
                print(f"  {args.rewarm} hot.", flush=True)
            except Exception as e:
                print(f"  WARN: re-warm failed: {e}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
