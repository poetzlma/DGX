#!/usr/bin/env python3
"""Compare old flat benchmark results with new deep benchmark results."""

import json
import sys
from pathlib import Path

DEFAULT_OLD = "/home/max/llm-stack/logs/bench-results-old-20260414.json"
DEFAULT_NEW = "/home/max/llm-stack/logs/bench-deep-latest.json"


def load_json(path):
    p = Path(path)
    if p.is_symlink():
        p = p.parent / p.resolve().name
    return json.loads(p.read_text())


def fmt_delta(old_val, new_val, higher_is_better=True):
    if old_val is None or new_val is None:
        return f"{new_val:.1f}" if new_val is not None else "-"
    if old_val == 0:
        return f"- -> {new_val:.1f}"
    pct = (new_val - old_val) / old_val * 100
    arrow = "▲" if (pct > 0) == higher_is_better else "▼"
    sign = "+" if pct >= 0 else ""
    return f"{old_val:.1f} -> {new_val:.1f}  {sign}{pct:.0f}% {arrow}"


def main():
    old_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OLD
    new_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_NEW

    old_data = {r["model"]: r for r in load_json(old_path)}
    new_data = {r["model"]: r for r in load_json(new_path)}

    all_models = list(dict.fromkeys(list(old_data) + list(new_data)))

    # ---------------------------------------------------------------
    # TABLE 1: Before/After Overview
    # ---------------------------------------------------------------
    print("=" * 105)
    print("TABLE 1: Before/After Overview")
    print(f"  Old: {old_path}")
    print(f"  New: {new_path}")
    print("=" * 105)
    print(f"{'Model':<25} {'Cold Start (s)':<22} {'Memory (GB)':<22} "
          f"{'tok/s (short)':<24} {'TTFT (ms)'}")
    print("-" * 105)

    model_improvements = []
    for model in all_models:
        old = old_data.get(model, {})
        new = new_data.get(model, {})

        if new.get("status") != "ok":
            print(f"{model:<25} FAILED: {new.get('error', new.get('status', '?'))}")
            continue

        old_cold = old.get("cold_s")
        new_cold = new.get("cold_s")
        old_mem = old.get("model_gb")
        new_mem = new.get("model_gb")
        old_tps = old.get("tok_per_s")

        sp = new.get("prompts", {}).get("short", {})
        new_tps = sp.get("e2e_tok_s", {}).get("mean")
        ttft = sp.get("ttft_ms", {})
        ttft_str = (f"{ttft.get('mean', 0):.0f} +/- {ttft.get('std', 0):.0f}"
                    if ttft.get("mean") else "-")

        if old_tps and new_tps:
            model_improvements.append(
                (model, (new_tps - old_tps) / old_tps * 100))

        print(f"{model:<25} "
              f"{fmt_delta(old_cold, new_cold, False):<22} "
              f"{fmt_delta(old_mem, new_mem, False):<22} "
              f"{fmt_delta(old_tps, new_tps, True):<24} "
              f"{ttft_str}")

    if model_improvements:
        improved = sum(1 for _, pct in model_improvements if pct > 0)
        avg_imp = sum(pct for _, pct in model_improvements) / len(model_improvements)
        best_model, best_pct = max(model_improvements, key=lambda x: x[1])
        print("-" * 105)
        print(f"  {improved}/{len(model_improvements)} models improved tok/s. "
              f"Average change: {avg_imp:+.0f}%. "
              f"Biggest winner: {best_model} ({best_pct:+.0f}%)")

    # ---------------------------------------------------------------
    # TABLE 2: Per-Model Deep Profile
    # ---------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print("TABLE 2: Per-Model Deep Profile (new config)")
    print(f"{'='*90}")

    tier_labels = {
        "short": "short  (~30 tok prompt)",
        "medium": "medium (~500 tok prompt)",
        "long_prefill": "long   (~4K tok prompt)",
    }

    for model in all_models:
        new = new_data.get(model, {})
        if new.get("status") != "ok":
            continue
        prompts = new.get("prompts", {})
        print(f"\n  {model}")
        for tier in ["short", "medium", "long_prefill"]:
            p = prompts.get(tier, {})
            e2e = p.get("e2e_tok_s", {})
            dec = p.get("decode_tok_s", {})
            ttft = p.get("ttft_ms", {})
            comp = p.get("completion_tokens", {})
            print(f"    {tier_labels.get(tier, tier):<27} "
                  f"{e2e.get('mean', 0):>6.1f} +/- {e2e.get('std', 0):<5.1f} tok/s   "
                  f"TTFT: {ttft.get('mean', 0):>6.0f} +/- {ttft.get('std', 0):<5.0f} ms   "
                  f"decode: {dec.get('mean', 0):>6.1f} tok/s   "
                  f"({comp.get('mean', 0):.0f} tok)")

    # ---------------------------------------------------------------
    # TABLE 3: Concurrency Scaling
    # ---------------------------------------------------------------
    print(f"\n\n{'='*95}")
    print("TABLE 3: Concurrency Scaling (short prompt)")
    print(f"{'='*95}")
    print(f"{'Model':<25} {'conc=1 tok/s':>13} "
          f"{'conc=3 agg':>12} {'conc=5 agg':>12} {'Scales?'}")
    print("-" * 95)

    for model in all_models:
        new = new_data.get(model, {})
        if new.get("status") != "ok":
            continue
        sp = new.get("prompts", {}).get("short", {})
        c1 = sp.get("e2e_tok_s", {}).get("mean", 0)
        conc = new.get("concurrency", {})
        c3 = conc.get("3", {}).get("agg_tok_s", 0)
        c5 = conc.get("5", {}).get("agg_tok_s", 0)

        if c1 > 0 and c3 > 0:
            if c3 > c1 * 1.5:
                scales = "Yes"
            elif c3 < c1 * 1.2:
                scales = "No (serial)"
            else:
                scales = "Partial"
        else:
            scales = "-"

        print(f"{model:<25} {c1:>13.1f} {c3:>12.1f} {c5:>12.1f} {scales}")

    # Concurrency detail
    print(f"\n  Detail:")
    for model in all_models:
        new = new_data.get(model, {})
        if new.get("status") != "ok":
            continue
        conc = new.get("concurrency", {})
        if not conc:
            continue
        print(f"\n  {model}")
        for level in ["3", "5"]:
            c = conc.get(level, {})
            if "error" in c:
                print(f"    conc={level}: ERROR - {c['error']}")
            else:
                err_note = (f"   ({c.get('errors', 0)} errors)"
                            if c.get("errors") else "")
                print(f"    conc={level}: "
                      f"agg {c.get('agg_tok_s', 0):>6.1f} tok/s   "
                      f"per-req {c.get('mean_tok_s', 0):>6.1f} tok/s   "
                      f"TTFT {c.get('mean_ttft_ms', 0):>5.0f}ms   "
                      f"latency {c.get('mean_latency_s', 0):>5.1f}s   "
                      f"wall {c.get('wall_time_s', 0):>5.1f}s"
                      f"{err_note}")

    print()


if __name__ == "__main__":
    main()
