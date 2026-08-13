"""vLLM engine discovery + /metrics fetch + Prometheus-text parsing."""
import re
import subprocess
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Optional


_METRIC_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([0-9eE.+-]+)\s*$'
)
_LABEL_RE = re.compile(
    r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"'
)


def discover_engines() -> list[dict]:
    """Return [{name, port, status}] for every running vllm-* container."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format",
             "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    engines = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, status, ports = parts[0], parts[1], parts[2]
        if not name.startswith("vllm-"):
            continue
        m = re.search(r"0\.0\.0\.0:(\d+)->", ports)
        if not m:
            continue
        engines.append({"name": name, "port": int(m.group(1)),
                        "status": status})
    return engines


def parse_metrics(text: str) -> dict:
    out = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, labels_raw, val_raw = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = float(val_raw)
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(labels_raw))
        out[name].append((labels, val))
        # DwarfStar mainline exposes these counters under a neutral `ds4:`
        # namespace — upstream would not take a vllm:-branded one. Alias them
        # onto the vllm: names so every consumer here keeps working unchanged,
        # and so the real vLLM engines (which still emit vllm:) and the ds4
        # lane can be read by the same code during the cutover.
        if name.startswith("ds4:"):
            out["vllm:" + name[len("ds4:"):]].append((labels, val))
    return out


def fetch_metrics(port: int, timeout: float = 2.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def first_value(metrics: dict, name: str, **filt) -> Optional[float]:
    for labels, val in metrics.get(name, []):
        if all(labels.get(k) == v for k, v in filt.items()):
            return val
    return None


def sum_values(metrics: dict, name: str, **filt) -> float:
    s = 0.0
    for labels, val in metrics.get(name, []):
        if all(labels.get(k) == v for k, v in filt.items()):
            s += val
    return s


def gauge_snapshot(metrics: dict) -> dict:
    return {
        "running": first_value(metrics, "vllm:num_requests_running") or 0,
        "waiting": first_value(metrics, "vllm:num_requests_waiting") or 0,
        "kv": (first_value(metrics, "vllm:kv_cache_usage_perc") or 0) * 100,
        "awake": first_value(metrics, "vllm:engine_sleep_state",
                             sleep_state="awake") or 0,
        "spec_drafted": sum_values(
            metrics, "vllm:spec_decode_num_draft_tokens_total"),
        "spec_accepted": sum_values(
            metrics, "vllm:spec_decode_num_accepted_tokens_total"),
        "preempt": sum_values(metrics, "vllm:num_preemptions_total"),
        "succ_stop": sum_values(metrics, "vllm:request_success_total",
                                finished_reason="stop"),
        "succ_length": sum_values(metrics, "vllm:request_success_total",
                                  finished_reason="length"),
        "succ_abort": sum_values(metrics, "vllm:request_success_total",
                                 finished_reason="abort"),
        "succ_error": sum_values(metrics, "vllm:request_success_total",
                                 finished_reason="error"),
        "gen_total": sum_values(metrics, "vllm:generation_tokens_total"),
        "prompt_total": sum_values(metrics, "vllm:prompt_tokens_total"),
        "ttft_count": sum_values(metrics,
                                 "vllm:time_to_first_token_seconds_count"),
        "ttft_sum": sum_values(metrics,
                               "vllm:time_to_first_token_seconds_sum"),
        "tpot_count": sum_values(metrics,
                                 "vllm:time_per_output_token_seconds_count"),
        "tpot_sum": sum_values(metrics,
                               "vllm:time_per_output_token_seconds_sum"),
        "e2e_count": sum_values(metrics,
                                "vllm:e2e_request_latency_seconds_count"),
        "e2e_sum": sum_values(metrics,
                              "vllm:e2e_request_latency_seconds_sum"),
    }


# Default-zero snapshot used when /metrics fetch fails (engine warming up,
# port not yet bound, etc.). Renderers / serializers can read it without
# special-casing the None branch.
EMPTY_SNAP = {
    "running": 0, "waiting": 0, "kv": 0, "awake": 1,
    "spec_drafted": 0, "spec_accepted": 0, "preempt": 0,
    "succ_stop": 0, "succ_length": 0, "succ_abort": 0,
    "succ_error": 0, "gen_total": 0, "prompt_total": 0,
    "ttft_count": 0, "ttft_sum": 0.0,
    "tpot_count": 0, "tpot_sum": 0.0,
    "e2e_count": 0, "e2e_sum": 0.0,
}
