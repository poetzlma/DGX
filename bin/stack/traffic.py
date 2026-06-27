"""TrafficMonitor: rolling window of log-proxy meta.json files.

Polls the date/model-bucketed proxy log tree
(`/home/max/llm-stack/logs/proxy/{YYYY-MM-DD}/{model}/*.meta.json`),
maintains an in-memory rolling deque of recent requests, and computes:

- request rate (rps)
- latency percentiles: TTFT, E2E (and a few derived rates)
- malformed-task signatures with per-signature counts
- recent failures with a one-line excerpt of the offending field
- per-key roll-up (top-N by request count, with error %)
- per-tool roll-up (calls, arg_parse_errors)

The "active engine" notion is heuristic: the most recent `request_model`
field in the window. With llama-swap's exclusive group this maps to the
currently-loaded engine.
"""
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional
from urllib.request import urlopen
from urllib.error import URLError

LOG_PROXY_BASE = "/home/max/llm-stack/logs/proxy"
LLAMA_SWAP_YAML = "/home/max/llm-stack/config/llama-swap.yaml"


def _parse_swap_ports(path: str) -> dict:
    """Read llama-swap.yaml and build {model_name: port}.

    We grep for `model:` headings and the first `proxy: http://...:PORT` that
    follows. Robust to indent / comment lines. Done without pyyaml to keep the
    monitor zero-deps beyond what's already installed.
    """
    out: dict = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return out
    cur = None
    in_models = False
    for ln in lines:
        s = ln.rstrip()
        # The models block in llama-swap is `models:` at top level.
        if re.match(r"^models:\s*$", s):
            in_models = True
            continue
        # Other top-level blocks end the models section.
        if in_models and re.match(r"^[a-z][a-zA-Z0-9_]*:\s*$", s):
            in_models = False
            continue
        if not in_models:
            continue
        m = re.match(r"^\s{2}([a-zA-Z0-9._-]+):\s*$", s)
        if m:
            cur = m.group(1)
            continue
        if cur:
            m = re.search(r"proxy:\s*https?://[^:]+:(\d+)", s)
            if m:
                out[cur] = int(m.group(1))
                cur = None
    return out


_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<val>[-+0-9.eE]+|NaN)"
)


def _parse_prom(text: str) -> dict:
    """Minimal Prometheus text-format parser. Returns:
        - "{name}"                  -> float, if no labels
        - "{name}{label=value}"     -> float, if labeled (full original form)
    Plus convenience keys for vllm: counters with one label (e.g.
    `vllm:request_success_total{finished_reason="stop"}` becomes available
    both as the full string and as `vllm:request_success_total[stop]`).
    """
    out: dict = {}
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        m = _PROM_LINE.match(ln)
        if not m:
            continue
        name, labels, val = m.group("name"), m.group("labels"), m.group("val")
        try:
            v = float(val)
        except ValueError:
            continue
        if labels:
            full = f"{name}{{{labels}}}"
            out[full] = v
            # Convenience: extract single label value for sugar keys.
            simple = re.match(r'^(\w+)="([^"]*)"$', labels)
            if simple:
                out[f"{name}[{simple.group(2)}]"] = v
        else:
            out[name] = v
    return out


class TrafficMonitor:
    def __init__(self, log_dir: str = LOG_PROXY_BASE, max_keep: int = 20000):
        self.log_dir = log_dir
        self.max_keep = max_keep
        self.lock = threading.RLock()
        # Each entry is the parsed meta dict, with two added fields:
        #   _ts_unix     – parsed UTC timestamp from the `ts` field
        #   _signatures  – list of malformed-task signature strings
        self.entries = deque(maxlen=max_keep)
        self._seen: set = set()
        # Engine-side state caches (refreshed by tick()).
        self._engine_ports = _parse_swap_ports(LLAMA_SWAP_YAML)
        self._engine_cache: dict = {}      # model -> last metrics dict
        self._engine_cache_ts = 0.0
        self._gpu_cache: dict = {}
        self._gpu_cache_ts = 0.0

    # ────────────────────────────────────────────────────────── parse / scan

    @staticmethod
    def _parse_ts(ts_str: str) -> float:
        try:
            return (
                datetime.strptime(ts_str[:15], "%Y%m%d-%H%M%S")
                .replace(tzinfo=timezone.utc).timestamp()
            )
        except (ValueError, TypeError):
            return time.time()

    def scan(self) -> int:
        """Discover new meta.json files and append to the rolling window.

        Walks today and yesterday (for the midnight crossover) under the
        date/model bucket layout. Files are de-duplicated by absolute path
        in ``self._seen``. Returns the count of newly-loaded entries.
        """
        if not os.path.isdir(self.log_dir):
            return 0
        today = date.today()
        days = (today - timedelta(days=1), today)
        new_files = []
        for d in days:
            day_dir = os.path.join(self.log_dir, d.isoformat())
            if not os.path.isdir(day_dir):
                continue
            try:
                for m_entry in os.scandir(day_dir):
                    if not m_entry.is_dir():
                        continue
                    try:
                        for f in os.scandir(m_entry.path):
                            if not f.name.endswith(".meta.json"):
                                continue
                            if f.path in self._seen:
                                continue
                            new_files.append(f.path)
                    except OSError:
                        continue
            except OSError:
                continue

        if not new_files:
            return 0
        new_files.sort()  # timestamp-prefixed names ⇒ roughly chronological
        added = 0
        for path in new_files:
            self._seen.add(path)
            try:
                with open(path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            meta["_ts_unix"] = self._parse_ts(meta.get("ts", ""))
            meta["_signatures"] = list(self._signatures(meta))
            with self.lock:
                self.entries.append(meta)
            added += 1
        return added

    # ─────────────────────────────────────────────────── signature detection

    @staticmethod
    def _signatures(meta: dict) -> Iterable[str]:
        """Yield malformed-task signature strings present in this meta entry.

        Order is not significant; counts are aggregated by name elsewhere.
        Each signature corresponds to a heuristic the operator wants flagged.
        """
        status = meta.get("status") or 0
        if status >= 500:
            yield "status_5xx"
        elif status >= 400:
            yield "status_4xx"

        if meta.get("tool_call_arg_errors", 0):
            yield "arg_parse_error"

        tools_in = {t for t in (meta.get("tool_names") or []) if t}
        for tc in meta.get("tool_calls") or []:
            name = ((tc.get("function") or {}).get("name") or "")
            if tools_in and name and name not in tools_in:
                yield "hallucinated_tool"
                break

        tc_count = meta.get("tool_call_count", 0) or 0
        rc = meta.get("reasoning_chars", 0) or 0
        cc = meta.get("completion_chars", 0) or 0
        fr = meta.get("finish_reason")

        if tc_count == 0 and rc > 0 and cc == 0:
            yield "reasoning_only"
        if tc_count == 0 and rc == 0 and cc == 0 and (status or 200) == 200:
            yield "empty_response"
        if fr == "length":
            yield "length_truncated"
        if fr == "tool_calls" and tc_count == 0:
            yield "finish_inconsistent"
        if fr in ("error", "abort"):
            yield "engine_error"

        if meta.get("aborted") == "client":
            yield "client_abort"
        elif meta.get("aborted") == "upstream":
            yield "upstream_abort"

        pt = meta.get("prompt_tokens") or 0
        ct = meta.get("completion_tokens") or 0
        if pt > 50000 and ct < 5 and fr not in ("tool_calls", "stop"):
            yield "giant_prompt_no_progress"

        if meta.get("n_messages") == 0:
            yield "no_messages"

    # ─────────────────────────────────────────────────────────── views

    def window(self, seconds: int = 300) -> list:
        """List of entries whose ts is within last `seconds`."""
        cutoff = time.time() - seconds
        with self.lock:
            return [e for e in self.entries if e.get("_ts_unix", 0) >= cutoff]

    def rps(self, seconds: int = 60) -> float:
        return len(self.window(seconds)) / seconds if seconds else 0.0

    def percentiles(self, seconds: int, key: str,
                    ps=(50, 95, 99)) -> dict:
        vals = sorted(
            e.get(key) for e in self.window(seconds) if e.get(key) is not None
        )
        if not vals:
            return {p: None for p in ps}
        out = {}
        for p in ps:
            idx = max(0, min(len(vals) - 1,
                             int(round(p / 100.0 * (len(vals) - 1)))))
            out[p] = vals[idx]
        return out

    def malformed_counts(self, seconds: int = 300) -> Counter:
        c: Counter = Counter()
        for e in self.window(seconds):
            for s in e.get("_signatures") or []:
                c[s] += 1
        return c

    def recent_failures(self, n: int = 6,
                        seconds: Optional[int] = None) -> list:
        """Newest-first failures. If `seconds` is given, restrict to that
        window — otherwise scan the entire keep buffer."""
        if seconds:
            es = [e for e in self.window(seconds) if e.get("_signatures")]
        else:
            with self.lock:
                es = [e for e in self.entries if e.get("_signatures")]
        es.sort(key=lambda e: e.get("_ts_unix", 0), reverse=True)
        return es[:n]

    def per_key(self, seconds: int = 300, n: int = 8) -> list:
        win = self.window(seconds)
        agg: dict = defaultdict(lambda: {"req": 0, "err": 0, "tokens": 0})
        for e in win:
            k = e.get("key_masked") or e.get("key_prefix") or "?"
            agg[k]["req"] += 1
            if e.get("_signatures"):
                agg[k]["err"] += 1
            agg[k]["tokens"] += (
                (e.get("prompt_tokens") or 0)
                + (e.get("completion_tokens") or 0)
            )
        rows = []
        for k, v in agg.items():
            req = v["req"]
            rows.append({
                "key": k,
                "req": req,
                "rps": req / seconds if seconds else 0,
                "tokens_per_min": v["tokens"] * 60 / seconds if seconds else 0,
                "err_pct": (v["err"] / req * 100) if req else 0,
            })
        rows.sort(key=lambda r: r["req"], reverse=True)
        return rows[:n]

    def tool_calls(self, seconds: int = 300) -> list:
        win = self.window(seconds)
        agg: dict = defaultdict(lambda: {"calls": 0, "arg_err": 0})
        for e in win:
            for tc in e.get("tool_calls") or []:
                name = ((tc.get("function") or {}).get("name") or "?")
                agg[name]["calls"] += 1
                if tc.get("_args_parse_error"):
                    agg[name]["arg_err"] += 1
        return sorted(
            [{"name": k, **v} for k, v in agg.items()],
            key=lambda r: r["calls"], reverse=True,
        )

    def latency_summary(self, seconds: int = 300) -> dict:
        """Window summary. Throughput rates are token-weighted (sum/sum),
        not mean-of-ratios — otherwise a 5-token warmup with 0.05s ttft
        skews the prefill average into the thousands t/s."""
        win = self.window(seconds)
        ttfts, e2es = [], []
        pt_sum = ttft_sum = ct_sum = decode_time_sum = 0.0
        for e in win:
            ttft = e.get("ttft_s")
            elapsed = e.get("elapsed_s")
            pt = e.get("prompt_tokens") or 0
            ct = e.get("completion_tokens") or 0
            if ttft is not None:
                ttfts.append(ttft)
            if elapsed is not None:
                e2es.append(elapsed)
            # Token-weighted: only count "real" prompts (>= 50 toks) for
            # prefill rate, otherwise we measure mostly request overhead.
            if ttft and ttft > 0 and pt >= 50:
                pt_sum += pt
                ttft_sum += ttft
            if elapsed is not None and ttft is not None and ct > 0 \
               and (elapsed - ttft) > 0:
                ct_sum += ct
                decode_time_sum += (elapsed - ttft)
        avg = lambda L: sum(L) / len(L) if L else None
        return {
            "ttft_avg": avg(ttfts),
            "e2e_avg": avg(e2es),
            "decode_tps_avg": (ct_sum / decode_time_sum) if decode_time_sum else None,
            "prefill_tps_avg": (pt_sum / ttft_sum) if ttft_sum else None,
        }

    # ───────────────────────────────────────────────────── time-bucketed series
    def time_series(self, seconds: int, buckets: int = 30,
                    key: Optional[str] = None,
                    agg: str = "count") -> list:
        """Return `buckets` values spanning the last `seconds`.

        - agg='count' : count of entries per bucket (rate)
        - agg='mean'  : mean of meta[key] in each bucket (None if empty)
        - agg='p99'   : p99 of meta[key] (None if empty)
        """
        now = time.time()
        win_start = now - seconds
        bucket_w = seconds / buckets
        bins: list = [[] for _ in range(buckets)]
        with self.lock:
            for e in self.entries:
                t = e.get("_ts_unix", 0)
                if t < win_start or t >= now:
                    continue
                idx = min(buckets - 1, int((t - win_start) / bucket_w))
                if key is None:
                    bins[idx].append(1)
                else:
                    v = e.get(key)
                    if v is not None:
                        bins[idx].append(v)
        out: list = []
        for b in bins:
            if not b:
                out.append(None if agg != "count" else 0)
            elif agg == "count":
                out.append(len(b))
            elif agg == "mean":
                out.append(sum(b) / len(b))
            elif agg == "p99":
                s = sorted(b)
                idx = max(0, min(len(s) - 1,
                                 int(round(0.99 * (len(s) - 1)))))
                out.append(s[idx])
            else:
                out.append(None)
        return out

    def spike_detected(self, short_s: int = 60, long_s: int = 300,
                       factor: float = 2.0) -> bool:
        """True if the short-window rps exceeds factor × the long-window rps,
        AND both windows have meaningful data."""
        s = len(self.window(short_s)) / short_s if short_s else 0
        L = len(self.window(long_s)) / long_s if long_s else 0
        if L < 0.01:  # too quiet to call anything a spike
            return False
        return s > factor * L

    # ───────────────────────────────────────────────────── tool-arg distribution
    def tool_arg_sizes(self, seconds: int = 300) -> dict:
        """{tool_name: [arg_len, arg_len, ...]} over window."""
        win = self.window(seconds)
        out: dict = defaultdict(list)
        for e in win:
            for tc in e.get("tool_calls") or []:
                name = ((tc.get("function") or {}).get("name") or "?")
                args = ((tc.get("function") or {}).get("arguments") or "")
                out[name].append(len(args))
        return dict(out)

    # ───────────────────────────────────────────────────── engine /metrics join
    def refresh_engine_state(self, max_age_s: float = 1.0) -> None:
        """Fetch /metrics for the currently-active engine. Cheap (local
        loopback HTTP). Caches results between calls so the TUI render loop
        can call this every frame without slamming the engine."""
        now = time.time()
        if now - self._engine_cache_ts < max_age_s:
            return
        model = self.active_engine()
        if not model:
            return
        port = self._engine_ports.get(model)
        if not port:
            return
        try:
            with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=0.5) as r:
                text = r.read().decode("utf-8", "ignore")
        except (URLError, TimeoutError, OSError):
            return
        parsed = _parse_prom(text)
        prev = self._engine_cache.get(model) or {}
        # rate fields (per-second since last sample)
        dt = now - (prev.get("_t") or now)
        for c in ("prompt_tokens_total", "generation_tokens_total",
                  "spec_decode_num_draft_tokens_total",
                  "spec_decode_num_accepted_tokens_total"):
            cur = parsed.get(f"vllm:{c}")
            old = prev.get(c)
            if cur is not None and old is not None and dt > 0:
                parsed[f"{c}_rate"] = (cur - old) / dt
            if cur is not None:
                parsed[c] = cur
        parsed["_t"] = now
        parsed["_model"] = model
        parsed["_port"] = port
        self._engine_cache[model] = parsed
        self._engine_cache_ts = now

    def engine_state(self) -> dict:
        """Return last fetched engine /metrics view (or {} if none)."""
        m = self.active_engine()
        return self._engine_cache.get(m or "") or {}

    def refresh_gpu_state(self, max_age_s: float = 2.0) -> None:
        """Cache nvidia-smi output for a couple seconds (it's ~80ms to invoke)."""
        now = time.time()
        if now - self._gpu_cache_ts < max_age_s:
            return
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode("utf-8", "ignore").strip()
            parts = [p.strip() for p in out.split(",")]
            self._gpu_cache = {
                "util_pct": float(parts[0]) if parts[0] not in ("", "[N/A]") else None,
                "temp_c":   float(parts[1]) if parts[1] not in ("", "[N/A]") else None,
                "power_w":  float(parts[2]) if len(parts) > 2 and parts[2] not in ("", "[N/A]") else None,
            }
        except (subprocess.SubprocessError, ValueError, OSError):
            self._gpu_cache = {}
        self._gpu_cache_ts = now

    def gpu_state(self) -> dict:
        return self._gpu_cache

    def active_engine(self) -> Optional[str]:
        """Most recently observed *successful* `request_model` — with
        llama-swap's exclusive `main` group this is the currently-loaded
        engine. Skips entries with non-2xx status or no completion at all
        (those tell us about the request, not what's loaded). Returns None
        if nothing qualifies."""
        with self.lock:
            for e in reversed(self.entries):
                m = e.get("request_model")
                if not m:
                    continue
                status = e.get("status") or 0
                if status < 200 or status >= 300:
                    continue
                if (e.get("completion_tokens") or 0) == 0 \
                   and (e.get("reasoning_chars") or 0) == 0:
                    continue
                return m
            # Fallback: most recent model name period, even if unsuccessful.
            for e in reversed(self.entries):
                m = e.get("request_model")
                if m:
                    return m
        return None
