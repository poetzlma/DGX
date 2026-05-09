"""Per-request state, log tailer, and EngineMonitor (the tick loop).

vLLM 0.20.x doesn't log per-request prompt_tokens / first_token / finish_reason
events. EngineMonitor.tick() derives all of those from /metrics histogram
deltas and attributes them oldest-first (FIFO) to the in-memory request map
populated by LogTailer's "Received request" matches.

If log-proxy is in front of llama-swap and writing meta.json files to
LOG_PROXY_META_DIR, EngineMonitor.tick() also enriches RequestState entries
with the originating bearer token (key_prefix / key_masked), joined by
chatcmpl-XXX response id.
"""
import json
import os
import re
import subprocess
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, asdict, field
from typing import Optional

from .engine import fetch_metrics, gauge_snapshot, EMPTY_SNAP
from .system import read_nvidia_stats, read_meminfo
from .stages import RX_STAGE_PATTERNS, stage_progress

LOG_PROXY_META_DIR = "/tmp/log-proxy"


@dataclass
class RequestState:
    request_id: str
    start_ts: float
    prompt_tokens: Optional[int] = None
    prefill_done_ts: Optional[float] = None
    end_ts: Optional[float] = None
    gen_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    # accumulated share of gen_total deltas while this request was decoding
    gen_accum: float = 0.0
    # populated by log-proxy meta-file join (Phase 3) when traffic flows
    # Pi → LiteLLM (forward_headers=true) → log-proxy → llama-swap.
    key_masked: Optional[str] = None
    key_prefix: Optional[str] = None


_RX_RECV_PATTERNS = [
    re.compile(r"Received request (?P<id>[\w\-]+).*?prompt[_ ]tokens?[= ]+(?P<pt>\d+)"),
    re.compile(r"Received request (?P<id>[\w\-]+).*?num_prompt_tokens[= ]+(?P<pt>\d+)"),
    re.compile(r"Received request (?P<id>[\w\-]+)"),
]
_RX_FIRST = re.compile(
    r"(?:First token for|first token request|FirstToken) (?:request )?(?P<id>[\w\-]+)"
)
_RX_FINISH_PATTERNS = [
    re.compile(
        r"Finished request (?P<id>[\w\-]+).*?finish(?:ed)?_reason[= ]+(?P<fr>\w+)"
        r".*?(?:gen(?:erated|eration)?_tokens|completion_tokens)[= ]+(?P<gt>\d+)"
    ),
    re.compile(
        r"Finished request (?P<id>[\w\-]+).*?(?:gen(?:erated|eration)?_tokens|completion_tokens)[= ]+(?P<gt>\d+)"
        r".*?finish(?:ed)?_reason[= ]+(?P<fr>\w+)"
    ),
    re.compile(r"Finished request (?P<id>[\w\-]+).*?finish(?:ed)?_reason[= ]+(?P<fr>\w+)"),
    re.compile(r"Finished request (?P<id>[\w\-]+)"),
]


class LogTailer(threading.Thread):
    """Tails `docker logs -f <container>`, populates the requests map, tracks
    cold-start stage. Multiple consumers can run their own tailer concurrently;
    docker logs supports multiple readers."""

    def __init__(self, container: str,
                 requests: "OrderedDict[str, RequestState]",
                 lock: threading.Lock):
        super().__init__(daemon=True)
        self.container = container
        self.requests = requests
        self.lock = lock
        self.proc: Optional[subprocess.Popen] = None
        self.stop_flag = threading.Event()
        self.lines_seen = 0
        self.lines_matched = 0
        self.stage = "container starting"
        self.stage_detail = ""
        self.stage_start_ts = time.time()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.proc = subprocess.Popen(
                    ["docker", "logs", "-f", "--tail", "50", self.container],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert self.proc.stdout is not None
                for line in self.proc.stdout:
                    if self.stop_flag.is_set():
                        break
                    self.lines_seen += 1
                    if self._handle(line):
                        self.lines_matched += 1
            except Exception:
                pass
            if self.stop_flag.is_set():
                break
            self.stage = "container starting"
            self.stage_detail = ""
            self.stage_start_ts = time.time()
            time.sleep(2.0)

    def _handle(self, line: str) -> bool:
        now = time.time()
        for rx, stage, fn in RX_STAGE_PATTERNS:
            m = rx.search(line)
            if m:
                if stage != self.stage:
                    self.stage = stage
                    self.stage_start_ts = now
                self.stage_detail = fn(m)
                break
        for rx in _RX_RECV_PATTERNS:
            m = rx.search(line)
            if m:
                rid = m.group("id")
                pt = None
                try:
                    pt = int(m.group("pt"))
                except (IndexError, ValueError):
                    pass
                with self.lock:
                    if rid not in self.requests:
                        self.requests[rid] = RequestState(
                            request_id=rid, start_ts=now, prompt_tokens=pt,
                        )
                    elif pt is not None and self.requests[rid].prompt_tokens is None:
                        self.requests[rid].prompt_tokens = pt
                return True
        m = _RX_FIRST.search(line)
        if m:
            rid = m.group("id")
            with self.lock:
                r = self.requests.get(rid)
                if r and r.prefill_done_ts is None:
                    r.prefill_done_ts = now
            return True
        for rx in _RX_FINISH_PATTERNS:
            m = rx.search(line)
            if m:
                rid = m.group("id")
                fr = m.groupdict().get("fr")
                gt = None
                try:
                    gt = int(m.group("gt"))
                except (IndexError, ValueError):
                    pass
                with self.lock:
                    r = self.requests.get(rid)
                    if r is None:
                        r = RequestState(request_id=rid, start_ts=now - 1.0)
                        self.requests[rid] = r
                    if r.end_ts is None:
                        r.end_ts = now
                        r.finish_reason = fr
                        r.gen_tokens = gt
                return True
        return False

    def stop(self):
        self.stop_flag.set()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


class EngineMonitor:
    """One engine, one monitor instance. Owns:
        - the requests OrderedDict + its lock (also exposed for direct
          read-locked access from the TUI gantt renderer)
        - a LogTailer thread that reads docker logs
        - the per-tick deltas needed for histogram-based lifecycle attribution
        - the rolling tps_window (5-tick mean of decode tok/s)

    Call ``tick()`` periodically (~1s). Read state via attributes (TUI) or via
    ``snapshot()`` (API).
    """

    REASONS = ("stop", "length", "abort", "error")

    def __init__(self, engine: dict, view_duration_s: float = 300.0):
        self.engine = engine
        self.view_duration_s = view_duration_s
        self.requests: "OrderedDict[str, RequestState]" = OrderedDict()
        self.lock = threading.Lock()
        self.tailer = LogTailer(engine["name"], self.requests, self.lock)
        self.tailer.start()

        self.start_ts = time.time()
        self.prev_gen_total = 0.0
        self.prev_tick_ts = time.time()
        self.tps_window: deque = deque(maxlen=5)
        self.prev_succ: dict = {}
        self.prev_ttft_count = 0.0
        self.prev_ttft_sum = 0.0
        self.prev_prompt_total = 0.0
        # accumulator for prompt tokens between first-token events:
        # prompt_tokens_total can increment in earlier ticks (chunked prefill)
        # before ttft fires, so we accumulate and divide at first-token time
        self.prompt_accum = 0.0

        # most-recent tick state, exposed via attributes / snapshot()
        self.snap = dict(EMPTY_SNAP)
        self.gpu = {"util": None, "temp": None, "power": None, "clock": None}
        self.mem = {"total_gb": None, "available_gb": None, "used_gb": None}
        self.smoothed_tps = 0.0
        self.metrics_ok = False

        # log-proxy meta enrichment: track which meta files we've already
        # consumed so we re-read only newer ones each tick.
        self._meta_seen: set = set()

    def tick(self) -> None:
        m = fetch_metrics(self.engine["port"])
        self.metrics_ok = m is not None
        snap = gauge_snapshot(m) if m else dict(EMPTY_SNAP)

        now = time.time()
        dt = max(0.001, now - self.prev_tick_ts)
        d_gen = (max(0.0, snap["gen_total"] - self.prev_gen_total)
                 if self.prev_gen_total > 0 else 0.0)
        if self.prev_gen_total > 0:
            self.tps_window.append(d_gen / dt)

        if not self.prev_succ:
            # baseline tick: don't attribute pre-existing counts
            self.prev_succ = {r: snap[f"succ_{r}"] for r in self.REASONS}
            self.prev_ttft_count = snap["ttft_count"]
            self.prev_ttft_sum = snap["ttft_sum"]
            self.prev_prompt_total = snap["prompt_total"]
            self.prompt_accum = 0.0
        else:
            d_prompt = max(0.0, snap["prompt_total"] - self.prev_prompt_total)
            self.prompt_accum += d_prompt

            # FIRST-TOKEN attribution (set prefill_done_ts and prompt_tokens)
            d_ttft_count = int(snap["ttft_count"] - self.prev_ttft_count)
            d_ttft_sum = max(0.0, snap["ttft_sum"] - self.prev_ttft_sum)
            if d_ttft_count > 0:
                avg_ttft = d_ttft_sum / d_ttft_count
                avg_prompt = (self.prompt_accum / d_ttft_count
                              if d_ttft_count else 0)
                with self.lock:
                    candidates = sorted(
                        (r for r in self.requests.values()
                         if r.end_ts is None and r.prefill_done_ts is None),
                        key=lambda x: x.start_ts,
                    )
                    for r in candidates[:d_ttft_count]:
                        r.prefill_done_ts = r.start_ts + avg_ttft
                        if r.prompt_tokens is None and avg_prompt > 0:
                            r.prompt_tokens = int(avg_prompt)
                self.prompt_accum = 0.0

            # GEN-TOKEN accumulation: distribute this tick's gen_total delta
            # evenly across currently-decoding requests so completion sees the
            # full output, not just the last tick's slice.
            if d_gen > 0:
                with self.lock:
                    decoding = [
                        r for r in self.requests.values()
                        if r.prefill_done_ts is not None and r.end_ts is None
                    ]
                    if decoding:
                        share = d_gen / len(decoding)
                        for r in decoding:
                            r.gen_accum += share

            # COMPLETION attribution (set end_ts, finish_reason, gen_tokens)
            deltas = {r: int(snap[f"succ_{r}"] - self.prev_succ.get(r, 0))
                      for r in self.REASONS}
            total_new = sum(d for d in deltas.values() if d > 0)
            if total_new > 0:
                with self.lock:
                    pending = sorted(
                        (r for r in self.requests.values()
                         if r.end_ts is None),
                        key=lambda x: x.start_ts,
                    )
                    for reason, d in deltas.items():
                        if d <= 0:
                            continue
                        for _ in range(d):
                            if not pending:
                                break
                            req = pending.pop(0)
                            req.end_ts = now
                            req.finish_reason = reason
                            if req.gen_tokens is None:
                                if req.gen_accum > 0:
                                    req.gen_tokens = int(req.gen_accum)
                                elif d_gen > 0 and total_new > 0:
                                    # request started+finished within a single
                                    # tick under concurrency
                                    req.gen_tokens = int(d_gen / total_new)
                            if (req.prompt_tokens is None
                                    and self.prompt_accum > 0
                                    and total_new > 0):
                                req.prompt_tokens = int(
                                    self.prompt_accum / total_new)

            self.prev_succ = {r: snap[f"succ_{r}"] for r in self.REASONS}
            self.prev_ttft_count = snap["ttft_count"]
            self.prev_ttft_sum = snap["ttft_sum"]
            self.prev_prompt_total = snap["prompt_total"]

        self.prev_gen_total = snap["gen_total"]
        self.prev_tick_ts = now
        self.snap = snap
        self.gpu = read_nvidia_stats()
        self.mem = read_meminfo()
        self.smoothed_tps = sum(self.tps_window) / max(1, len(self.tps_window))

        self._enrich_from_log_proxy()

        # prune stale completed requests outside double-view-window
        cutoff = now - self.view_duration_s * 2
        with self.lock:
            stale = [k for k, v in self.requests.items()
                     if v.end_ts is not None and v.end_ts < cutoff]
            for k in stale:
                del self.requests[k]

    def _enrich_from_log_proxy(self) -> None:
        """If log-proxy is running (LOG_PROXY_META_DIR exists), join its
        meta.json files to in-memory RequestState entries by chatcmpl-XXX.
        Silent no-op when log-proxy isn't deployed."""
        if not os.path.isdir(LOG_PROXY_META_DIR):
            return
        try:
            entries = os.listdir(LOG_PROXY_META_DIR)
        except OSError:
            return
        new = [f for f in entries
               if f.endswith(".meta.json") and f not in self._meta_seen]
        if not new:
            return
        for fname in new:
            self._meta_seen.add(fname)
            path = os.path.join(LOG_PROXY_META_DIR, fname)
            try:
                with open(path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            response_id = meta.get("response_id")
            if not response_id:
                continue
            with self.lock:
                r = self.requests.get(response_id)
                if r is None:
                    continue
                if r.key_masked is None and meta.get("key_masked"):
                    r.key_masked = meta["key_masked"]
                if r.key_prefix is None and meta.get("key_prefix"):
                    r.key_prefix = meta["key_prefix"]

    def stage(self) -> dict:
        return stage_progress(self.tailer.stage, self.tailer.stage_detail,
                              self.tailer.stage_start_ts)

    def counts(self) -> dict:
        """In-flight / completed-in-view / truncation counts."""
        now = time.time()
        view_start = now - self.view_duration_s
        with self.lock:
            n_inflight = sum(1 for r in self.requests.values()
                             if r.end_ts is None)
            n_completed = sum(1 for r in self.requests.values()
                              if r.end_ts is not None and r.end_ts >= view_start)
            trunc_count = sum(
                1 for r in self.requests.values()
                if r.end_ts is not None and r.gen_tokens is not None
                and r.gen_tokens < 5 and (r.prompt_tokens or 0) > 100
                and r.finish_reason == "stop"
            )
        return {"inflight": n_inflight,
                "completed_in_view": n_completed,
                "trunc": trunc_count}

    def snapshot(self) -> dict:
        """JSON-safe view of current state. Used by stack-api; the TUI accesses
        attributes + ``requests`` directly under ``lock``."""
        with self.lock:
            requests_serialized = [asdict(r) for r in self.requests.values()]
        return {
            "engine": {
                "name": self.engine["name"],
                "port": self.engine["port"],
                "uptime_s": time.time() - self.start_ts,
                "metrics_ok": self.metrics_ok,
            },
            "raw_snap": self.snap,
            "gpu": self.gpu,
            "mem": self.mem,
            "stage": self.stage(),
            "tps_smoothed": self.smoothed_tps,
            "tailer": {
                "lines_seen": self.tailer.lines_seen,
                "lines_matched": self.tailer.lines_matched,
            },
            "counts": self.counts(),
            "requests": requests_serialized,
        }

    def stop(self):
        self.tailer.stop()
