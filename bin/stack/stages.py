"""Cold-start stage detection from vLLM container log lines."""
import re
import time

COLD_STAGES = [
    "container starting",
    "loading weights",
    "compiling graphs",
    "capturing cudagraphs",
    "starting server",
    "ready",
]
COLD_STAGE_TYPICAL_S = {
    "container starting":     15.0,
    "loading weights":       180.0,
    "compiling graphs":       90.0,
    "capturing cudagraphs":  240.0,
    "starting server":        10.0,
}

# (regex, target_stage, detail_fn(match) -> str)
RX_STAGE_PATTERNS = [
    (re.compile(r"Loading safetensors checkpoint shards:\s+(?P<pct>\d+)%"
                r"[^|]*\|\s*(?P<idx>\d+)/(?P<tot>\d+)"),
     "loading weights",
     lambda m: f"shards {m.group('idx')}/{m.group('tot')} ({m.group('pct')}%)"),
    (re.compile(r"Loading weights took"),
     "compiling graphs", lambda m: "weights loaded"),
    (re.compile(r"Dynamo bytecode transform time"),
     "compiling graphs", lambda m: "dynamo"),
    (re.compile(r"Compiling a graph for compile range \((?P<r>[^)]+)\)"),
     "compiling graphs", lambda m: f"compile range {m.group('r')}"),
    (re.compile(r"torch\.compile took"),
     "capturing cudagraphs", lambda m: "starting capture"),
    (re.compile(r"Capturing cudagraph for batch_size=(?P<bs>\d+)"),
     "capturing cudagraphs", lambda m: f"batch_size={m.group('bs')}"),
    (re.compile(r"Capturing CUDA graph shapes:\s+(?P<pct>\d+)%"),
     "capturing cudagraphs", lambda m: f"{m.group('pct')}%"),
    (re.compile(r"Started server process|Application startup complete"),
     "ready", lambda m: ""),
    (re.compile(r"Engine core initialization failed|Engine core failed to start"),
     "CRASHED", lambda m: ""),
]


def stage_progress(stage: str, stage_detail: str,
                   stage_start_ts: float) -> dict:
    """Compute the cold-start progress dict from the tailer's stage state."""
    elapsed = time.time() - stage_start_ts
    typical = COLD_STAGE_TYPICAL_S.get(stage, 0.0)
    remaining = max(0.0, typical - elapsed) if typical else 0.0
    try:
        idx = COLD_STAGES.index(stage)
    except ValueError:
        idx = -1
    return {
        "stage": stage,
        "detail": stage_detail,
        "idx": idx,
        "n_progress_stages": len(COLD_STAGES) - 1,
        "elapsed": elapsed,
        "typical": typical,
        "remaining": remaining,
    }
