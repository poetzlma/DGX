"""Engine-state monitor for the llama-swap LLM stack.

Modules:
    engine   — vLLM /metrics fetch + Prometheus parse + gauge_snapshot
    system   — nvidia-smi GPU stats + /proc/meminfo
    stages   — cold-start stage detection from container log lines
    monitor  — RequestState + LogTailer + EngineMonitor (the tick loop)

Both ``stack-tui`` (Rich-rendered TUI) and ``stack-api`` (aiohttp service)
build on the same EngineMonitor — same data, different presentation.
"""
