"""System-level data: nvidia-smi GPU stats + /proc/meminfo (unified mem)."""
import subprocess


def read_nvidia_stats() -> dict:
    out = {"util": None, "temp": None, "power": None, "clock": None}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,"
             "power.draw,clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        line = (r.stdout or "").strip().splitlines()
        if line:
            parts = [p.strip() for p in line[0].split(",")]
            try:
                out["util"] = float(parts[0]) if parts[0] != "[N/A]" else None
                out["temp"] = float(parts[1]) if parts[1] != "[N/A]" else None
                out["power"] = float(parts[2]) if parts[2] != "[N/A]" else None
                out["clock"] = float(parts[3]) if parts[3] != "[N/A]" else None
            except (ValueError, IndexError):
                pass
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return out


def read_meminfo() -> dict:
    out = {"total_gb": None, "available_gb": None, "used_gb": None}
    try:
        with open("/proc/meminfo") as f:
            kv = {}
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    kv[k.strip()] = v.strip()
        total_kb = int(kv.get("MemTotal", "0").split()[0])
        avail_kb = int(kv.get("MemAvailable", "0").split()[0])
        out["total_gb"] = total_kb / 1024 / 1024
        out["available_gb"] = avail_kb / 1024 / 1024
        out["used_gb"] = out["total_gb"] - out["available_gb"]
    except (OSError, ValueError):
        pass
    return out
