#!/usr/bin/env python3
"""Resumable parallel-range HTTP downloader for big model files.

WHY: a single curl to the HF CDN sustains only ~9-16 MB/s from this box, but
range requests parallelise cleanly (measured: 4 connections -> ~33 MB/s), so an
80 GiB pull goes from >2 h to well under one. Written because neither aria2c
nor the hf CLI is installed here and this needed no root.

Resumes: chunk completion is journalled to <out>.state, so a killed run picks up
where it stopped. A partially-written chunk is simply redone (chunks are only
marked complete after the full range lands), so the result is never silently
truncated.

Usage: pardl.py URL OUT TOTAL_BYTES [--jobs N] [--chunk MB] [--seeded BYTES]
  --seeded  bytes at the head of OUT already known-good (e.g. from a prior
            sequential `curl -C -`), so we don't refetch them.
"""
import json, os, sys, threading, time, urllib.request

url, out, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
chunk = (int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 256) * 1024 * 1024
seeded = int(sys.argv[sys.argv.index("--seeded") + 1]) if "--seeded" in sys.argv else 0
state_path = out + ".state"

# Chunk list. The seeded head becomes pre-completed chunks so the journal and
# the byte ranges stay aligned on the same grid across restarts.
chunks = [(off, min(off + chunk, total) - 1) for off in range(0, total, chunk)]
done = set()
if os.path.exists(state_path):
    done = set(json.load(open(state_path))["done"])
elif seeded:
    done = {i for i, (a, b) in enumerate(chunks) if b < seeded}
    print(f"seeded: {len(done)} chunks already present from sequential download")

# Preallocate so parallel pwrites never race the allocator.
fd = os.open(out, os.O_WRONLY | os.O_CREAT)
if os.fstat(fd).st_size != total:
    os.ftruncate(fd, total)

lock = threading.Lock()
bytes_done = sum(chunks[i][1] - chunks[i][0] + 1 for i in done)
t_start = time.time()
queue = [i for i in range(len(chunks)) if i not in done]
qlock = threading.Lock()
failures = []


def save_state():
    tmp = state_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"done": sorted(done), "total": total, "chunks": len(chunks)}, fh)
    os.replace(tmp, state_path)


def worker():
    global bytes_done
    while True:
        with qlock:
            if not queue:
                return
            i = queue.pop(0)
        a, b = chunks[i]
        for attempt in range(5):
            try:
                # Stream to disk in small pieces. Buffering a whole chunk in RAM
                # would put jobs*chunk bytes of heap against a box that is
                # already 113/121 GB committed to the resident engine, and this
                # hardware hard-hangs on host OOM with no remote recovery.
                req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                want = b - a + 1
                got = 0
                with urllib.request.urlopen(req, timeout=180) as r:
                    while got < want:
                        piece = r.read(8 * 1024 * 1024)
                        if not piece:
                            break
                        os.pwrite(fd, piece, a + got)
                        got += len(piece)
                if got != want:
                    raise IOError(f"short read {got} != {want}")
                with lock:
                    done.add(i)
                    bytes_done += got
                    save_state()
                    el = time.time() - t_start
                    pct = 100.0 * bytes_done / total
                    rate = (bytes_done - seeded_bytes0) / el / 1e6 if el > 0 else 0
                    eta = (total - bytes_done) / (rate * 1e6) / 60 if rate > 0 else 0
                    print(f"\r{pct:5.1f}%  {bytes_done/2**30:7.2f}/{total/2**30:.2f} GiB  "
                          f"{rate:5.1f} MB/s  ETA {eta:5.1f} min  ({len(done)}/{len(chunks)} chunks)",
                          end="", flush=True)
                break
            except Exception as e:
                if attempt == 4:
                    with lock:
                        failures.append((i, str(e)[:80]))
                else:
                    time.sleep(2 * (attempt + 1))


seeded_bytes0 = bytes_done
print(f"total {total/2**30:.2f} GiB in {len(chunks)} chunks of {chunk//2**20} MiB | "
      f"{len(queue)} to fetch | {jobs} connections")
threads = [threading.Thread(target=worker, daemon=True) for _ in range(jobs)]
for t in threads:
    t.start()
for t in threads:
    t.join()
os.close(fd)
print()
if failures:
    print(f"FAILED chunks: {failures[:10]} (total {len(failures)}) — rerun to retry")
    sys.exit(1)
if len(done) != len(chunks):
    print(f"INCOMPLETE: {len(done)}/{len(chunks)} — rerun to resume")
    sys.exit(1)
print(f"complete: {out} ({os.path.getsize(out)} bytes) in {(time.time()-t_start)/60:.1f} min")
os.remove(state_path)
