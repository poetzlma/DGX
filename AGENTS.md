# Working in this repo (agent entry point)

This repo operates a **live, paid LLM serving stack** on one DGX Spark (GB10,
119 GB unified memory). Nothing here is a sandbox: `config/llama-swap.yaml` is
watched, so **editing it deploys**, and the box **hard-hangs on host OOM** with
no remote recovery (it has, three times — each needing a physical power cycle;
latest 2026-08-15: engine warmup at 0.85 util beside a 30 GB download. The
kernel journal logs NVRM `NV_ERR_NO_MEMORY` minutes before the hang — check it
after every engine launch; `free`'s "available" will look fine right up to the
end. One big memory consumer at a time.)

Read [README.md](README.md) first for the architecture, then the page below that
matches your task. Prefer reading the repo over asking, and prefer
`git log`/[docs/decisions.md](docs/decisions.md) over assuming a number in prose
is current.

| Task | Read |
|---|---|
| Configure a **client**, pick a model id, debug a client-side failure | **[docs/gateway-setup.md](docs/gateway-setup.md)** — self-contained and safe to hand to an outside agent |
| What the machine-readable contract says (routes, ctx, pricing, concurrency) | **[deployed.yaml](deployed.yaml)** — the `live:` block at the top is the whole client contract; everything under `not_serving:` is history and answers nothing today |
| Why anything is configured the way it is | [docs/decisions.md](docs/decisions.md) (§1–§44, numbering is stable and cross-referenced from launchers) |
| Change/inspect the running stack, roll back, troubleshoot | [docs/operations.md](docs/operations.md) |
| Per-model and per-launcher detail | [docs/models.md](docs/models.md) |
| Benchmark something | [docs/benchmarks.md](docs/benchmarks.md) — includes the traps that make numbers lie |
| The production engine's history and constraints | [docs/deepseek-v4-flash.md](docs/deepseek-v4-flash.md) |

**Launcher scripts are primary documentation.** Each `bin/launch-*.sh` header
carries the measured reasoning for its flags, including what was tried and
rejected. Read the header before changing a flag; if you change one, update the
header in the same edit.

## Hard rules

- **Do not probe unknown gateway endpoints against the live stack.**
  `GET :8080/unload` unloads the running model and returns `200` — it reads like
  a status check and is not one.
- **Do not edit `config/llama-swap.yaml` casually.** It is live-watched: any
  write bounces the resident (~10 min). That includes a `git checkout`, `merge`,
  or `stash` that touches the file.
- **Do not load a second large engine beside the resident.** It holds ~90 GB of
  121 GB. Park production first (`bin/park-prod-ds4.sh`), which also clears the
  startup preload so a stray request cannot spawn one. Host OOM is unrecoverable
  remotely.
- **Do not raise the resident's `--gpu-memory-utilization` above 0.70.** 0.80+
  logged 60× NVRM `NV_ERR_NO_MEMORY` during warmup — the precursor to power
  cycle #3 ([§42](docs/decisions.md#42-qwen38-27b-nvfp4-is-the-coding-default-ds4-dormant-2026-08-16)).
  (The old "never exceed 131 k context" rule was a *ds4* memory-floor
  constraint, [§41](docs/decisions.md#41-the-256-k-context-outage-a-memory-floor-that-refuses-instead-of-shrinking-2026-08-10).
  It binds again only after a rollback — the current resident serves the full
  262 144 by design.)
- **Do not conclude a route is missing from `GET /v1/models`.** That endpoint is
  filtered by the calling key's allowlist. Check `deployed.yaml` instead.
- **Do not trust a single benchmark run.** GB10 decode swings ~44× on allocator
  state alone — 3-run medians, idle 2–3 min between runs — and never bench
  speculative decoding with random text (acceptance collapses on noise).
- **Check the HF cache before proposing a download**
  (`ls ~/.cache/huggingface/hub/`). Several models listed in the docs have had
  their weights deleted to reclaim disk; the docs say so per model, and a
  "rollback" for those means a multi-GB re-download, not a config swap.

## Facts that go stale fastest

The resident model, its engine, and the tok/s numbers have all changed multiple
times per month. Before relying on any of them, confirm against:

```sh
curl -s http://127.0.0.1:8080/running        # what is actually loaded
grep -A3 '^models:' config/llama-swap.yaml   # what the live config declares
git log --oneline -15                        # what changed recently
```

Prose in `README.md` and `docs/` is kept current at commit time; where it
disagrees with the three commands above, the commands win — and the prose is a
bug worth fixing in the same session.
