# Dashboard MVP — Spark side

## What runs on the Spark

| Service | Port | Description |
|---|---|---|
| `llama-swap` | 8080 | OpenAI-compatible gateway, hot-swaps vLLM backends |
| `log-proxy` | 8079 | Transparent logging proxy (Phase 3, optional) — fronts llama-swap to capture requests + Authorization headers |
| `stack-api` | 8090 | HTTP snapshot API for the dashboard (NEW) |
| vLLM container | 9016 | Currently `vllm-qwen-27b-fp8` (production model, see §26c of llm-stack README) |

## stack-api

Single endpoint: `GET /api/snapshot` returns a JSON view of the current
engine state — cockpit gauges (GPU, MEM, KV, ACC, TPS, PWR), header stats
(mean TTFT, mean per-request decode rate, in/out tokens, spec acceptance,
e2e latency), in-flight + completed-in-window request counts, and a list
of recent requests with their lifecycle timestamps.

`GET /api/healthz` is the liveness probe (no auth).

Auth is `Authorization: Bearer $STACK_API_TOKEN`. The token lives in
`/home/max/llm-stack/.env-stack-api` (mode 0600, systemd EnvironmentFile).

```sh
sudo systemctl status stack-api
curl http://192.168.1.12:8090/api/healthz
curl -H "Authorization: Bearer $TOKEN" http://192.168.1.12:8090/api/snapshot | jq .engine
```

## stack-tui

Lives in `bin/stack-tui`. Imports the same `stack.monitor.EngineMonitor`
the API uses — same data, terminal rendering. Nothing changed for the user.

## Dashboard architecture (cockroach side)

```
Friend's browser
   │ (Cloudflare Access — email allowlist)
   ▼
cockroach-dashboard (cockroach :3001 or new sub-route)
   │ static HTML page
   ▼ fetch every 2s
http://192.168.1.12:8090/api/snapshot
   Authorization: Bearer $STACK_API_TOKEN
```

Cockroach side is responsible for:
- Hosting the static HTML page (rendering gauges + gantt + stats)
- Holding the `STACK_API_TOKEN` secret (never reaches the browser; the
  cockroach backend or Caddy injects the header)
- Cloudflare Access policy (email allowlist for friends)
- Optional: querying LiteLLM Postgres `LiteLLM_VerificationToken` to
  resolve `key_prefix` → friendly `key_alias` for the gantt rows

The Spark side stays dumb — it just exposes JSON.

## Phase 3: per-user enrichment

When a request flows
**Pi → LiteLLM → log-proxy → llama-swap**, log-proxy captures:
- The `Authorization: Bearer sk-pi-…` header from LiteLLM (which forwarded
  it via `general_settings.forward_headers: true`)
- The vLLM-emitted `chatcmpl-XXX` response id

Both are written to `/tmp/log-proxy/<ts>-<reqid>.meta.json`.

`EngineMonitor._enrich_from_log_proxy()` reads new meta files each tick and
joins them onto the in-memory request map by `chatcmpl-XXX`, populating
`RequestState.key_masked` and `RequestState.key_prefix`. These appear in
the snapshot JSON.

### LiteLLM config change (codeserver side)

In `cockroach/infra/litellm-config.yaml`:

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  forward_headers: true                 # NEW

# Repoint every model entry's api_base from :8080 to :8079
model_list:
  - model_name: qwen3.6-27b
    litellm_params:
      model: openai/qwen3.6-27b
      api_base: http://192.168.1.12:8079/v1   # was :8080
      ...
```

Then `docker compose restart litellm` on the cockroach.

### Without log-proxy

`EngineMonitor._enrich_from_log_proxy()` no-ops when `/tmp/log-proxy/` is
absent. Per-user info is just empty in the snapshot — Phase 1 and 2 still
work.
