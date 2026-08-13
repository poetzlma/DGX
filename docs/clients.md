# Client integration contract

What a client (or an agent configuring one) needs to call this stack correctly.
Stack internals: [operations.md](operations.md) · model detail: [models.md](models.md)
· machine-readable source of truth: [`deployed.yaml`](../deployed.yaml).

## Endpoint and route names

| | |
|---|---|
| LAN | `http://192.168.1.12:8079/v1` (log-proxy → llama-swap) |
| Internal gateway | `http://192.168.1.7:4000/v1` (LiteLLM, auth + alias map + billing) |
| Public edge | `https://<gateway-host>/v1` (Cloudflare → LiteLLM) |

**Every one of these `model` values reaches the same engine today** — the ds4
resident. They are gateway aliases, not separate models:

`deepseek-v4-flash-0731` (canonical) · `deepseek-v4-flash-ds4` ·
`laguna-s-2.1` · `nemotron-3-puzzle-75b` · `qwen3.6-27b` · `qwen3.6-35b-a3b`

Prefer the canonical name in new configs. Keep the legacy names working — paying
clients use them, and they are how the 2026-08-01 model change happened without
a client-side migration.

**Everything else in `deployed.yaml`'s `model_list` is currently unroutable.**
The live llama-swap config is in locked mode (resident only), so those names are
accepted by the gateway and then 404 at llama-swap. Notably
`qwen3.6-35b-a3b-vision` — **there is no working vision route right now.**

## The five settings that matter

Each of these has a failure mode that does *not* look like a config error.

| Setting | Value | Why, and what breaks otherwise |
|---|---|---|
| Context window | **131072** | Not 262144. The 256 k window belonged to the previous resident; 262144 on this engine caused a 35-minute outage ([§41](decisions.md#41-the-256-k-context-outage-a-memory-floor-that-refuses-instead-of-shrinking-2026-08-10)). A client that still declares 256 k will build prompts the engine refuses. |
| Request timeout | **≥ 1200 s** | TTFT is ~60 s at 100 k on a warm prefix and up to ~370 s cold. A short client timeout aborts *after* the box has already paid for the prefill — the work is lost and the retry re-does it. |
| Requests in flight | **1–2** | The engine serializes prefills: c=4 aggregate measures 0.92× of c=1. Parallel sub-agent fan-out does not go faster, it queues — at c=4 every stream's TTFT was 67 s on an 8 k prompt. Sequential work, not fan-out. |
| Thinking field | read **both** `content` and `reasoning_content` | The current engine returns thinking inline in `content`; the rollback engine splits it into `reasoning_content`. Hard-coding either one breaks silently on an engine swap — a client reading only `content` sees empty responses. |
| `max_tokens` | generous (≥ 2000 for real work) | Thinking is charged against the budget. Measured on the rollback engine: `max_tokens=300` → 0 characters of content, `finish_reason=length`. |

Streaming is supported and recommended — it is also the only way to see that a
long prefill is progressing rather than hung.

## Auth, and why `/v1/models` under-reports

Keys are LiteLLM virtual keys. **`GET /v1/models` is filtered by the calling
key's allowlist**, so it returns *your scope*, not the roster. A key scoped to
two routes reports two models while the gateway exposes seventeen — which reads
exactly like "the route I want isn't published," and is not. To see the real
roster, probe with an unscoped key, or read `deployed.yaml`.

If a route you need returns an authorization-style error, the key's `models`
list needs it added (gateway-side, `POST /key/update` — see
[operations.md](operations.md#litellm-integration)). Adding a route to a key
does **not** make an unloadable lane work.

Client inventory (which key belongs to which install, and its scope) is **not
tracked in this repo** — it lives in the gateway's own key store, which is the
only copy that can't go stale. List it with `/key/list` on the gateway host.

## Symptom → cause

| What the client sees | What it actually is |
|---|---|
| `200` but empty `content` | Thinking consumed the whole `max_tokens`; or you are reading the wrong field (see above) |
| Fast `503`s, engine looks healthy | The resident's memory floor is latched — the lane is up but refusing. A watchdog reloads it within 2 min ([§41](decisions.md#41-the-256-k-context-outage-a-memory-floor-that-refuses-instead-of-shrinking-2026-08-10)) |
| `404` / `502` from a valid-looking model name | That lane is not loadable in locked mode. Only the six aliases above serve |
| First request hangs for minutes, later ones are fast | Cold engine load (~10 min) or a cold prefix — the disk-KV cache makes repeated prefixes ~6× faster on TTFT |
| Parallel requests all slow down together | Expected: prefills serialize. Reduce in-flight requests |
| Costs read as $0 | A gateway route missing its price block — check `spend = 0 AND prompt_tokens > 0` ([§38](decisions.md#38-gateway-pricing-for-the-ds4-lane-the-silent-zero-spend-window-2026-08-03)) |
