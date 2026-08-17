# Gateway setup for clients

**Audience: anyone (or any agent) configuring a client against this gateway.**
Everything needed to get a working, correctly-tuned connection is here — and
nothing about the serving infrastructure behind it, deliberately. If you are
operating the stack rather than calling it, see
[operations.md](operations.md) instead.

*Current as of 2026-08-17. The served model changes every few weeks; the
machine-readable version of this page is [`deployed.yaml`](../deployed.yaml)
(`live:` block), and it is the tiebreaker if the two ever disagree.*

## Connection

```
Base URL   https://<gateway-host>/v1          — host supplied with your key
API        OpenAI-compatible (chat completions, streaming, tool calls)
Auth       Authorization: Bearer <your key>   — issued per client by the operator
Model      qwen3.8-27b
```

`<gateway-host>` is not published here; you receive it together with your key.
Everything below is independent of it.

**That is the only model id the gateway serves.** As of 2026-08-17 the six
legacy aliases (`deepseek-v4-flash-0731`, `deepseek-v4-flash-ds4`,
`laguna-s-2.1`, `nemotron-3-puzzle-75b`, `qwen3.6-27b`, `qwen3.6-35b-a3b`) are
**retired** — they now return HTTP 400, and every key is scoped to
`qwen3.8-27b` alone. If your client still sends one of those names, change the
model id; there is no alias to fall back on.

**What you're talking to:** Qwen3.8-27B (dense, NVFP4), a reasoning model with a
**262 144-token** context window. It accepts **images** (up to 4 per prompt) and
does tool calling. Served since 2026-08-16; it replaced DeepSeek V4-Flash, and
every setting below changed at that cutover.

```sh
GATEWAY=https://<gateway-host>   # as supplied with your key
KEY=<your key>

curl "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 4000,
    "stream": true
  }'
```

## The five settings that matter

Defaults tuned for hosted APIs are wrong here in ways that don't look like
configuration errors. Set these explicitly.

| Setting | Value | What goes wrong otherwise |
|---|---|---|
| **Context window** | `262144` | A client that declares less silently wastes most of the window; one that declares more builds prompts the model will refuse, with no warning at config time. |
| **Request timeout** | **≥ 1200 s** | Time-to-first-token is tens of seconds on a large prompt and ~3 minutes on a long, previously-unseen one. A 60–120 s timeout aborts *after* the expensive work is done, and the retry repeats it. |
| **Reasoning field** | read **both** `reasoning` and `reasoning_content` | This model puts thinking in **`reasoning`**. The previous one used `reasoning_content`, and it can change again at the next model swap. A client reading only one field silently shows empty responses. |
| **`max_tokens`** | ≥ 2000 for real work | Thinking is charged against the same budget. Too small a budget can be spent entirely on reasoning, returning empty content with `finish_reason: length`. |
| **Concurrent requests** | **up to 8** | 8 are accepted and parallelism genuinely pays: aggregate goes 26.7 → 93.7 → 167.3 tok/s at c=1/4/8, so 6.3× more work for 17 % slower individual responses. Request 9 queues rather than failing. The exception is very long prompts — the cache holds ~5.8 requests at the full 262 k, so eight simultaneous near-max prompts will thrash. Fan out freely at ordinary sizes; stay near 4 if every request is enormous. |

Leave sampling alone unless you have a reason: the model ships its own
recommended defaults (`temperature 1.0`, `top_p 0.95`, `top_k 20`) and the
gateway does not override them. Sending hosted-API generics (e.g. `temperature
0.0`) measurably degrades this model.

Streaming is supported and recommended: on a long prompt it's the difference
between watching progress and guessing whether you're hung. Reasoning streams
incrementally — you see thinking tokens within a second, not buffered to the end.

## Symptom → cause

| What you see | What it is | What to do |
|---|---|---|
| `200` with empty `content` | Thinking consumed the whole budget, or you're reading the wrong field | Raise `max_tokens`; read `reasoning` **and** `reasoning_content` |
| First request slow, later ones fast | Cold start, plus prompt-prefix caching that makes repeated prefixes far cheaper | Keep prompt prefixes stable across turns; don't reshuffle system context |
| Throughput stops improving as you add parallelism | Expected — the box is bandwidth-bound, flat past ~2 in flight | Reduce in-flight requests to 1–2 |
| Brief `503`s that clear on their own | A transient server-side condition with automatic recovery | Retry with backoff; it self-heals within a couple of minutes |
| `404` or an error naming an unknown model | That model id isn't served | Use `qwen3.8-27b` |
| An error on a model id you found in the docs | Every other id in the repo is **historical** — `qwen3.8-27b` is the only one served | Use `qwen3.8-27b` |
| An authorization-style error on a model that exists | Your key isn't scoped to that model | Ask the operator to add it |
| `GET /v1/models` seems to be missing models | **This endpoint returns only what *your key* is allowed to use — not the full roster.** | Don't infer availability from it; ask the operator |

That last row is worth internalizing: an agent debugging a client here once
concluded the canonical model wasn't published, because its key's allowlist
happened to contain two other ids. The endpoint is a permissions view, not a
catalog.

## Asking for a change

Requests the operator can action: a new key, adding a model to your key's
allowlist, or raising a rate/budget limit. Include the client name and the model
id you want. Adding a model to a key does not create capacity that isn't there —
if a route isn't currently served, access won't make it work.
