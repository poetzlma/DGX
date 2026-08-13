# Gateway setup for clients

**Audience: anyone (or any agent) configuring a client against this gateway.**
Everything needed to get a working, correctly-tuned connection is here — and
nothing about the serving infrastructure behind it, deliberately. If you are
operating the stack rather than calling it, see
[operations.md](operations.md) instead.

## Connection

```
Base URL   https://<gateway-host>/v1          — host supplied with your key
API        OpenAI-compatible (chat completions, streaming, tool calls)
Auth       Authorization: Bearer <your key>   — issued per client by the operator
Model      deepseek-v4-flash-0731
```

`<gateway-host>` is not published here; you receive it together with your key.
Everything below is independent of it.

That model id is the one to put in new configs. Several older ids
(`deepseek-v4-flash-ds4`, `laguna-s-2.1`, `nemotron-3-puzzle-75b`,
`qwen3.6-27b`, `qwen3.6-35b-a3b`) still resolve to the same model so existing
clients keep working — don't add them to anything new, and don't assume a
different id means a different model.

**What you're talking to:** DeepSeek V4-Flash (304 B total / 13 B active MoE),
a reasoning model with a 131 072-token context window. Text only — there is no
working image/vision route at present.

```sh
GATEWAY=https://<gateway-host>   # as supplied with your key
KEY=<your key>

curl "$GATEWAY/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 2000,
    "stream": true
  }'
```

## The five settings that matter

Defaults tuned for hosted APIs are wrong here in ways that don't look like
configuration errors. Set these explicitly.

| Setting | Value | What goes wrong otherwise |
|---|---|---|
| **Context window** | `131072` | A client that declares 256 k builds prompts the model will refuse. Nothing warns you at config time. |
| **Request timeout** | **≥ 1200 s** | Time-to-first-token is tens of seconds on a large prompt and can reach several minutes on a long, previously-unseen one. A 60–120 s timeout aborts *after* the expensive work is done, and the retry repeats it. |
| **Concurrent requests** | **1–2** | This model processes prompts one at a time. Parallel fan-out does not run faster — it queues, and every request's latency grows together. Sequential steps beat parallel sub-agents here. |
| **Reasoning field** | read **both** `content` and `reasoning_content` | Where thinking appears depends on the serving engine and can change without notice. A client reading only one field can silently show empty responses. |
| **`max_tokens`** | ≥ 2000 for real work | Thinking is charged against the same budget. Too small a budget can be spent entirely on reasoning, returning empty content with `finish_reason: length`. |

Streaming is supported and recommended: on a long prompt it's the difference
between watching progress and guessing whether you're hung.

## Symptom → cause

| What you see | What it is | What to do |
|---|---|---|
| `200` with empty `content` | Thinking consumed the whole budget, or you're reading the wrong field | Raise `max_tokens`; read `reasoning_content` too |
| First request slow, later ones fast | Cold start, plus prompt-prefix caching that makes repeated prefixes far cheaper | Keep prompt prefixes stable across turns; don't reshuffle system context |
| All parallel requests slow together | Expected — prompts are processed serially | Reduce in-flight requests to 1–2 |
| Brief `503`s that clear on their own | A transient server-side condition with automatic recovery | Retry with backoff; it self-heals within a couple of minutes |
| `404` or an error naming an unknown model | That model id isn't served | Use `deepseek-v4-flash-0731` |
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
