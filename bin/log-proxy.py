#!/home/max/llm-stack/venv/bin/python3
"""Transparent logging proxy in front of llama-swap.

Listens on 0.0.0.0:8079, forwards to localhost:8080, captures every
POST /v1/chat/completions request and the full response stream to disk
so we can diagnose intermittent model-side failures (e.g., the "</think>C"
truncation we are chasing).

Each request is saved as two files in /tmp/log-proxy/:
    <ts>-<reqid>.req.json   — full incoming request body
    <ts>-<reqid>.resp.txt   — raw response stream (SSE chunks or JSON)
    <ts>-<reqid>.meta.json  — status, durations, model, finish_reason, sizes

Repoint LiteLLM (or opencode directly) from :8080 → :8079 to start logging.
"""
import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone

import aiohttp
from aiohttp import web

UPSTREAM = "http://127.0.0.1:8080"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8079
LOG_DIR = "/home/max/llm-stack/logs/proxy"

os.makedirs(LOG_DIR, mode=0o700, exist_ok=True)
# Logs hold request bodies + masked client keys; keep them owner-only even if
# the dir already existed (makedirs mode is a no-op on an existing dir).
os.chmod(LOG_DIR, 0o700)


def _ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _safe(name):
    if not name:
        return "_unknown"
    return name.replace("/", "_").replace(":", "_").replace("..", "_")[:64]


def _bucket(ts, model):
    day = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    d = f"{LOG_DIR}/{day}/{_safe(model)}"
    os.makedirs(d, exist_ok=True)
    return d


async def proxy(request: web.Request) -> web.StreamResponse:
    reqid = uuid.uuid4().hex[:12]
    ts = _ts()
    is_chat = request.path == "/v1/chat/completions" and request.method == "POST"

    body_bytes = await request.read()
    # Peek at the request body early so we can bucket logs by model.
    early_model = None
    try:
        peek = json.loads(body_bytes)
        if isinstance(peek, dict):
            early_model = peek.get("model")
    except (json.JSONDecodeError, TypeError, ValueError):
        peek = None
    base = f"{_bucket(ts, early_model)}/{ts}-{reqid}"

    if is_chat:
        try:
            with open(f"{base}.req.json", "wb") as f:
                f.write(body_bytes)
        except OSError:
            pass

    # Strip hop-by-hop headers
    hop = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host",
    }
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in hop}
    fwd_headers["X-Log-Proxy-ReqID"] = reqid

    t_start = time.perf_counter()
    upstream_url = UPSTREAM + request.path_qs

    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        async with session.request(
            request.method, upstream_url,
            headers=fwd_headers, data=body_bytes,
        ) as upstream:
            # Build response
            resp_headers = {
                k: v for k, v in upstream.headers.items()
                if k.lower() not in hop
            }
            resp = web.StreamResponse(
                status=upstream.status,
                reason=upstream.reason,
                headers=resp_headers,
            )
            await resp.prepare(request)

            log_f = open(f"{base}.resp.txt", "wb") if is_chat else None
            chunks_written = 0
            t_first_byte = None
            t_first_token = None
            aborted = None  # None / "client" / "upstream"
            try:
                async for chunk in upstream.content.iter_any():
                    now = time.perf_counter()
                    if t_first_byte is None:
                        t_first_byte = now
                    chunks_written += len(chunk)
                    if log_f:
                        log_f.write(chunk)
                    # First-content detection: scan SSE deltas until a
                    # content / reasoning_content / tool_calls field appears.
                    if t_first_token is None and is_chat:
                        for line in chunk.split(b"\n"):
                            if not line.startswith(b"data: "):
                                continue
                            pl = line[6:].strip()
                            if pl in (b"[DONE]", b""):
                                continue
                            try:
                                obj = json.loads(pl)
                            except json.JSONDecodeError:
                                continue
                            for choice in obj.get("choices") or []:
                                d = choice.get("delta") or {}
                                if (d.get("content") or
                                        d.get("reasoning_content") or
                                        d.get("tool_calls")):
                                    t_first_token = now
                                    break
                            if t_first_token is not None:
                                break
                    try:
                        await resp.write(chunk)
                    except (ConnectionResetError, aiohttp.ClientError):
                        aborted = "client"
                        break
            except aiohttp.ClientPayloadError:
                aborted = "upstream"
            finally:
                if log_f:
                    log_f.close()

            if aborted is None:
                try:
                    await resp.write_eof()
                except (ConnectionResetError, aiohttp.ClientError):
                    aborted = "client"
            elapsed = time.perf_counter() - t_start

            if is_chat:
                # Try to extract finish_reason / usage / model from response
                meta = {
                    "reqid": reqid,
                    "ts": ts,
                    "method": request.method,
                    "path": request.path_qs,
                    "client_ip": request.remote,
                    "user_agent": request.headers.get("User-Agent"),
                    "status": upstream.status,
                    "elapsed_s": round(elapsed, 3),
                    "request_bytes": len(body_bytes),
                    "response_bytes": chunks_written,
                }
                if t_first_byte is not None:
                    meta["tfb_s"] = round(t_first_byte - t_start, 4)
                if t_first_token is not None:
                    meta["ttft_s"] = round(t_first_token - t_start, 4)
                if aborted is not None:
                    meta["aborted"] = aborted

                # ── Auth: capture a *masked* form of the client key for
                # downstream user/key enrichment. The full token is NEVER
                # written to disk — only a 6+4 masked form and an 8-char prefix
                # for correlation. (Do not reintroduce key_full: these logs are
                # for debugging, not a credential store.)
                auth = request.headers.get("Authorization", "")
                if auth.lower().startswith("bearer "):
                    token = auth[7:].strip()
                    if len(token) > 8:
                        meta["key_masked"] = f"{token[:6]}…{token[-4:]}"
                    else:
                        meta["key_masked"] = "…"
                    meta["key_prefix"] = token[:8]

                # Also dump every X-* and a few LiteLLM-relevant headers, so
                # we can see what the upstream proxy is actually forwarding.
                # If LiteLLM tags the call with x-litellm-key-name or similar,
                # we'll catch it here.
                meta["fwd_headers"] = {
                    k: v for k, v in request.headers.items()
                    if k.lower().startswith(("x-", "litellm-", "openai-"))
                }
                # Standard OpenAI `user` field from request body — many
                # clients (incl. Pi) set it as their identity hint.
                try:
                    rb_peek = json.loads(body_bytes)
                    if isinstance(rb_peek, dict) and rb_peek.get("user"):
                        meta["body_user"] = rb_peek["user"]
                    md = (rb_peek.get("metadata") or {}) if isinstance(rb_peek, dict) else {}
                    for k in ("user_api_key", "user_api_key_alias",
                              "user_api_key_user_id", "user_api_key_team_id"):
                        if k in md:
                            meta[f"body_{k}"] = md[k]
                except (json.JSONDecodeError, TypeError):
                    pass
                # Pull a few notable fields out of the request body
                try:
                    rb = json.loads(body_bytes)
                    meta["request_model"] = rb.get("model")
                    msgs = rb.get("messages") or []
                    meta["n_messages"] = len(msgs)
                    tools_list = rb.get("tools") or []
                    meta["tools_count"] = len(tools_list)
                    meta["tool_names"] = [
                        (t.get("function") or {}).get("name")
                        for t in tools_list if isinstance(t, dict)
                    ]
                    meta["tool_choice"] = rb.get("tool_choice")
                    meta["max_tokens"] = rb.get("max_tokens")
                    meta["stream"] = rb.get("stream", False)
                    last = msgs[-1] if msgs else {}
                    last_content = last.get("content") if isinstance(last, dict) else None
                    if isinstance(last_content, str):
                        meta["last_user_content_chars"] = len(last_content)
                    # Count tool_result / tool messages already in conversation;
                    # useful when chasing repeated tool-call failures across turns.
                    meta["prior_tool_msgs"] = sum(
                        1 for m in msgs
                        if isinstance(m, dict) and m.get("role") == "tool"
                    )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
                # Inspect response for finish_reason if it's a non-streaming JSON
                try:
                    txt = open(f"{base}.resp.txt", "rb").read()
                    if not (rb.get("stream") if isinstance(rb, dict) else False):
                        rj = json.loads(txt)
                        meta["response_id"] = rj.get("id")
                        ch = rj.get("choices", [{}])[0]
                        meta["finish_reason"] = ch.get("finish_reason")
                        msg = ch.get("message", {})
                        cstr = msg.get("content") or ""
                        rstr = msg.get("reasoning_content") or ""
                        meta["completion_chars"] = len(cstr)
                        meta["reasoning_chars"] = len(rstr)
                        meta["has_close_think"] = "</think>" in cstr
                        meta["has_open_think"] = "<think>" in cstr
                        post = cstr.split("</think>", 1)[1] if "</think>" in cstr else None
                        if post is not None:
                            meta["post_think_chars"] = len(post)
                            meta["post_think_first40"] = post.lstrip()[:40]
                        meta["completion_tokens"] = (rj.get("usage") or {}).get(
                            "completion_tokens"
                        )
                        meta["prompt_tokens"] = (rj.get("usage") or {}).get(
                            "prompt_tokens"
                        )
                        if meta.get("post_think_chars") is not None and \
                           meta["post_think_chars"] < 5:
                            meta["SUSPECT_TRUNC"] = True
                        # Tool calls (non-stream): already assembled by server.
                        tcs = msg.get("tool_calls") or []
                        meta["tool_calls"] = tcs
                        meta["tool_call_count"] = len(tcs)
                        arg_errors = 0
                        for tc in tcs:
                            args = (tc.get("function") or {}).get("arguments", "")
                            if args:
                                try:
                                    json.loads(args)
                                except json.JSONDecodeError as e:
                                    tc["_args_parse_error"] = str(e)
                                    arg_errors += 1
                        meta["tool_call_arg_errors"] = arg_errors
                    else:
                        # SSE stream — single full pass to assemble tool_calls,
                        # split content vs reasoning_content, and pick up usage
                        # if include_usage was set.
                        chunks = [l for l in txt.split(b"\n") if l.startswith(b"data: ")]
                        meta["sse_chunks"] = len(chunks)
                        tool_buf = {}
                        content_chars = 0
                        reasoning_chars = 0
                        for c in chunks:
                            payload = c[6:].strip()
                            if payload in (b"[DONE]", b""):
                                continue
                            try:
                                cj = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            rid = cj.get("id")
                            if rid and "response_id" not in meta:
                                meta["response_id"] = rid
                            u = cj.get("usage")
                            if u:
                                meta["prompt_tokens"] = (
                                    u.get("prompt_tokens") or meta.get("prompt_tokens")
                                )
                                meta["completion_tokens"] = (
                                    u.get("completion_tokens") or meta.get("completion_tokens")
                                )
                            for choice in cj.get("choices") or []:
                                d = choice.get("delta") or {}
                                if d.get("content"):
                                    content_chars += len(d["content"])
                                if d.get("reasoning_content"):
                                    reasoning_chars += len(d["reasoning_content"])
                                for tc in d.get("tool_calls") or []:
                                    idx = tc.get("index", 0)
                                    slot = tool_buf.setdefault(idx, {
                                        "id": None, "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    })
                                    if tc.get("id"):
                                        slot["id"] = tc["id"]
                                    if tc.get("type"):
                                        slot["type"] = tc["type"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        slot["function"]["name"] += fn["name"]
                                    if fn.get("arguments"):
                                        slot["function"]["arguments"] += fn["arguments"]
                                if choice.get("finish_reason"):
                                    meta["finish_reason"] = choice["finish_reason"]
                        meta["completion_chars"] = content_chars
                        meta["reasoning_chars"] = reasoning_chars
                        tcs = list(tool_buf.values())
                        meta["tool_calls"] = tcs
                        meta["tool_call_count"] = len(tcs)
                        arg_errors = 0
                        for tc in tcs:
                            args = (tc.get("function") or {}).get("arguments", "")
                            if args:
                                try:
                                    json.loads(args)
                                except json.JSONDecodeError as e:
                                    tc["_args_parse_error"] = str(e)
                                    arg_errors += 1
                        meta["tool_call_arg_errors"] = arg_errors
                except (OSError, json.JSONDecodeError, KeyError, IndexError):
                    pass

                with open(f"{base}.meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

            return resp
    finally:
        await session.close()


async def main():
    # 64 MB cap: the proxy buffers the whole body in RAM before forwarding, so
    # an unbounded cap let a few concurrent large POSTs exhaust memory. 64 MB
    # still comfortably fits multimodal base64 image/short-audio bodies while
    # killing the DoS amplification. Raise only if a real payload needs it.
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", proxy)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, LISTEN_HOST, LISTEN_PORT)
    await site.start()
    print(f"log-proxy: listening {LISTEN_HOST}:{LISTEN_PORT} → {UPSTREAM}, logs to {LOG_DIR}/", flush=True)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
