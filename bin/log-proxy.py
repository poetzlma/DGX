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
LOG_DIR = "/tmp/log-proxy"

os.makedirs(LOG_DIR, exist_ok=True)


def _ts():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


async def proxy(request: web.Request) -> web.StreamResponse:
    reqid = uuid.uuid4().hex[:12]
    ts = _ts()
    base = f"{LOG_DIR}/{ts}-{reqid}"
    is_chat = request.path == "/v1/chat/completions" and request.method == "POST"

    body_bytes = await request.read()
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
            try:
                async for chunk in upstream.content.iter_any():
                    chunks_written += len(chunk)
                    if log_f:
                        log_f.write(chunk)
                    await resp.write(chunk)
            finally:
                if log_f:
                    log_f.close()

            await resp.write_eof()
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

                # ── Auth: capture Authorization for downstream user/key
                # enrichment. Stores prefix (display form) only; the full
                # token is in /tmp/log-proxy and never leaves the host.
                auth = request.headers.get("Authorization", "")
                if auth.lower().startswith("bearer "):
                    token = auth[7:].strip()
                    meta["key_full"] = token
                    if len(token) > 8:
                        meta["key_masked"] = f"{token[:6]}…{token[-4:]}"
                    else:
                        meta["key_masked"] = "…"
                    meta["key_prefix"] = token[:8]
                # Pull a few notable fields out of the request body
                try:
                    rb = json.loads(body_bytes)
                    meta["request_model"] = rb.get("model")
                    msgs = rb.get("messages") or []
                    meta["n_messages"] = len(msgs)
                    meta["tools_count"] = len(rb.get("tools") or [])
                    meta["max_tokens"] = rb.get("max_tokens")
                    meta["stream"] = rb.get("stream", False)
                    last = msgs[-1] if msgs else {}
                    last_content = last.get("content") if isinstance(last, dict) else None
                    if isinstance(last_content, str):
                        meta["last_user_content_chars"] = len(last_content)
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
                        meta["completion_chars"] = len(cstr)
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
                    else:
                        # SSE stream — pull response id from first chunk +
                        # finish_reason from last meaningful chunk.
                        chunks = [l for l in txt.split(b"\n") if l.startswith(b"data: ")]
                        meta["sse_chunks"] = len(chunks)
                        for c in chunks:
                            payload = c[6:].strip()
                            if payload in (b"[DONE]", b""):
                                continue
                            try:
                                cj = json.loads(payload)
                                rid = cj.get("id")
                                if rid:
                                    meta["response_id"] = rid
                                    break
                            except json.JSONDecodeError:
                                continue
                        for c in reversed(chunks):
                            payload = c[6:].strip()
                            if payload in (b"[DONE]", b""):
                                continue
                            try:
                                cj = json.loads(payload)
                                fr = (cj.get("choices") or [{}])[0].get("finish_reason")
                                if fr:
                                    meta["finish_reason"] = fr
                                    break
                            except json.JSONDecodeError:
                                continue
                except (OSError, json.JSONDecodeError, KeyError, IndexError):
                    pass

                with open(f"{base}.meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

            return resp
    finally:
        await session.close()


async def main():
    app = web.Application(client_max_size=1024 * 1024 * 1024)  # 1 GB cap
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
