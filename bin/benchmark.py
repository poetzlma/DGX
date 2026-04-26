#!/usr/bin/env python3
"""
LLM Serving Benchmark — coding agent workload profile.

Measures TTFT, TPOT, tok/s for single and concurrent requests
against an OpenAI-compatible endpoint (llama-swap / vLLM).

Usage:
    python benchmark.py [--model MODEL] [--base-url URL] [--concurrency 1,2,4]
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp

# ---------------------------------------------------------------------------
# Realistic coding-agent prompts (varying input sizes)
# ---------------------------------------------------------------------------
PROMPTS = [
    {
        "name": "bug-fix (short input)",
        "messages": [
            {"role": "system", "content": "You are a senior software engineer. Be concise."},
            {"role": "user", "content": (
                "Fix the off-by-one error in this Python function:\n\n"
                "```python\n"
                "def binary_search(arr, target):\n"
                "    lo, hi = 0, len(arr)\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            lo = mid\n"
                "        else:\n"
                "            hi = mid\n"
                "    return -1\n"
                "```\n\n"
                "Explain the bug and provide the corrected code."
            )},
        ],
        "max_tokens": 512,
    },
    {
        "name": "refactor (medium input)",
        "messages": [
            {"role": "system", "content": "You are a senior software engineer. Be concise and precise."},
            {"role": "user", "content": (
                "Refactor this Express.js middleware stack to use async/await "
                "and proper error handling. The current code has callback hell "
                "and swallows errors:\n\n"
                "```javascript\n"
                "const express = require('express');\n"
                "const app = express();\n"
                "const db = require('./db');\n"
                "const cache = require('./cache');\n"
                "const auth = require('./auth');\n\n"
                "app.get('/api/users/:id', function(req, res) {\n"
                "  auth.verify(req.headers.authorization, function(err, user) {\n"
                "    if (err) { res.status(401).send('Unauthorized'); return; }\n"
                "    cache.get('user:' + req.params.id, function(err, cached) {\n"
                "      if (cached) { res.json(JSON.parse(cached)); return; }\n"
                "      db.query('SELECT * FROM users WHERE id = ?', [req.params.id], function(err, rows) {\n"
                "        if (err) { console.log(err); res.status(500).send('Error'); return; }\n"
                "        if (rows.length === 0) { res.status(404).send('Not found'); return; }\n"
                "        var user = rows[0];\n"
                "        db.query('SELECT * FROM orders WHERE user_id = ?', [user.id], function(err, orders) {\n"
                "          if (err) { console.log(err); res.status(500).send('Error'); return; }\n"
                "          user.orders = orders;\n"
                "          db.query('SELECT * FROM preferences WHERE user_id = ?', [user.id], function(err, prefs) {\n"
                "            user.preferences = prefs || {};\n"
                "            cache.set('user:' + user.id, JSON.stringify(user), 300);\n"
                "            res.json(user);\n"
                "          });\n"
                "        });\n"
                "      });\n"
                "    });\n"
                "  });\n"
                "});\n\n"
                "app.get('/api/users/:id/analytics', function(req, res) {\n"
                "  auth.verify(req.headers.authorization, function(err, user) {\n"
                "    if (err) { res.status(401).send('Unauthorized'); return; }\n"
                "    db.query('SELECT * FROM page_views WHERE user_id = ? ORDER BY ts DESC LIMIT 100',\n"
                "      [req.params.id], function(err, views) {\n"
                "        if (err) { res.status(500).send('Error'); return; }\n"
                "        db.query('SELECT action, COUNT(*) as cnt FROM events WHERE user_id = ? GROUP BY action',\n"
                "          [req.params.id], function(err, events) {\n"
                "            if (err) { res.status(500).send('Error'); return; }\n"
                "            res.json({ views: views, events: events });\n"
                "          });\n"
                "      });\n"
                "  });\n"
                "});\n\n"
                "module.exports = app;\n"
                "```\n\n"
                "Provide the fully refactored code with:\n"
                "1. async/await throughout\n"
                "2. Centralized error handling middleware\n"
                "3. Input validation\n"
                "4. Proper HTTP status codes"
            )},
        ],
        "max_tokens": 1024,
    },
    {
        "name": "architecture (long output)",
        "messages": [
            {"role": "system", "content": "You are a senior software architect."},
            {"role": "user", "content": (
                "Design a real-time collaborative code editor backend in Python. "
                "Requirements:\n"
                "- WebSocket-based with operational transform (OT) or CRDT\n"
                "- Support 50 concurrent editors per document\n"
                "- Cursor presence and selection awareness\n"
                "- Undo/redo per user\n"
                "- Persistence to PostgreSQL\n"
                "- Language server protocol (LSP) integration for completions\n\n"
                "Provide:\n"
                "1. System architecture diagram (ASCII)\n"
                "2. Key data structures\n"
                "3. WebSocket message protocol\n"
                "4. OT/CRDT algorithm choice with justification\n"
                "5. Core Python implementation (~200 lines)\n"
                "6. Load testing strategy"
            )},
        ],
        "max_tokens": 2048,
    },
    {
        "name": "code-review (large context)",
        "messages": [
            {"role": "system", "content": "You are a senior code reviewer. Be thorough but concise."},
            {"role": "user", "content": (
                "Review this Rust HTTP server implementation for correctness, "
                "safety, and performance issues:\n\n"
                "```rust\n"
                "use std::collections::HashMap;\n"
                "use std::io::{Read, Write, BufRead, BufReader};\n"
                "use std::net::{TcpListener, TcpStream};\n"
                "use std::sync::{Arc, Mutex, RwLock};\n"
                "use std::thread;\n"
                "use std::time::{Duration, Instant};\n"
                "use std::fs;\n\n"
                "struct RateLimiter {\n"
                "    requests: HashMap<String, Vec<Instant>>,\n"
                "    max_requests: usize,\n"
                "    window: Duration,\n"
                "}\n\n"
                "impl RateLimiter {\n"
                "    fn new(max_requests: usize, window_secs: u64) -> Self {\n"
                "        RateLimiter {\n"
                "            requests: HashMap::new(),\n"
                "            max_requests,\n"
                "            window: Duration::from_secs(window_secs),\n"
                "        }\n"
                "    }\n\n"
                "    fn check(&mut self, ip: &str) -> bool {\n"
                "        let now = Instant::now();\n"
                "        let entry = self.requests.entry(ip.to_string()).or_insert_with(Vec::new);\n"
                "        entry.retain(|t| now.duration_since(*t) < self.window);\n"
                "        if entry.len() >= self.max_requests {\n"
                "            return false;\n"
                "        }\n"
                "        entry.push(now);\n"
                "        true\n"
                "    }\n"
                "}\n\n"
                "struct SessionStore {\n"
                "    sessions: HashMap<String, (String, Instant)>,\n"
                "    ttl: Duration,\n"
                "}\n\n"
                "impl SessionStore {\n"
                "    fn new(ttl_secs: u64) -> Self {\n"
                "        SessionStore {\n"
                "            sessions: HashMap::new(),\n"
                "            ttl: Duration::from_secs(ttl_secs),\n"
                "        }\n"
                "    }\n\n"
                "    fn get(&self, token: &str) -> Option<&String> {\n"
                "        self.sessions.get(token).and_then(|(user, created)| {\n"
                "            if created.elapsed() < self.ttl {\n"
                "                Some(user)\n"
                "            } else {\n"
                "                None\n"
                "            }\n"
                "        })\n"
                "    }\n\n"
                "    fn set(&mut self, token: String, user: String) {\n"
                "        self.sessions.insert(token, (user, Instant::now()));\n"
                "    }\n"
                "}\n\n"
                "struct Router {\n"
                "    routes: Vec<(String, String, Box<dyn Fn(&Request) -> Response + Send + Sync>)>,\n"
                "}\n\n"
                "struct Request {\n"
                "    method: String,\n"
                "    path: String,\n"
                "    headers: HashMap<String, String>,\n"
                "    body: String,\n"
                "    params: HashMap<String, String>,\n"
                "}\n\n"
                "struct Response {\n"
                "    status: u16,\n"
                "    headers: HashMap<String, String>,\n"
                "    body: String,\n"
                "}\n\n"
                "impl Response {\n"
                "    fn ok(body: String) -> Self {\n"
                "        let mut headers = HashMap::new();\n"
                "        headers.insert(\"Content-Type\".into(), \"application/json\".into());\n"
                "        Response { status: 200, headers, body }\n"
                "    }\n"
                "    fn not_found() -> Self {\n"
                "        Response { status: 404, headers: HashMap::new(), body: \"Not Found\".into() }\n"
                "    }\n"
                "    fn error(msg: &str) -> Self {\n"
                "        Response { status: 500, headers: HashMap::new(), body: msg.to_string() }\n"
                "    }\n"
                "    fn rate_limited() -> Self {\n"
                "        Response { status: 429, headers: HashMap::new(), body: \"Too Many Requests\".into() }\n"
                "    }\n"
                "}\n\n"
                "fn parse_request(stream: &mut BufReader<&TcpStream>) -> Option<Request> {\n"
                "    let mut first_line = String::new();\n"
                "    stream.read_line(&mut first_line).ok()?;\n"
                "    let parts: Vec<&str> = first_line.trim().split_whitespace().collect();\n"
                "    if parts.len() < 2 { return None; }\n"
                "    let method = parts[0].to_string();\n"
                "    let path = parts[1].to_string();\n"
                "    let mut headers = HashMap::new();\n"
                "    loop {\n"
                "        let mut line = String::new();\n"
                "        stream.read_line(&mut line).ok()?;\n"
                "        let line = line.trim().to_string();\n"
                "        if line.is_empty() { break; }\n"
                "        if let Some((k, v)) = line.split_once(':') {\n"
                "            headers.insert(k.trim().to_lowercase(), v.trim().to_string());\n"
                "        }\n"
                "    }\n"
                "    let content_length: usize = headers.get(\"content-length\")\n"
                "        .and_then(|v| v.parse().ok()).unwrap_or(0);\n"
                "    let mut body = vec![0u8; content_length];\n"
                "    if content_length > 0 {\n"
                "        stream.read_exact(&mut body).ok()?;\n"
                "    }\n"
                "    Some(Request {\n"
                "        method, path, headers,\n"
                "        body: String::from_utf8_lossy(&body).to_string(),\n"
                "        params: HashMap::new(),\n"
                "    })\n"
                "}\n\n"
                "fn main() {\n"
                "    let listener = TcpListener::bind(\"0.0.0.0:8080\").unwrap();\n"
                "    let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(100, 60)));\n"
                "    let sessions = Arc::new(RwLock::new(SessionStore::new(3600)));\n"
                "    let counter = Arc::new(Mutex::new(0u64));\n\n"
                "    for stream in listener.incoming() {\n"
                "        let stream = stream.unwrap();\n"
                "        let rl = rate_limiter.clone();\n"
                "        let sess = sessions.clone();\n"
                "        let ctr = counter.clone();\n"
                "        thread::spawn(move || {\n"
                "            let peer = stream.peer_addr().unwrap().ip().to_string();\n"
                "            let mut reader = BufReader::new(&stream);\n"
                "            if let Some(req) = parse_request(&mut reader) {\n"
                "                let allowed = rl.lock().unwrap().check(&peer);\n"
                "                if !allowed {\n"
                "                    send_response(&stream, Response::rate_limited());\n"
                "                    return;\n"
                "                }\n"
                "                let mut count = ctr.lock().unwrap();\n"
                "                *count += 1;\n"
                "                let resp = match (req.method.as_str(), req.path.as_str()) {\n"
                "                    (\"GET\", \"/health\") => Response::ok('{\"status\":\"ok\"}'.to_string()),\n"
                "                    (\"GET\", \"/metrics\") => Response::ok(format!('{{\"requests\":{}}}', count)),\n"
                "                    _ => Response::not_found(),\n"
                "                };\n"
                "                send_response(&stream, resp);\n"
                "            }\n"
                "        });\n"
                "    }\n"
                "}\n\n"
                "fn send_response(mut stream: &TcpStream, resp: Response) {\n"
                "    let status_text = match resp.status {\n"
                "        200 => \"OK\", 404 => \"Not Found\", 429 => \"Too Many Requests\",\n"
                "        500 => \"Internal Server Error\", _ => \"Unknown\",\n"
                "    };\n"
                "    let header_str: String = resp.headers.iter()\n"
                "        .map(|(k, v)| format!(\"{}: {}\\r\\n\", k, v)).collect();\n"
                "    let response = format!(\n"
                "        \"HTTP/1.1 {} {}\\r\\n{}Content-Length: {}\\r\\n\\r\\n{}\",\n"
                "        resp.status, status_text, header_str, resp.body.len(), resp.body\n"
                "    );\n"
                "    stream.write_all(response.as_bytes()).ok();\n"
                "    stream.flush().ok();\n"
                "}\n"
                "```\n\n"
                "Focus on: memory leaks, race conditions, DoS vectors, "
                "missing error handling, and performance bottlenecks."
            )},
        ],
        "max_tokens": 1536,
    },
]


@dataclass
class RequestResult:
    prompt_name: str
    ttft: float           # seconds
    total_time: float     # seconds
    input_tokens: int
    output_tokens: int
    tpot: float           # seconds per output token
    tok_per_sec: float    # output tokens / total_time
    error: str | None = None


async def stream_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: dict,
) -> RequestResult:
    """Send a single streaming request and measure timing."""
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt["max_tokens"],
        "stream": True,
        "temperature": 0.7,
    }

    t_start = time.monotonic()
    t_first_token = None
    output_tokens = 0
    input_tokens = 0
    error = None
    token_times: list[float] = []

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    prompt_name=prompt["name"],
                    ttft=0, total_time=0, input_tokens=0, output_tokens=0,
                    tpot=0, tok_per_sec=0,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Extract usage from final chunk if present
                if "usage" in chunk and chunk["usage"]:
                    input_tokens = chunk["usage"].get("prompt_tokens", 0)
                    output_tokens = chunk["usage"].get("completion_tokens", output_tokens)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # Qwen3.6 streams reasoning tokens as "reasoning"
                # and final answer tokens as "content"
                text = delta.get("content", "") or delta.get("reasoning", "")
                if text:
                    now = time.monotonic()
                    if t_first_token is None:
                        t_first_token = now
                    token_times.append(now)
                    output_tokens += 1

    except Exception as e:
        error = str(e)

    t_end = time.monotonic()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time

    # If server returned usage with completion_tokens, use that and reset our count
    # Otherwise our streaming count is the best estimate
    if output_tokens == 0:
        output_tokens = len(token_times)

    tpot = (total_time - ttft) / max(output_tokens - 1, 1) if output_tokens > 0 else 0
    tok_s = output_tokens / total_time if total_time > 0 else 0

    return RequestResult(
        prompt_name=prompt["name"],
        ttft=ttft,
        total_time=total_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tpot=tpot,
        tok_per_sec=tok_s,
        error=error,
    )


async def run_batch(
    base_url: str,
    model: str,
    concurrency: int,
    prompts: list[dict],
) -> list[RequestResult]:
    """Run prompts concurrently up to `concurrency` level."""
    connector = aiohttp.TCPConnector(limit=concurrency + 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Each concurrent "agent" gets a different prompt (round-robin)
        tasks = []
        for i in range(concurrency):
            prompt = prompts[i % len(prompts)]
            tasks.append(stream_request(session, base_url, model, prompt))
        return await asyncio.gather(*tasks)


def print_results(results: list[RequestResult], concurrency: int):
    """Pretty-print benchmark results for a concurrency level."""
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print(f"\n{'='*70}")
    print(f"  Concurrency: {concurrency} agent(s)")
    print(f"{'='*70}")

    if failed:
        for r in failed:
            print(f"  FAILED [{r.prompt_name}]: {r.error}")

    if not ok:
        print("  All requests failed!")
        return

    for r in ok:
        print(f"\n  [{r.prompt_name}]")
        print(f"    TTFT:          {r.ttft*1000:>8.0f} ms")
        print(f"    Total time:    {r.total_time:>8.2f} s")
        print(f"    Output tokens: {r.output_tokens:>8d}")
        print(f"    TPOT:          {r.tpot*1000:>8.1f} ms/tok")
        print(f"    Throughput:    {r.tok_per_sec:>8.1f} tok/s")

    # Aggregate
    ttfts = [r.ttft for r in ok]
    tpots = [r.tpot for r in ok]
    toks = [r.tok_per_sec for r in ok]
    total_toks = sum(r.output_tokens for r in ok)
    wall_time = max(r.total_time for r in ok)
    agg_throughput = total_toks / wall_time if wall_time > 0 else 0

    print(f"\n  --- Aggregate (n={len(ok)}) ---")
    print(f"    Avg TTFT:            {statistics.mean(ttfts)*1000:>8.0f} ms")
    print(f"    Avg TPOT:            {statistics.mean(tpots)*1000:>8.1f} ms/tok")
    print(f"    Avg tok/s per req:   {statistics.mean(toks):>8.1f}")
    print(f"    Total tok/s (agg):   {agg_throughput:>8.1f}  ({total_toks} tokens in {wall_time:.1f}s)")


async def main():
    parser = argparse.ArgumentParser(description="LLM coding-agent benchmark")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--base-url", default="http://192.168.1.12:8080")
    parser.add_argument(
        "--concurrency", default="1,2,4",
        help="Comma-separated concurrency levels (default: 1,2,4)",
    )
    parser.add_argument("--warmup", action="store_true", help="Send a warmup request first")
    args = parser.parse_args()

    levels = [int(c) for c in args.concurrency.split(",")]

    print(f"Benchmark: {args.model} @ {args.base_url}")
    print(f"Concurrency levels: {levels}")
    print(f"Prompts: {', '.join(p['name'] for p in PROMPTS)}")

    if args.warmup:
        print("\nSending warmup request (may trigger cold-start)...")
        connector = aiohttp.TCPConnector(limit=2)
        async with aiohttp.ClientSession(connector=connector) as session:
            r = await stream_request(session, args.base_url, args.model, PROMPTS[0])
            if r.error:
                print(f"  Warmup failed: {r.error}")
                print("  Model may not be loaded. Waiting 30s and retrying...")
                await asyncio.sleep(30)
                r = await stream_request(session, args.base_url, args.model, PROMPTS[0])
            print(f"  Warmup done: {r.total_time:.1f}s, {r.tok_per_sec:.1f} tok/s")

    for level in levels:
        results = await run_batch(args.base_url, args.model, level, PROMPTS)
        print_results(results, level)

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
