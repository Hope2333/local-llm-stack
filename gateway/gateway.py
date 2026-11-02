#!/usr/bin/env python3
"""Multi-dialect LLM gateway: one port, OpenAI / Anthropic / Ollama / Gemini shapes."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import socket
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HOME = os.path.expanduser("~")
REG_PATH = os.environ.get("GATEWAY_REGISTRY", f"{HOME}/models/llm_gateway/registry.json")
LLAMA_BIN = os.environ.get("GATEWAY_LLAMA_BIN", f"{HOME}/opt/llama-prism/bin/llama-server")
HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GATEWAY_PORT", "8100"))

with open(REG_PATH) as fh:
    REG = json.load(fh)


def expand(path: Any) -> str:
    return os.path.expanduser(str(path))
MODELS: dict[str, Any] = REG["models"]
IDLE = float(os.environ.get("GATEWAY_IDLE", REG.get("idle_seconds", 300)))
POOL = REG.get("port_pool", [8310, 8399])
ALIASES = {
    "bonsai": REG["default"],
    "default": REG["default"],
    "2b": "bonsai-2b",
    "4b": "bonsai-4b",
    "9b": "bonsai-9b",
    "27b": "bonsai-27b",
}


def _die_with_parent() -> None:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9)


class Backend:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.last_used = time.monotonic()
        self.active = 0
        self.lock = asyncio.Lock()

    @property
    def static(self) -> bool:
        return self.cfg.get("backend") == "static"

    def base(self) -> str:
        if self.static:
            return str(self.cfg["url"]).rstrip("/")
        return f"http://127.0.0.1:{self.port}/v1"


BACKENDS = {name: Backend(name, cfg) for name, cfg in MODELS.items()}


def resolve(name: str | None) -> str:
    if not name:
        return REG["default"]
    raw = str(name).strip()
    low = raw.lower()
    if raw in MODELS:
        return raw
    if low in ALIASES:
        return ALIASES[low]
    for key in MODELS:
        if low == key or low.endswith(key) or key.endswith(low):
            return key
    for key in MODELS:
        if low in key:
            return key
    return REG["default"]


def _pick_port() -> int:
    taken = {b.port for b in BACKENDS.values() if b.port}
    for candidate in range(POOL[0], POOL[1] + 1):
        if candidate in taken:
            continue
        with socket.socket() as sock:
            try:
                sock.bind((HOST, candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError("no free backend port")


async def ensure(name: str) -> Backend:
    backend = BACKENDS[name]
    backend.last_used = time.monotonic()
    if backend.static:
        return backend
    async with backend.lock:
        if backend.proc is not None and backend.proc.poll() is None:
            return backend
        backend.port = _pick_port()
        cfg = backend.cfg
        for other in BACKENDS.values():
            if other is backend or other.static or other.proc is None:
                continue
            if other.proc.poll() is None:
                other.proc.terminate()
                try:
                    other.proc.wait(timeout=15)
                except Exception:
                    other.proc.kill()
                other.proc = None
        args = [
            LLAMA_BIN,
            "--model", expand(cfg["path"]),
            "--ctx-size", str(cfg.get("ctx", 32768)),
            "--flash-attn", "on",
            "--jinja",
            "--reasoning", "off",
            "--host", HOST,
            "--port", str(backend.port),
            "--alias", name,
            "--threads", str(cfg.get("threads", 12)),
            "--temp", "0.7", "--top-p", "0.80", "--min-p", "0",
            "--presence-penalty", "1.0", "--repeat-penalty", "1.0", "--repeat-last-n", "64",
            "--n-gpu-layers", str(cfg.get("n_gpu_layers", 0)),
            "--device", str(cfg.get("device", "none")),
        ]
        if cfg.get("override"):
            # 显存放不下长上下文时，把 FFN 张量下沉到 CPU、注意力与 SSM 状态留在 GPU，
            # 这样 KV 不必每个 token 都过 PCIe（比 -nkvo 快 2~3 倍）
            ov = cfg["override"]
            args += ["-ot", ov if isinstance(ov, str) else ",".join(ov)]
        if cfg.get("mmproj"):
            args += ["--mmproj", expand(cfg["mmproj"])]
        if cfg.get("cache_type"):
            args += ["--cache-type-k", str(cfg["cache_type"]), "--cache-type-v", str(cfg["cache_type"])]
        if cfg.get("mtp"):
            args += [
                "--model-draft", expand(cfg["mtp"]),
                "--spec-type", cfg.get("spec_type", "draft-mtp"),
                "--spec-draft-n-max", str(cfg.get("spec_n_max", 4)),
                "--spec-draft-n-min", str(cfg.get("spec_n_min", 0)),
                "--spec-draft-p-min", str(cfg.get("spec_p_min", 0)),
            ]
        elif cfg.get("spec_type"):
            args += ["--spec-type", str(cfg["spec_type"])]
        log_path = f"/tmp/llama-backend-{name}.log"
        backend.proc = subprocess.Popen(
            args, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=_die_with_parent,
        )
        for _ in range(360):
            await asyncio.sleep(0.5)
            if backend.proc.poll() is not None:
                backend.proc = None
                raise RuntimeError(f"{name}: llama-server exited during startup")
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"http://127.0.0.1:{backend.port}/health")
                if resp.status_code == 200:
                    return backend
            except Exception:
                pass
        raise RuntimeError(f"{name}: startup timeout")


async def _reaper() -> None:
    while True:
        await asyncio.sleep(10)
        now = time.monotonic()
        for backend in BACKENDS.values():
            if backend.static or backend.proc is None or backend.proc.poll() is not None:
                continue
            if backend.active == 0 and now - backend.last_used > IDLE:
                backend.proc.terminate()
                try:
                    backend.proc.wait(10)
                except subprocess.TimeoutExpired:
                    backend.proc.kill()
                backend.proc = None
                backend.port = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_reaper())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)


async def post_json(backend: Backend, path: str, payload: dict[str, Any]) -> httpx.Response:
    backend.active += 1
    backend.last_used = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=1800.0) as client:
            return await client.post(backend.base() + path, json=payload)
    finally:
        backend.active -= 1
        backend.last_used = time.monotonic()


async def post_json_root(root: str, path: str, payload: dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120.0) as client:
        return await client.post(root + path, json=payload)


async def open_stream(backend: Backend, path: str, payload: dict[str, Any]):
    backend.active += 1
    backend.last_used = time.monotonic()
    client = httpx.AsyncClient(timeout=httpx.Timeout(1800.0, connect=10.0))
    try:
        request = client.build_request("POST", backend.base() + path, json=payload)
        response = await client.send(request, stream=True)
    except Exception:
        backend.active -= 1
        await client.aclose()
        raise
    return client, response


async def iter_openai_chunks(response: httpx.Response):
    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def data_url(media_type: str, data: str) -> str:
    return f"data:{media_type or 'image/png'};base64,{data}"


def error_payload(style: str, message: str, status: int = 400) -> dict[str, Any]:
    if style == "anthropic":
        return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    if style == "ollama":
        return {"error": message}
    if style == "gemini":
        return {"error": {"code": status, "message": message, "status": "INVALID_ARGUMENT"}}
    return {"error": {"message": message, "type": "invalid_request_error", "code": status}}


def model_card(name: str) -> dict[str, Any]:
    cfg = MODELS[name]
    return {
        "id": name,
        "object": "model",
        "created": 0,
        "owned_by": "local",
        "context_length": cfg.get("ctx", 32768),
        "capabilities": ["completion", "chat"] + (["vision"] if cfg.get("mmproj") or cfg.get("kind") == "vision" else []),
        "meta": {"parameters": cfg.get("params"), "quantization": cfg.get("quant"), "backend": cfg.get("backend")},
    }


@app.get("/")
async def index():
    return {
        "service": "llm-gateway",
        "models": list(MODELS),
        "default": REG["default"],
        "dialects": {
            "openai": ["POST /v1/chat/completions", "POST /v1/completions", "POST /v1/responses", "GET /v1/models"],
            "anthropic": ["POST /v1/messages", "POST /v1/messages/count_tokens"],
            "ollama": ["POST /api/chat", "POST /api/generate", "GET /api/tags", "POST /api/show", "GET /api/ps"],
            "gemini": ["POST /v1beta/models/{model}:generateContent", "POST /v1beta/models/{model}:streamGenerateContent", "GET /v1beta/models"],
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": {
            name: {
                "static": backend.static,
                "running": backend.static or (backend.proc is not None and backend.proc.poll() is None),
                "port": backend.port,
                "active": backend.active,
                "idle_seconds": round(time.monotonic() - backend.last_used, 1),
            }
            for name, backend in BACKENDS.items()
        },
    }


@app.get("/v1/models")
async def openai_models():
    return {"object": "list", "data": [model_card(name) for name in MODELS]}


@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    body["model"] = name
    backend = await ensure(name)
    if body.get("stream"):
        client, response = await open_stream(backend, "/chat/completions", body)
        if response.status_code >= 400:
            raw = await response.aread()
            await response.aclose()
            await client.aclose()
            return JSONResponse(json.loads(raw or b"{}"), status_code=response.status_code)
        return StreamingResponse(_relay_sse(response, client), media_type="text/event-stream")
    response = await post_json(backend, "/chat/completions", body)
    return JSONResponse(response.json(), status_code=response.status_code)


async def _relay_sse(response: httpx.Response, client: httpx.AsyncClient):
    try:
        async for line in response.aiter_lines():
            if line.strip():
                yield (line + "\n\n").encode()
    finally:
        await response.aclose()
        await client.aclose()


@app.post("/v1/completions")
async def openai_completions(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    body["model"] = name
    backend = await ensure(name)
    if body.get("stream"):
        client, response = await open_stream(backend, "/completions", body)
        if response.status_code >= 400:
            raw = await response.aread()
            await response.aclose()
            await client.aclose()
            return JSONResponse(json.loads(raw or b"{}"), status_code=response.status_code)
        return StreamingResponse(_relay_sse(response, client), media_type="text/event-stream")
    response = await post_json(backend, "/completions", body)
    return JSONResponse(response.json(), status_code=response.status_code)


def responses_input_to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    raw = body.get("input")
    if isinstance(raw, str):
        messages.append({"role": "user", "content": raw})
        return messages
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts = []
        for block in content or []:
            kind = block.get("type")
            if kind in ("input_text", "text", "output_text"):
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "input_image":
                url = block.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts:
            messages.append({"role": role, "content": parts})
    return messages


@app.post("/v1/responses")
async def openai_responses(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    messages = responses_input_to_messages(body)
    payload: dict[str, Any] = {"model": name, "messages": messages}
    if body.get("max_output_tokens"):
        payload["max_tokens"] = body["max_output_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    backend = await ensure(name)
    response = await post_json(backend, "/chat/completions", payload)
    if response.status_code >= 400:
        return JSONResponse(response.json(), status_code=response.status_code)
    result = response.json()
    text = result["choices"][0]["message"].get("content") or ""
    usage = result.get("usage", {})
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "model": name,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "output_text": text,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "".join(part.get("text", "") for part in system if isinstance(part, dict))
        if text:
            messages.append({"role": "system", "content": text})
    for message in body.get("messages", []):
        role = "assistant" if message.get("role") == "assistant" else "user"
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url(source.get("media_type"), source.get("data", ""))},
                    })
            elif kind == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "".join(p.get("text", "") for p in inner if isinstance(p, dict))
                parts.append({"type": "text", "text": str(inner)})
        if parts:
            messages.append({"role": role, "content": parts})
    payload: dict[str, Any] = {"messages": messages}
    if body.get("max_tokens"):
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    return payload


ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    payload = anthropic_to_openai(body)
    payload["model"] = name
    backend = await ensure(name)
    if body.get("stream"):
        payload["stream"] = True
        client, response = await open_stream(backend, "/chat/completions", payload)
        if response.status_code >= 400:
            raw = await response.aread()
            await response.aclose()
            await client.aclose()
            return JSONResponse(error_payload("anthropic", raw.decode("utf-8", "replace")), status_code=response.status_code)
        return StreamingResponse(_anthropic_stream(response, client, name), media_type="text/event-stream")
    response = await post_json(backend, "/chat/completions", payload)
    if response.status_code >= 400:
        return JSONResponse(error_payload("anthropic", str(response.text), response.status_code), status_code=response.status_code)
    result = response.json()
    choice = result["choices"][0]
    usage = result.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": name,
        "content": [{"type": "text", "text": choice["message"].get("content") or ""}],
        "stop_reason": ANTHROPIC_STOP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def _anthropic_stream(response: httpx.Response, client: httpx.AsyncClient, name: str):
    def event(kind: str, data: dict[str, Any]) -> bytes:
        return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    try:
        yield event("message_start", {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
                "model": name, "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        output_tokens = 0
        async for chunk in iter_openai_chunks(response):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                yield event("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                })
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                output_tokens = usage["completion_tokens"]
        yield event("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield event("message_stop", {"type": "message_stop"})
    finally:
        await response.aclose()
        await client.aclose()


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    payload = anthropic_to_openai(body)
    backend = await ensure(name)
    flat = "\n".join(
        part if isinstance(m.get("content"), str) else
        "".join(p.get("text", "") for p in m.get("content", []) if isinstance(p, dict))
        for m in payload["messages"]
        for part in [m.get("content")]
    )
    root = backend.base()[:-3] if backend.base().endswith("/v1") else backend.base()
    try:
        response = await post_json_root(root, "/tokenize", {"content": flat})
        if response.status_code == 200:
            return {"input_tokens": len(response.json().get("tokens", []))}
    except Exception:
        pass
    return {"input_tokens": max(1, len(flat) // 3)}


@app.get("/api/tags")
async def ollama_tags():
    cards = []
    for name, cfg in MODELS.items():
        cards.append({
            "name": name,
            "model": name,
            "modified_at": iso_now(),
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen35",
                "families": ["qwen35"],
                "parameter_size": cfg.get("params", ""),
                "quantization_level": cfg.get("quant", ""),
            },
        })
    return {"models": cards}


@app.get("/api/ps")
async def ollama_ps():
    running = []
    for name, backend in BACKENDS.items():
        live = backend.static or (backend.proc is not None and backend.proc.poll() is None)
        if live:
            running.append({
                "name": name, "model": name, "size": 0,
                "digest": "", "details": {"parameter_size": MODELS[name].get("params", "")},
            })
    return {"models": running}


@app.post("/api/show")
async def ollama_show(request: Request):
    body = await request.json()
    name = resolve(body.get("name") or body.get("model"))
    cfg = MODELS[name]
    return {
        "license": "",
        "modelfile": f"FROM {cfg.get('path', name)}",
        "parameters": f"num_ctx {cfg.get('ctx', 32768)}",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "", "format": "gguf", "family": "qwen35", "families": ["qwen35"],
            "parameter_size": cfg.get("params", ""), "quantization_level": cfg.get("quant", ""),
        },
        "model_info": {"general.architecture": "qwen35", "general.parameter_count": cfg.get("params", "")},
        "capabilities": ["completion", "vision"] if cfg.get("mmproj") or cfg.get("kind") == "vision" else ["completion"],
    }


def ollama_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    for message in body.get("messages", []):
        parts: list[dict[str, Any]] = []
        content = message.get("content", "")
        if content:
            parts.append({"type": "text", "text": content})
        for image in message.get("images") or []:
            if isinstance(image, str):
                media = "image/png"
                if "." in image[:8]:
                    media = image.split(";")[0].split(":")[1] if image.startswith("data:") else media
                parts.append({"type": "image_url", "image_url": {"url": data_url(media, image)}})
        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": message.get("role", "user"), "content": parts[0]["text"]})
        elif parts:
            messages.append({"role": message.get("role", "user"), "content": parts})
    return messages


def ollama_options(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") or {}
    payload: dict[str, Any] = {}
    if options.get("num_predict"):
        payload["max_tokens"] = options["num_predict"]
    if options.get("temperature") is not None:
        payload["temperature"] = options["temperature"]
    if options.get("top_p") is not None:
        payload["top_p"] = options["top_p"]
    if options.get("stop"):
        payload["stop"] = options["stop"]
    if body.get("keep_alive") is None:
        payload.setdefault("max_tokens", 2048)
    return payload


@app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    payload: dict[str, Any] = {"model": name, "messages": ollama_messages(body)}
    payload.update(ollama_options(body))
    backend = await ensure(name)
    want_stream = body.get("stream", True)
    if want_stream:
        return StreamingResponse(
            _ollama_chat_stream(backend, payload, name, True),
            media_type="application/x-ndjson",
        )
    response = await post_json(backend, "/chat/completions", payload)
    if response.status_code >= 400:
        return JSONResponse(error_payload("ollama", response.text), status_code=response.status_code)
    result = response.json()
    choice = result["choices"][0]
    usage = result.get("usage", {})
    return {
        "model": name, "created_at": iso_now(), "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": choice["message"].get("content") or ""},
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


async def _ollama_chat_stream(backend: Backend, payload: dict[str, Any], name: str, stream: bool):
    if not stream:
        response = await post_json(backend, "/chat/completions", payload)
        result = response.json()
        line = {
            "model": name, "created_at": iso_now(), "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": result["choices"][0]["message"].get("content") or ""},
        }
        yield (json.dumps(line, ensure_ascii=False) + "\n").encode()
        return
    upstream = dict(payload)
    upstream["stream"] = True
    client, response = await open_stream(backend, "/chat/completions", upstream)
    try:
        if response.status_code >= 400:
            raw = await response.aread()
            yield (json.dumps({"error": raw.decode("utf-8", "replace")}, ensure_ascii=False) + "\n").encode()
            return
        async for chunk in iter_openai_chunks(response):
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                yield (json.dumps({
                    "model": name, "created_at": iso_now(), "done": False,
                    "message": {"role": "assistant", "content": delta["content"]},
                }, ensure_ascii=False) + "\n").encode()
        yield (json.dumps({
            "model": name, "created_at": iso_now(), "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": ""},
        }, ensure_ascii=False) + "\n").encode()
    finally:
        await response.aclose()
        await client.aclose()


@app.post("/api/generate")
async def ollama_generate(request: Request):
    body = await request.json()
    name = resolve(body.get("model"))
    payload: dict[str, Any] = {
        "model": name,
        "messages": [{"role": "user", "content": body.get("prompt", "")}],
    }
    if body.get("system"):
        payload["messages"].insert(0, {"role": "system", "content": body["system"]})
    payload.update(ollama_options(body))
    backend = await ensure(name)
    if body.get("stream", True):
        async def stream():
            upstream = dict(payload)
            upstream["stream"] = True
            client, response = await open_stream(backend, "/chat/completions", upstream)
            try:
                async for chunk in iter_openai_chunks(response):
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield (json.dumps({
                            "model": name, "created_at": iso_now(), "done": False,
                            "response": delta["content"],
                        }, ensure_ascii=False) + "\n").encode()
                yield (json.dumps({
                    "model": name, "created_at": iso_now(), "done": True,
                    "done_reason": "stop", "response": "",
                }, ensure_ascii=False) + "\n").encode()
            finally:
                await response.aclose()
                await client.aclose()
        return StreamingResponse(stream(), media_type="application/x-ndjson")
    response = await post_json(backend, "/chat/completions", payload)
    result = response.json()
    choice = result["choices"][0]
    usage = result.get("usage", {})
    return {
        "model": name, "created_at": iso_now(), "done": True, "done_reason": "stop",
        "response": choice["message"].get("content") or "",
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


GEMINI_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP"}


def gemini_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(system, dict):
        text = "".join(p.get("text", "") for p in system.get("parts", []) if isinstance(p, dict))
        if text:
            messages.append({"role": "system", "content": text})
    for entry in body.get("contents", []):
        role = "assistant" if entry.get("role") == "model" else "user"
        parts: list[dict[str, Any]] = []
        for part in entry.get("parts", []):
            if "text" in part:
                parts.append({"type": "text", "text": part["text"]})
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url(inline.get("mimeType") or inline.get("mime_type"), inline.get("data", ""))},
                })
        if len(parts) == 1 and parts[0]["type"] == "text":
            messages.append({"role": role, "content": parts[0]["text"]})
        elif parts:
            messages.append({"role": role, "content": parts})
    payload: dict[str, Any] = {"messages": messages}
    config = body.get("generationConfig") or body.get("generation_config") or {}
    if config.get("maxOutputTokens"):
        payload["max_tokens"] = config["maxOutputTokens"]
    if config.get("temperature") is not None:
        payload["temperature"] = config["temperature"]
    if config.get("topP") is not None:
        payload["top_p"] = config["topP"]
    if config.get("stopSequences"):
        payload["stop"] = config["stopSequences"]
    return payload


def openai_to_gemini(result: dict[str, Any], name: str) -> dict[str, Any]:
    choice = result["choices"][0]
    usage = result.get("usage", {})
    return {
        "candidates": [{
            "content": {"parts": [{"text": choice["message"].get("content") or ""}], "role": "model"},
            "finishReason": GEMINI_FINISH.get(choice.get("finish_reason"), "STOP"),
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": name,
    }


@app.get("/v1beta/models")
async def gemini_models():
    return {"models": [{
        "name": f"models/{name}",
        "displayName": name,
        "description": f"local {MODELS[name].get('params', '')} {MODELS[name].get('quant', '')}".strip(),
        "inputTokenLimit": MODELS[name].get("ctx", 32768),
        "outputTokenLimit": 8192,
        "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
    } for name in MODELS]}


@app.post("/v1beta/models/{model_path:path}")
async def gemini_generate(model_path: str, request: Request):
    streaming = model_path.endswith(":streamGenerateContent")
    raw_name = model_path.rsplit(":", 1)[0].removeprefix("models/")
    name = resolve(raw_name)
    body = await request.json()
    payload = gemini_to_openai(body)
    payload["model"] = name
    backend = await ensure(name)
    if streaming:
        async def stream():
            upstream = dict(payload)
            upstream["stream"] = True
            client, response = await open_stream(backend, "/chat/completions", upstream)
            try:
                if response.status_code >= 400:
                    yield f"data: {json.dumps(error_payload('gemini', response.text))}\n\n".encode()
                    return
                async for chunk in iter_openai_chunks(response):
                    delta = chunk["choices"][0].get("delta", {})
                    if not delta.get("content"):
                        continue
                    payload_chunk = {
                        "candidates": [{
                            "content": {"parts": [{"text": delta["content"]}], "role": "model"},
                            "index": 0,
                        }]
                    }
                    yield f"data: {json.dumps(payload_chunk, ensure_ascii=False)}\n\n".encode()
                yield f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': ''}], 'role': 'model'}, 'finishReason': 'STOP', 'index': 0}]})}\n\n".encode()
            finally:
                await response.aclose()
                await client.aclose()
        return StreamingResponse(stream(), media_type="text/event-stream")
    response = await post_json(backend, "/chat/completions", payload)
    if response.status_code >= 400:
        return JSONResponse(error_payload("gemini", response.text, response.status_code), status_code=response.status_code)
    return openai_to_gemini(response.json(), name)


if __name__ == "__main__":
    if os.environ.get("LISTEN_FDS"):
        uvicorn.run(app, fd=3, log_level="warning")
    else:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
