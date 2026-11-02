#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat.completions server for OXR-1.0.

Stands in for the vLLM plugin (which is GPU/CUDA-only and needs an out-of-tree
build) so `oxr serve` can talk to this box unchanged.

Endpoint: POST /v1/chat/completions
Image input: OpenAI style  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
             or          {"type":"image","image": <data url>}
"""
from __future__ import annotations

import base64
import gc
import io
import json
import os
import sys
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForCausalLM

torch.backends.cudnn.enabled = os.environ.get("OXR_CUDNN", "0") == "1"

MODEL_DIR = os.environ.get("OXR_MODEL_DIR", os.path.expanduser("~/models/oxr"))
HOST = os.environ.get("OXR_HOST", "127.0.0.1")
PORT = int(os.environ.get("OXR_PORT", "8090"))
DEVICE = os.environ.get("OXR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE_NAME = os.environ.get("OXR_DTYPE", "bf16" if DEVICE.startswith("cuda") else "fp32")
MAX_PIXELS = int(os.environ.get("OXR_MAX_PIXELS", str(1344 * 1792)))

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

print(f"[oxr] configured model={MODEL_DIR} device={DEVICE} dtype={DTYPE_NAME}", flush=True)
_dtype = _DTYPES[DTYPE_NAME]

# Upstream ships configuration_oxr.py / modeling_oxr.py via auto_map. transformers
# loads each auto_map file into its OWN dynamic module, so the Siglip2NavitConfig
# instance and the one registered with AutoModel end up as different classes and
# AutoModel.from_config() rejects them. Import both from one package namespace
# instead: single class identity, one registration, no dynamic modules.
sys.path.insert(0, MODEL_DIR)
from _oxrpkg.configuration_oxr import OXRConfig  # noqa: E402
from _oxrpkg.modeling_oxr import OXR  # noqa: E402

IDLE_UNLOAD = float(os.environ.get("OXR_IDLE_UNLOAD", "120"))
IDLE_EXIT = float(os.environ.get("OXR_IDLE_EXIT", "300"))

_state: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "last_used": time.monotonic(),
    "active": 0,
}
_state_lock = threading.Lock()


def _load_model():
    print(f"[oxr] loading model from {MODEL_DIR} device={DEVICE} dtype={DTYPE_NAME}", flush=True)
    config = OXRConfig.from_pretrained(MODEL_DIR, trust_remote_code=False)
    net = OXR.from_pretrained(
        MODEL_DIR,
        config=config,
        dtype=_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    net = net.to(DEVICE).eval()
    vocab = net.config.visual_vocab_size
    net.indicator_token_indices = torch.arange(
        vocab - 4, vocab, dtype=torch.long, device=DEVICE
    )
    tok = net.text_tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
        print(f"[oxr] gpu={torch.cuda.get_device_name(0)} "
              f"alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)
    return net, tok


def _acquire():
    with _state_lock:
        if _state["model"] is None:
            net, tok = _load_model()
            _state["model"], _state["tokenizer"] = net, tok
            print(f"[oxr] ready on {DEVICE}", flush=True)
        _state["active"] += 1
        _state["last_used"] = time.monotonic()
        return _state["model"], _state["tokenizer"]


def _release():
    with _state_lock:
        _state["active"] = max(0, _state["active"] - 1)
        _state["last_used"] = time.monotonic()


def _unload_model():
    with _state_lock:
        if _state["model"] is None or _state["active"] > 0:
            return
        _state["model"] = None
        _state["tokenizer"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[oxr] idle {IDLE_UNLOAD:.0f}s, GPU released", flush=True)


def _watch_idle():
    while True:
        time.sleep(5)
        idle = time.monotonic() - _state["last_used"]
        if _state["model"] is not None and _state["active"] == 0 and idle > IDLE_UNLOAD:
            _unload_model()
        elif (_state["model"] is None and _state["active"] == 0
              and IDLE_EXIT > 0 and idle > IDLE_EXIT):
            print(f"[oxr] idle {idle:.0f}s, exiting so the CUDA context is freed", flush=True)
            os._exit(0)


threading.Thread(target=_watch_idle, daemon=True, name="oxr-idle").start()


app = FastAPI()


def _data_url_to_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
    else:
        b64 = url
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _to_oxr_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        content = m.get("content")
        items: list[dict[str, Any]] = []
        if isinstance(content, str):
            items.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for it in content:
                t = it.get("type")
                if t in ("text", "input_text"):
                    items.append({"type": "text", "text": it.get("text", "")})
                elif t in ("image_url", "input_image"):
                    iu = it.get("image_url") or it.get("image") or {}
                    url = iu.get("url", "") if isinstance(iu, dict) else str(iu)
                    items.append({"type": "image", "image": _data_url_to_image(url)})
                elif t == "image":
                    items.append({"type": "image", "image": _data_url_to_image(it.get("image", ""))})
        out.append({"role": "user", "content": items})
    return out


class ChatRequest(BaseModel):
    model: str = "oxr"
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_DIR}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "oxr", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
@torch.inference_mode()
def chat_completions(req: ChatRequest):
    model, tokenizer = _acquire()
    try:
        return _complete(req, model, tokenizer)
    finally:
        _release()


@torch.inference_mode()
def _complete(req: ChatRequest, model, tokenizer):
    t0 = time.time()
    oxr_messages = _to_oxr_messages(req.messages)
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        oxr_messages, max_pixels=MAX_PIXELS
    )
    input_ids = input_ids.to(DEVICE)
    if pixel_values is not None:
        pixel_values = pixel_values.to(DEVICE, dtype=_dtype)
        grid_thws = grid_thws.to(DEVICE)

    max_new = req.max_tokens or 2048
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": False}
    if req.temperature and req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature)
        if req.top_p:
            gen_kwargs["top_p"] = req.top_p
        if req.top_k:
            gen_kwargs["top_k"] = req.top_k

    out = model.generate(
        inputs=input_ids,
        pixel_values=pixel_values,
        grid_thws=grid_thws,
        **gen_kwargs,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    dt = time.time() - t0
    print(f"[oxr] {len(input_ids[0])} prompt tok -> {len(out[0])} out tok in {dt:.1f}s", flush=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(out.shape[1]),
            "total_tokens": int(input_ids.shape[1] + out.shape[1]),
        },
    }


if __name__ == "__main__":
    if os.environ.get("LISTEN_FDS"):
        uvicorn.run(app, fd=3, log_level="warning")
    else:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
