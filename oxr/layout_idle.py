#!/usr/bin/env python3
"""D-FINE layout service that gives its GPU memory back when no request arrives."""

import os
import queue
import threading
import time

import torch
import uvicorn

from dfine_la.api import create_app
from dfine_la.cuda_graph import CUDAGraphModel
from dfine_la.model import load_model
from dfine_la.runtime import ModelRuntime


class IdleModelRuntime(ModelRuntime):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._idle_seconds = float(os.getenv("LAYOUT_IDLE_UNLOAD", "120"))
        self._idle_exit = float(os.getenv("LAYOUT_IDLE_EXIT", "300"))
        self._lock = threading.Lock()
        self._last_used = time.monotonic()
        self._loaded = True
        threading.Thread(target=self._watch, daemon=True, name="layout-idle").start()

    def _build(self):
        model = load_model(
            self.settings.model_path,
            self.device,
            num_classes=self.settings.num_classes,
            image_size=self.settings.image_size,
        )
        if self.cuda_graph_enabled:
            model = CUDAGraphModel(model)
            model.warmup(self.settings.image_size)
        return model

    def _release(self) -> None:
        with self._lock:
            if not self._loaded:
                return
            try:
                model = self._engines.get_nowait()
            except queue.Empty:
                return
            del model
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"[layout] idle {self._idle_seconds:.0f}s, GPU released", flush=True)

    def infer(self, ready):
        with self._lock:
            if not self._loaded:
                self._engines.put(self._build())
                self._loaded = True
                print("[layout] model reloaded", flush=True)
        outputs = super().infer(ready)
        self._last_used = time.monotonic()
        return outputs

    def _watch(self) -> None:
        while True:
            time.sleep(5)
            idle = time.monotonic() - self._last_used
            if self._loaded and idle > self._idle_seconds:
                self._release()
            elif not self._loaded and self._idle_exit > 0 and idle > self._idle_exit:
                print(f"[layout] idle {idle:.0f}s, exiting so the CUDA context is freed",
                      flush=True)
                os._exit(0)


app = create_app(runtime_factory=IdleModelRuntime)


if __name__ == "__main__":
    if os.getenv("LISTEN_FDS"):
        uvicorn.run(app, fd=3, log_level=os.getenv("LOG_LEVEL", "info"))
    else:
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8091")),
            workers=1,
            log_level=os.getenv("LOG_LEVEL", "info"),
        )
