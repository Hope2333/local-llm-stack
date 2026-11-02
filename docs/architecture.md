# 架构与进程模型

这套栈的核心约束是 **8GB 显存**，所以架构上的一切设计都围绕两件事：
**显存独占** 和 **无请求就释放**。

## 进程与端口

| 组件 | 端口 | 启动方式 | 空闲行为 |
|---|---:|---|---|
| `llm-gateway` | 8100 | systemd 常驻 | 常驻（内存约 40 MB，不碰 GPU） |
| `llama-server`（按模型） | 8310-8399 | 网关按需 spawn | 300s 无请求后 terminate |
| `oxr-serve` | 8000 | systemd 常驻 | 常驻（编排层，不含 torch） |
| `layout-dfine` | 8091 | systemd socket 激活 | 120s 卸权重，300s 退出进程 |
| `oxr-model` | 8090 | systemd socket 激活 | 120s 卸权重，300s 退出进程 |

进程树：

```
systemd --user
├── llm-gateway.service          (常驻)
│   └── llama-server :83xx       (网关 spawn，PDEATHSIG 绑定)
├── oxr-serve.service :8000      (常驻)
├── layout-dfine.socket :8091 ──▶ layout-dfine.service   (激活后存在)
└── oxr-model.socket   :8090 ──▶ oxr-model.service       (激活后存在)
```

## 三级生命周期

每个 GPU 服务都走同一套状态机：

```
   ┌──────────┐  请求到达   ┌──────────┐  60s 空闲   ┌──────────┐  300s 空闲  ┌──────────┐
   │ 未启动    │───────────▶│ 权重在显存 │───────────▶│ 权重已卸载 │───────────▶│ 进程退出  │
   │ (无进程)  │            │ (热)      │            │ (冷)      │            │ (socket) │
   └──────────┘            └──────────┘            └──────────┘            └──────────┘
        ▲                                                            │
        └──────────────── 下一个请求触发 socket 激活 ◀────────────────┘
```

对应参数：

| 服务 | 卸载权重 | 退出进程 |
|---|---|---|
| oxr-model | `OXR_IDLE_UNLOAD=120` | `OXR_IDLE_EXIT=300` |
| layout-dfine | `LAYOUT_IDLE_UNLOAD=120` | `LAYOUT_IDLE_EXIT=300` |
| llama 后端 | — | `idle_seconds=300`（registry） |

卸载权重用 `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`。
进程退出用 `os._exit(0)`，因为正常退出时 CUDA context 有时不释放干净。

实测空闲 5.5 分钟后显存 **55 MiB**（只剩 kwin 的 7 MiB）。

## systemd socket 激活

进程退出后端口不能消失，否则下一次请求直接连接失败。用 socket 单元占住端口：

`oxr-model.socket`：

```ini
[Socket]
ListenStream=127.0.0.1:8090
Accept=no

[Install]
WantedBy=sockets.target
```

`Accept=no` 让 systemd 把监听 fd 直接传给服务进程（fd=3），而不是 fork 一个 per-connection 进程。
服务端要自己接管这个 fd：

```python
if os.environ.get("LISTEN_FDS"):
    uvicorn.run(app, fd=3)
```

uvicorn 0.54.0 起 `Config.fd` / `run.fd` 都支持。

效果：`systemctl is-active oxr-model` 显示 `inactive` 是 **正常状态**，只要 socket 是 `active`。

```bash
$ systemctl --user is-active oxr-model.socket oxr-model.service
active
inactive
```

## 网关的模型路由

`gateway/registry.json`：

```json
{
  "default": "bonsai-9b",
  "idle_seconds": 300,
  "port_pool": [8310, 8399],
  "models": {
    "bonsai-9b": {
      "backend": "llama",
      "kind": "text",
      "path": "~/models/bonsai2/9B/Ternary-Bonsai-2-9B-PQ2_0.gguf",
      "ctx": 262144,
      "device": "Vulkan1",
      "n_gpu_layers": 99,
      "threads": 12,
      "quant": "PQ2_0",
      "cache_type": "q4_0"
    },
    "oxr": {
      "backend": "static",
      "url": "http://127.0.0.1:8090/v1",
      "kind": "vision"
    }
  }
}
```

字段：

| 字段 | 说明 |
|---|---|
| `backend` | `llama`（网关自己 spawn）或 `static`（已在别处运行，只做转发） |
| `path` | GGUF 路径，支持 `~` |
| `mmproj` | 视觉投影器，可选。纯文本任务不要挂，省 848 MiB |
| `mtp` | MTP 草稿模型，可选。**只在 CPU 档开**，GPU 上负收益 |
| `spec_n_max` | MTP 草稿深度，默认 4 |
| `override` | `-ot` 参数，字符串或数组，用于把指定层下沉 CPU |
| `ctx` | 上下文长度 |
| `device` | `Vulkan1` / `none` / 空 |
| `n_gpu_layers` | 卸载层数，`99` = 全量 |
| `cache_type` | KV 缓存类型，长上下文用 `q4_0` |
| `threads` | CPU 线程数，Zen4 16 核用 12 |
| `kind` | `text` / `vision`，仅用于 `/v1/models` 标注 |

别名在 `gateway.py` 的 `ALIASES` 里硬编码：`bonsai`/`default` → 默认模型，`2b`/`4b`/`9b`/`27b` 短名。
未知模型名回落到 `default`，不报错（便于客户端直接写模型名试探）。

## 显存独占策略

8GB 装不下两个模型。网关在 spawn 新后端前会遍历所有后端：

```python
for other in BACKENDS.values():
    if other is not target and other.proc and other.proc.poll() is None:
        other.proc.terminate()   # 15s 后仍未退出则 kill
```

所以在 `bonsai-27b` 和 `oxr` 之间切换时，前一个会被主动驱逐，而不是等 OOM 报错。

## 子进程绑定

```python
subprocess.Popen(args, start_new_session=True, preexec_fn=_die_with_parent)
```

```python
def _die_with_parent():
    ctypes.CDLL("libc.so.6").prctl(1, 9)   # PR_SET_PDEATHSIG = SIGKILL
```

网关重启或被 kill 时，所有 llama-server 子进程立即消失，不会留下野进程占显存。
这是排查「显存被谁占了」时最常踩的坑：历史遗留的孤儿后端。

查占用：

```bash
ss -ltnpH 'sport = :8310' | grep -oP 'pid=\K[0-9]+'
```

不要用 `pgrep -f 'llama'`——它会匹配到自己的 shell。

## 请求路径

```
POST /v1/chat/completions  {"model":"bonsai-27b"}
        │
        ├─ 解析方言 → 统一内部 messages
        ├─ 查 registry → Backend("bonsai-27b")
        ├─ _ensure(): 端口已监听？→ 复用 : 驱逐其他后端并 spawn
        ├─ 等 /health（最多 900 次 × 0.5s）
        ├─ 转发到 127.0.0.1:8310/v1/chat/completions
        └─ 按客户端方言重新封装（流式则逐帧转换）
```

方言转换里最容易出错的三个地方（都已修）：

1. **Ollama 的 `stream` 默认是 true**，不是 false。`body.get("stream", True)`
2. **Anthropic 流式必须把 `stream: true` 透传上游**，否则上游返回整包 JSON，转换出的 `content_block_delta` 帧数为 0
3. **`count_tokens` 走根路径 `/tokenize`**，不是 `/v1/tokenize`。base URL 以 `/v1` 结尾时要剥掉再拼

## 失败模式与排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `load_model: failed to load model` | 显存被别的后端占着 | 重启网关，确认 `nvidia-smi` 回 55 MiB |
| `failed to allocate Vulkan1 buffer` | 参数组合超出 8GB | 降 ctx、加 `override`、或换小模型 |
| 解码突然掉到 6 tok/s | KV+compute buffer 撞墙（断崖） | 去掉 mmproj，或把 ctx 降一档 |
| 服务 `inactive` 但请求正常 | socket 激活的正常状态 | 看 `.socket` 而不是 `.service` |
| 端口被占 | 孤儿 llama-server | `ss -ltnp 'sport = :PORT'` 找 pid |
| 网关 500 `KeyError: 'mmproj'` | registry 缺字段 | 已修为条件添加 |

日志位置：

```bash
journalctl --user -u llm-gateway -f
tail -f /tmp/llama-backend-bonsai-9b.log
tail -f /tmp/oxr-model.log /tmp/layout-dfine.log
```
