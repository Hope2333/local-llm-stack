# local-llm-stack

在 **8GB 显存笔记本** 上把「本地大模型 + 文档解析 VLM + OCR」跑成一套可用的服务栈：
一个多方言 LLM 网关、一条文档解析流水线，以及针对显存约束的卸载调优方法。

三件事彼此独立，可只用其中一块：

| 类别 | 目录 | 作用 |
|---|---|---|
| API 网关 | [`gateway/`](gateway/) | 一个端口说 OpenAI / Anthropic / Ollama / Gemini 四种方言，按模型名路由 |
| 本地推理运行时 | [`llama/`](llama/) | llama.cpp 服务封装、模型切换、卸载参数 |
| 文档解析 VLM | [`oxr/`](oxr/) | OXR-1.0 三层流水线（版面检测 → 模型服务 → 聚合 API） |
| OCR | [`ocr/`](ocr/) | PP-OCRv4 命令行封装 |
| 工具 | [`tools/`](tools/) | 分块严格下载、模型清单生成、基准测试 |

文档：

- [架构与进程模型](docs/architecture.md)
- [基准数据](docs/benchmarks.md)
- [8GB 显存的卸载调优手册](docs/offload-tuning.md)
- [按显存档位生成的默认配置](docs/vram-tiers.md)

---

## 核心结论（先看这个）

在 8GB 显存上跑长上下文，直觉是错的：

> **不要把大模型卸载，要把小模型全部塞进显存。**

同一台机器、同样的 262144 上下文：

| 方案 | 解码速度 | 显存 |
|---|---:|---:|
| 27B 部分卸载到 128K | 12.7 tok/s | 7549 MiB |
| 27B 全量 GPU 到 64K | 25.7 tok/s | 7837 MiB |
| **9B 全量 GPU 到 262K** | **48.3 tok/s** | 6018 MiB |
| **4B 全量 GPU 到 262K** | **78.6 tok/s** | 4649 MiB |

27B 在 128K 上做 FFN 下沉，比 9B 全量上卡慢 **3.8 倍**。需要 27B 的推理质量时才付这个代价。

另外两个反直觉结论：

- **`-nkvo`（KV 放内存）是最差的选择**：128K 下只有 6.38 tok/s，因为每生成一个 token 都要把整份 KV 过一遍 PCIe。正确做法是用 `-ot` 把 FFN 下沉到 CPU，让 KV 一步都不离开显存。
- **MTP 投机解码在 GPU 上是负收益**：4B 带 MTP 54.4 tok/s → 去掉 78.6；9B 带 MTP 39.5 → 去掉 48.3。草稿模型要多吃 2.68 GiB 显存和一份带宽。它只在 CPU 推理时有明显收益（9B：12.4 → 23.0 tok/s）。

---

## 按显存选默认配置

`gateway/profiles/` 下有 4/6/8/12/16/24/32GB 七个档位的 registry 预设，`install.sh` 会用
`nvidia-smi` 检测显存自动挑一个。表里的数字来自实测回归：

```
VRAM(MiB) = W + K·ctx/1024 + C        # W=权重, K=每 token KV, C=固定开销
```

| 显存 | 可用预算 | 2B | 4B | 9B | 27B (PTQ1_0) | 27B (PQ2_0) |
|---|---:|---|---|---|---|---|
| 4 GB | 3768 MiB | 256K | 128K | 16K | 16K↓ | — |
| 6 GB | 5652 MiB | 256K | 256K | 192K | 64K↓ | — |
| **8 GB** | **7536 MiB** | 256K | 256K | **256K** | **64K** | — |
| 12 GB | 11304 MiB | 256K | 256K | 256K | 256K | 192K |
| 16 GB | 15073 MiB | 256K | 256K | 256K | 256K | 256K |
| 24 GB | 22609 MiB | 256K | 256K | 256K | 256K | 256K |
| 32 GB | 30146 MiB | 256K | 256K | 256K | 256K | 256K |

`K` = 全量 GPU 的上下文；`↓` = FFN 下沉到 CPU；`—` = 该档不用此量化。

模型系数（实测两点回归，最大偏差 7 MiB）：

| 模型 | W (MiB) | K (KiB/token) | C (MiB) |
|---|---:|---:|---:|
| 2B | 483 | 4.38 | 216 |
| 4B | 1085 | 10.0 | 370 |
| 9B | 2304 | 10.0 | 196 |
| 27B | 5663 | 14.6 | 890 |

挂了 mmproj 再加 960 MiB。想按自己的卡重新生成：`python3 tools/registry_tiers.py`。
细节见 [docs/vram-tiers.md](docs/vram-tiers.md)。

---

## 架构

```
                    ┌──────────────────────────────┐
   OpenAI SDK ─────▶│                              │
   Anthropic SDK ──▶│   llm-gateway  :8100         │──▶ llama-server (按需拉起)
   Ollama CLI ─────▶│   多方言 + 模型路由          │      :8310-8399
   Gemini SDK ─────▶│   空闲 300s 回收后端          │
                    └──────────────────────────────┘

                    ┌──────────────────────────────┐
   POST /v1/parse ──▶│   oxr-serve  :8000           │
                    └───────────┬──────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌────────────────────┐            ┌────────────────────┐
   │ layout-dfine :8091 │            │ oxr-model   :8090  │
   │ D-FINE 版面检测     │            │ OXR-1.0 文档解析    │
   └────────────────────┘            └────────────────────┘

   后两者用 systemd socket 激活：空闲 120s 卸载权重，300s 退出进程，显存归零。
```

设计上的三条硬规则：

1. **显存独占**：网关启动新后端前会先驱逐其它正在跑的后端，避免 8GB 被两个模型同时抢爆。
2. **无请求就释放**：所有 GPU 服务都是懒加载 + 空闲回收，静默时显存回到 ~55 MiB。
3. **端口即接口**：进程可以随时消失，端口由 systemd socket 保证可连接，请求进来再拉起。

## 硬件基线

所有基准数据来自这台机器，换机器请重测：

- CPU：AMD Ryzen 9 7945HX（16C/32T Zen4，AVX-512），双通道 DDR5-5200，约 83 GB/s
- GPU：NVIDIA RTX 4060 Laptop（AD107，8GB GDDR6 128-bit，约 272 GB/s）
- iGPU：Radeon 610M（gfx1036，2 CU）——实测是三个后端里最慢的，别用
- 系统：Arch Linux，内核走 Vulkan RADV 调 GPU

## 快速开始

```bash
git clone <this-repo> ~/Projects/local-llm-stack
cd ~/Projects/local-llm-stack
./install.sh                 # 建目录、装脚本、装 systemd 单元（不启动）
```

模型与权重需要自己下（体积太大不进仓库），默认路径 `~/models/`：

```
~/models/bonsai2/{2B,4B,9B,27B}/    # Bonsai-2 三元模型 GGUF
~/models/oxr/                       # OXR-1.0 权重 + 版面检测 last.pth
```

下载用 `tools/` 里的脚本，它强制每段请求返回 HTTP 206 且按官方 sha256 校验：

```bash
python3 tools/bonsai_manifest.py                     # 生成清单
bash tools/bonsai_batch.sh                           # 逐条严格下载
bash tools/strict_fetch.sh <url> <out> <size> <sha>  # 单文件
```

启动：

```bash
systemctl --user enable --now llm-gateway
curl -s localhost:8100/v1/models | jq -r '.data[].id'
```

## 网关用法

四种方言都指向同一批模型，`model` 字段决定路由：

```bash
# OpenAI
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-9b","messages":[{"role":"user","content":"你好"}]}'

# Anthropic
curl -s localhost:8100/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-9b","max_tokens":64,"messages":[{"role":"user","content":"你好"}]}'

# Ollama
curl -s localhost:8100/api/chat -d '{"model":"bonsai-9b","messages":[{"role":"user","content":"你好"}]}'

# Gemini
curl -s localhost:8100/v1beta/models/bonsai-9b:generateContent -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"你好"}]}]}'
```

支持的端点：

| 方言 | 端点 |
|---|---|
| OpenAI | `POST /v1/chat/completions`、`POST /v1/completions`、`POST /v1/responses`、`GET /v1/models` |
| Anthropic | `POST /v1/messages`、`POST /v1/messages/count_tokens` |
| Ollama | `POST /api/chat`、`POST /api/generate`、`GET /api/tags`、`POST /api/show`、`GET /api/ps` |
| Gemini | `POST /v1beta/models/{model}:generateContent`、`:streamGenerateContent`、`GET /v1beta/models` |
| 运维 | `GET /`、`GET /health` |

流式都支持（OpenAI SSE、Anthropic 事件链、Ollama NDJSON、Gemini SSE）。

模型注册表在 `gateway/registry.json`，从 `registry.example.json` 复制后改路径：

```bash
cp gateway/registry.example.json gateway/registry.json
```

字段含义见 [docs/architecture.md#registry](docs/architecture.md)。

## 文档解析

```bash
curl -s -X POST localhost:8000/v1/parse/sync \
  -F file=@page.png -F api_key=any -F output_formats=markdown
```

返回 Markdown + 版面坐标，表格转 HTML、公式转 LaTeX。冷启动约 40s（要读 3GB 权重），
之后同一进程内连续请求是热的。真实扫描页实测（1232×1744 中文老扫描件）：4 分钟出 204 字 Markdown，内容完全正确。

## OCR

```bash
ppocr 图片.png          # 文本 + 置信度
ppocr 图片.png --json   # 带坐标
```

走 RapidOCR 的 PP-OCRv4 mobile 模型。注意 RapidOCR 3.x 默认是 PP-OCRv6，必须显式传
`OCRVersion.PPOCRV4` 才是 v4，且 `ocr_version` 必须是枚举值不是字符串。

## 已知限制

- 8GB 显存下 OXR（约 3GB）与 27B（约 7.6GB）不能同时驻留，网关会互斥
- OXR 对合成分辨率图（非真实扫描）质量会退化，属训练分布外
- PP-OCRv4 在低对比度老扫描件上会漏检，需先做二值化
- 27B 无 MTP 权重，长上下文档位只能靠 `-ot` 换空间

## License

本项目代码 MIT。依赖与权重各自遵循其许可：OXR-1.0 使用 OXR Open Source License
（Apache 2.0 + 禁止提供竞争性托管 API），Bonsai-2 为 Apache 2.0，llama.cpp 为 MIT。
