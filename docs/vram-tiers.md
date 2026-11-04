# 按显存档位生成的默认配置

`tools/registry_tiers.py` 会为常见的显存档位各生成一份 `gateway/profiles/registry.<N>gb.json`。
`install.sh` 用 `nvidia-smi`/`rocm-smi` 读出显存总量，挑一个最接近且不超过的档位装成
`~/models/llm_gateway/registry.json`。

这些配置不是拍脑袋估的——系数来自本机实测回归，预测与实测最大偏差 7 MiB。

---

## 估算模型

```
VRAM(MiB) = W + K · ctx/1024 + C
```

| 符号 | 含义 | 来源 |
|---|---|---|
| `W` | 权重体积 | 文件大小（PQ2_0 / PTQ1_0） |
| `K` | 每 token 的 KV 体积（KiB/token，q4_0） | 实测两点差商 |
| `C` | 固定开销：计算缓冲 + 驱动上下文 | 实测两点截距 |

`q8_0` 的 K 乘以 1.89（1.0625 / 0.5625）。挂 mmproj 时整体加 `MM_COST = 960 MiB`
（600 MiB 权重 + 360 MiB 计算缓冲，由 9B 带/不带 mmproj 的实测差 958 MiB 得出）。

### 实测校准数据

8GB RTX 4060 Laptop，Vulkan1，`--n-gpu-layers 99`，KV 用 `q4_0`，无 mmproj：

| 模型 | ctx | 实测 VRAM | 解得 K | 解得 C |
|---|---:|---:|---:|---:|
| 2B | 65536 | 979 MiB | 4.38 | 216 |
| 2B | 262144 | 1819 MiB | 4.38 | 216 |
| 4B | 65536 | 2095 MiB | 10.00 | 370 |
| 4B | 262144 | 4015 MiB | 10.00 | 370 |
| 9B | 65536 | 3140 MiB | 10.00 | 196 |
| 9B | 262144 | 5060 MiB | 10.00 | 196 |
| 27B | 31744 | 6998 MiB | 14.60 | 890 |
| 27B | 73728 | 7601 MiB | 14.60 | 890 |

正向验证（没有参与拟合的点）：9B @262144 **带 mmproj**

```
2304 + 960 + 10.0·256 + 196 = 6020 MiB      实测 6018 MiB      偏差 2 MiB
```

回归自检由 `tools/registry_tiers.py` 在结尾自动打印：

```
2B  @65536             预测   979  实测   979  偏差   0 MiB
4B  @262144            预测  4015  实测  4015  偏差   0 MiB
9B  @262144 +mmproj    预测  6020  实测  6018  偏差   2 MiB
27B @73728             预测  7604  实测  7601  偏差   3 MiB
  最大偏差 7 MiB
```

---

## 为什么预算取 92%

显存断崖不是线性的。27B 在 7601 MiB 时解码正常，**7720 MiB 直接
`vk::Device::allocateMemory: ErrorOutOfDeviceMemory`**——权重能加载，解码时分配
compute buffer 才炸。

所以：

- 预算 = 标称显存 × 0.92（8GB 卡 → 7536 MiB）
- 8GB 档 27B 取 64K（预测 7487 MiB）刚好装下；72K（7604 MiB）被排除

这个余量同时覆盖驱动占用与桌面合成器（Wayland 下 kwin 常驻约 7 MiB）。

---

## 档位结果

| 显存 | 可用预算 | 2B | 4B | 9B | 27B (PTQ1_0) | 27B (PQ2_0) |
|---|---:|---|---|---|---|---|
| 4 GB | 3768 MiB | 256K | 128K | 16K | 32K↓ | — |
| 6 GB | 5652 MiB | 256K | 256K | 192K | 64K↓ | — |
| 8 GB | 7536 MiB | 256K | 256K | 256K | 64K | — |
| 12 GB | 11304 MiB | 256K | 256K | 256K | 256K | 192K |
| 16 GB | 15073 MiB | 256K | 256K | 256K | 256K | 256K |
| 24 GB | 22609 MiB | 256K | 256K | 256K | 256K | 256K |
| 32 GB | 30146 MiB | 256K | 256K | 256K | 256K | 256K |

`K` = 全量 GPU 的上下文，`↓` = FFN 下沉到 CPU，`—` = 该档不生成此量化。

几条规则：

- **小卡优先小模型**。6GB 上 9B 能开 192K，而 27B 要下沉 8 层 FFN 才够 64K——后者
  解码慢 3 倍以上，除非需要 27B 的推理质量。
- **27B 默认不挂 mmproj**。多 960 MiB 会把 8GB 上的全量 GPU 档位从 64K 压到 32K 以下
  （32768 附近有显存断崖）。要视觉能力就用更小的模型。
- **PQ2_0 只在 ≥12GB 出现**。它比 PTQ1_0 多吃 1.2 GiB，12GB 上换来的上下文反而更少
  （192K vs 256K）。
- **显存真不够时回落纯 CPU**，并在配置里保留 MTP 字段（CPU 上 MTP 是正收益：9B 12.4 → 23.0 tok/s）。

---

## 重新生成

```bash
# 改 MODELS 里的 W/K/C 或 TIERS 后重跑
python3 tools/registry_tiers.py

# 输出到 gateway/profiles/，并在终端打印档位表与回归自检
```

换卡或换量化后重新校准的办法：固定 `--device`/`--n-gpu-layers 99`/`--cache-type q4_0`，
在 65536 和 262144 两个上下文点各起一次服务，读 `nvidia-smi` 的 `memory.used`，
两点解出 K 与 C：

```
K = (VRAM_hi - VRAM_lo) / ((ctx_hi - ctx_lo)/1024)
C = VRAM_lo - W - K·ctx_lo/1024
```

注意两点：`VRAM_lo` 必须取**服务就绪并静置 2 秒后**的值；测量前确认显存是干净的
（网关会拉起后端，跑独占实验前先 `systemctl --user stop llm-gateway`）。

---

## 与手工调优的关系

本机在用的 `~/models/llm_gateway/registry.json` 与 8GB 档预设逐项对比：2B/4B/9B/27B
的 `ctx`、`cache_type`、`device`、`n_gpu_layers`、`mmproj`、`override` 全部一致。
差异只有两个**本机特有**的额外条目，不在通用预设里：

- `bonsai-27b-long`（128K，FFN 下沉 20 层）——需要超长上下文时手动切
- `qwen35-4b`（第三方 FP 模型，用于和三元模型做同尺寸对照）
