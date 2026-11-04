#!/usr/bin/env python3
"""按显存档位生成 gateway registry 预设。

估算公式（系数由本机实测两点回归得出）：

    VRAM(MiB) = W + K * ctx / 1024 + C

    W      权重体积（MiB）
    K      每 token 的 KV 体积（KiB/token），q4_0 KV
    C      固定开销：计算缓冲 + 驱动上下文（MiB）
    挂 mmproj 时额外加 MM_COST（权重 + 其计算缓冲）

实测校准（8GB 4060 Laptop，Vulkan1，q4_0 KV，无 mmproj）：

| 模型 | ctx    | 实测 VRAM | W    | K     | C   |
|------|--------|----------:|-----:|------:|----:|
| 2B   |  65536 |   979 MiB |  483 |  4.38 | 216 |
| 2B   | 262144 |  1819 MiB |  483 |  4.38 | 216 |
| 4B   |  65536 |  2095 MiB | 1085 | 10.00 | 370 |
| 4B   | 262144 |  4015 MiB | 1085 | 10.00 | 370 |
| 9B   |  65536 |  3140 MiB | 2304 | 10.00 | 196 |
| 9B   | 262144 |  5060 MiB | 2304 | 10.00 | 196 |
| 27B  |  31744 |  6998 MiB | 5663 | 14.60 | 890 |
| 27B  |  73728 |  7601 MiB | 5663 | 14.60 | 890 |

正向验证：9B @262144 带 mmproj → 2304+960 + 10*256 + 196 = 6020 MiB，
网关实测 6018 MiB，误差 2 MiB。

断崖位置：27B 在 7601 MiB 可用、7720 MiB 解码 OOM。
故预算取标称显存的 92%，8GB 卡即 7533 MiB——刚好容纳 64K（7487）而排除 72K（7604）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MM_COST = 960         # MiB，mmproj 权重 600 + 计算缓冲 360（9B 实测差 958）
USABLE_RATIO = 0.92   # 标称显存可用比例（驱动 + 桌面 + 断崖余量）
Q8_RATIO = 1.89       # q8_0 / q4_0 的 KV 体积比（1.0625 / 0.5625）

FFN_SHARE = 0.77      # FFN 占 27B 权重的比例（实测张量统计）
FFN_LAYERS = 64

MODELS = [
    dict(
        name="bonsai-2b", params="1.88B", quant="PQ2_0", w=483, k=4.38, c=216,
        path="~/models/bonsai2/2B/Ternary-Bonsai-2-2B-PQ2_0.gguf",
        mmproj="~/models/bonsai2/2B/Ternary-Bonsai-2-2B-mmproj-Q8_0.gguf",
        mtp=None, max_ctx=262144, vision=True,
    ),
    dict(
        name="bonsai-4b", params="4.2B", quant="PQ2_0", w=1085, k=10.0, c=370,
        path="~/models/bonsai2/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf",
        mmproj="~/models/bonsai2/4B/Ternary-Bonsai-2-4B-mmproj-Q8_0.gguf",
        mtp="~/models/bonsai2/4B/Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf",
        max_ctx=262144, vision=True,
    ),
    dict(
        name="bonsai-9b", params="9B", quant="PQ2_0", w=2304, k=10.0, c=196,
        path="~/models/bonsai2/9B/Ternary-Bonsai-2-9B-PQ2_0.gguf",
        mmproj="~/models/bonsai2/9B/Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf",
        mtp="~/models/bonsai2/9B/Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf",
        max_ctx=262144, vision=True,
    ),
    # 27B 默认不挂 mmproj：多 960 MiB 会把全 GPU 的档位从 64K 压到 32K 以下
    dict(
        name="bonsai-27b", params="27B", quant="PTQ1_0", w=5663, k=14.6, c=890,
        path="~/models/bonsai2/27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        mmproj="~/models/bonsai2/27B/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        mtp=None, max_ctx=262144, vision=False,
    ),
    dict(
        name="bonsai-27b-q2", params="27B", quant="PQ2_0", w=6861, k=14.6, c=890,
        path="~/models/bonsai2/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf",
        mmproj="~/models/bonsai2/27B/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        mtp=None, max_ctx=262144, vision=False,
    ),
]

TIERS = [
    (4,  "4GB 老卡（GTX 1650、MX 系列）"),
    (6,  "6GB（GTX 1660、RTX 2060、RTX 3050 6G）"),
    (8,  "8GB（RTX 3060/4060 Laptop、RTX 3070 8G）"),
    (12, "12GB（RTX 3060 12G、RTX 4070、RTX 5070）"),
    (16, "16GB（RTX 4060 Ti 16G、RTX 4070 Ti Super、RTX 5070 Ti）"),
    (24, "24GB（RTX 3090/4090、RTX 5090）"),
    (32, "32GB 及以上（RTX 5090 32G、专业卡）"),
]

CTX_CANDIDATES = [262144, 196608, 131072, 98304, 73728, 65536, 49152, 32768, 16384, 8192]
OFFLOAD_CANDIDATES = [8, 12, 16, 20, 24, 28, 32, 36, 40, 48]


def need(model: dict, ctx: int, cache: str = "q4_0", vision: bool = False) -> int:
    ratio = Q8_RATIO if cache == "q8_0" else 1.0
    return int(model["w"] + (MM_COST if vision else 0)
               + model["k"] * ratio * ctx / 1024 + model["c"])


def plan_full_gpu(model: dict, budget: int) -> dict | None:
    for ctx in CTX_CANDIDATES:
        if ctx > model["max_ctx"]:
            continue
        for cache in ("q4_0", "q8_0"):
            n = need(model, ctx, cache, model["vision"])
            if n <= budget:
                return dict(mode="gpu", ctx=ctx, cache=cache, need=n)
    return None


def plan_offload(model: dict, budget: int) -> dict | None:
    """全量放不下时，用 -ot 把尾部 FFN 层下沉到 CPU，腾出显存给 KV。"""
    for ctx in (65536, 49152, 32768, 16384, 8192):
        if ctx > model["max_ctx"]:
            continue
        for n_layers in OFFLOAD_CANDIDATES:
            gpu_w = model["w"] * (1 - FFN_SHARE * n_layers / FFN_LAYERS)
            n = int(gpu_w + model["k"] * ctx / 1024 + model["c"])
            if n <= budget:
                return dict(mode="offload", ctx=ctx, cache="q4_0", need=n, layers=n_layers)
    return None


def build(tier_gib: int) -> dict:
    budget = int(tier_gib * 1024 * USABLE_RATIO)
    models: dict[str, dict] = {}

    for m in MODELS:
        if m["name"] == "bonsai-27b-q2" and tier_gib < 12:
            continue  # PQ2_0 比 PTQ1_0 多吃 1.2 GiB，小显存不值得

        plan = plan_full_gpu(m, budget) or plan_offload(m, budget)

        if plan is None:
            entry = {
                "backend": "llama", "kind": "text", "path": m["path"],
                "ctx": 32768, "device": None, "n_gpu_layers": 0,
                "threads": 12, "cache_type": "q4_0",
                "params": m["params"], "quant": m["quant"],
                "estimate": "显存放不下，回落纯 CPU",
            }
            if m["mtp"]:
                entry["mtp"] = m["mtp"]
                entry["spec_n_max"] = 4
                entry["estimate"] += "（可挂 MTP 提速约 1.5x）"
            models[m["name"]] = entry
            continue

        entry = {
            "backend": "llama",
            "kind": "vision" if m["vision"] else "text",
            "path": m["path"],
            "ctx": plan["ctx"],
            "device": "Vulkan1",
            "n_gpu_layers": 99,
            "threads": 12,
            "cache_type": plan["cache"],
            "params": m["params"],
            "quant": m["quant"],
            "estimate": f"约 {plan['need']} MiB",
        }
        if m["vision"]:
            entry["mmproj"] = m["mmproj"]
        if plan["mode"] == "offload":
            first = FFN_LAYERS - plan["layers"]
            layers = "|".join(str(i) for i in range(first, FFN_LAYERS))
            entry["override"] = f"blk\\.({layers})\\.ffn_.*=CPU"
            entry["estimate"] = f"FFN 下沉 {plan['layers']} 层（blk {first}-{FFN_LAYERS - 1}），约 {plan['need']} MiB"
        models[m["name"]] = entry

    models["oxr"] = {
        "backend": "static", "kind": "vision",
        "url": "http://127.0.0.1:8090/v1",
        "ctx": 16384,
        "estimate": "常驻约 3.1 GiB；与 27B 冲突时网关会自动驱逐",
    }

    return {
        "default": "bonsai-9b",
        "idle_seconds": 300 if tier_gib >= 12 else 180,
        "port_pool": [8310, 8399],
        "profile": {
            "vram_gib": tier_gib,
            "usable_mib": budget,
            "verified": tier_gib == 8,
            "generated_by": "tools/registry_tiers.py",
            "estimation": "VRAM = W + K*ctx/1024 + C，系数由实测两点回归",
        },
        "models": models,
    }


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "gateway" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tier_gib, _desc in TIERS:
        reg = build(tier_gib)
        (out_dir / f"registry.{tier_gib}gb.json").write_text(
            json.dumps(reg, ensure_ascii=False, indent=2) + "\n")

        cells = []
        for name in ("bonsai-2b", "bonsai-4b", "bonsai-9b", "bonsai-27b", "bonsai-27b-q2"):
            e = reg["models"].get(name)
            if e is None:
                cells.append("—")
            elif e["device"] is None:
                cells.append("CPU")
            elif e.get("override"):
                cells.append(f"{e['ctx'] // 1024}K↓")
            else:
                cells.append(f"{e['ctx'] // 1024}K")
        rows.append((tier_gib, reg["profile"]["usable_mib"], cells))

    default = build(8)
    default.pop("profile", None)
    for e in default["models"].values():
        e.pop("estimate", None)
    (out_dir / "registry.default.json").write_text(
        json.dumps(default, ensure_ascii=False, indent=2) + "\n")

    print(f"已生成 {len(TIERS) + 1} 个预设 → {out_dir}\n")
    print("| 显存 | 可用预算 | 2B | 4B | 9B | 27B (PTQ1_0) | 27B (PQ2_0) |")
    print("|---|---:|---|---|---|---|---|")
    for tier_gib, usable, cells in rows:
        print(f"| {tier_gib} GB | {usable} MiB | " + " | ".join(cells) + " |")
    print("\nK = 全量 GPU 的上下文；↓ = FFN 下沉到 CPU；CPU = 纯 CPU；— = 该档不用此量化")

    checks = [
        ("2B  @65536", need(MODELS[0], 65536), 979),
        ("2B  @262144", need(MODELS[0], 262144), 1819),
        ("4B  @65536", need(MODELS[1], 65536), 2095),
        ("4B  @262144", need(MODELS[1], 262144), 4015),
        ("9B  @65536", need(MODELS[2], 65536), 3140),
        ("9B  @262144", need(MODELS[2], 262144), 5060),
        ("9B  @262144 +mmproj", need(MODELS[2], 262144, vision=True), 6018),
        ("27B @31744", need(MODELS[3], 31744), 6998),
        ("27B @73728", need(MODELS[3], 73728), 7601),
    ]
    print("\n回归自检（预测 vs 实测）：")
    worst = 0
    for label, pred, actual in checks:
        err = abs(pred - actual)
        worst = max(worst, err)
        print(f"  {label:22s} 预测 {pred:5d}  实测 {actual:5d}  偏差 {err:3d} MiB")
    print(f"  最大偏差 {worst} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
