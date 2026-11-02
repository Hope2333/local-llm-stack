#!/usr/bin/env bash
# 安装脚本：把仓库里的脚本与 systemd 单元装到用户目录。
# 不下载模型，不自动启动服务。加 --enable 才会 enable/start。
set -euo pipefail

ENABLE=0
[ "${1:-}" = "--enable" ] || [ "${1:-}" = "-e" ] && ENABLE=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$HOME/.local/bin"
UNIT="$HOME/.config/systemd/user"
MODELS="$HOME/models"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }

say "仓库位置：$REPO"

# ---------- 依赖检查 ----------
FORCE=0
for c in llama-server python3 systemctl curl; do
  if ! command -v "$c" >/dev/null 2>&1; then
    warn "缺少命令：$c"
    FORCE=1
  fi
done
if [ "$FORCE" = 1 ]; then
  warn "请先安装缺失依赖。llama-server 需要带 Prism 补丁的构建（TQ/PTQ 量化支持）。"
  warn "官方 llama.cpp 可能无法加载 PTQ1_0/PQ2_0 权重。"
fi

# ---------- 目录 ----------
say "创建目录"
mkdir -p "$BIN" "$UNIT" "$MODELS" "$HOME/models/llm_gateway"

# ---------- 脚本 ----------
say "安装脚本到 $BIN"
install -m 755 "$REPO/llama/llama-bonsai-run"      "$BIN/llama-bonsai-run"
install -m 755 "$REPO/llama/bonsai-use"            "$BIN/bonsai-use"
install -m 755 "$REPO/ocr/ppocr"                   "$BIN/ppocr"
install -m 755 "$REPO/tools/strict_fetch.sh"       "$BIN/strict-fetch"
install -m 755 "$REPO/tools/bench.sh"              "$BIN/llm-bench"

# ---------- 网关 ----------
say "安装网关"
install -m 644 "$REPO/gateway/gateway.py" "$HOME/models/llm_gateway/gateway.py"
if [ ! -f "$HOME/models/llm_gateway/registry.json" ]; then
  install -m 644 "$REPO/gateway/registry.example.json" "$HOME/models/llm_gateway/registry.json"
  say "已从 registry.example.json 生成 registry.json —— 请检查里面的模型路径"
else
  say "registry.json 已存在，保留不动"
fi

# ---------- 配置模板 ----------
if [ ! -f "$HOME/.config/llama-bonsai.env" ] && [ -f "$REPO/llama/llama-bonsai.env.example" ]; then
  install -m 644 "$REPO/llama/llama-bonsai.env.example" "$HOME/.config/llama-bonsai.env"
  say "已生成 ~/.config/llama-bonsai.env"
fi
if [ ! -f "$HOME/models/oxr/oxr-config.yaml" ] && [ -f "$REPO/oxr/config.example.yaml" ]; then
  install -m 644 "$REPO/oxr/config.example.yaml" "$HOME/models/oxr/oxr-config.yaml"
  say "已生成 ~/models/oxr/oxr-config.yaml"
fi

# ---------- systemd 单元 ----------
say "安装 systemd 用户单元到 $UNIT"
for f in "$REPO"/gateway/systemd/*.service \
         "$REPO"/llama/systemd/*.service \
         "$REPO"/oxr/systemd/*.socket \
         "$REPO"/oxr/systemd/*.service; do
  [ -f "$f" ] || continue
  install -m 644 "$f" "$UNIT/$(basename "$f")"
done
systemctl --user daemon-reload

# ---------- 启用 ----------
if [ "$ENABLE" = 1 ]; then
  say "启用并启动服务"
  systemctl --user enable --now llm-gateway.service 2>/dev/null || warn "llm-gateway 启动失败，看 journalctl --user -u llm-gateway"
  for s in oxr-model.socket layout-dfine.socket; do
    systemctl --user enable --now "$s" 2>/dev/null || warn "$s 启用失败（OXR 未安装时可忽略）"
  done
  systemctl --user enable --now oxr-serve.service 2>/dev/null || warn "oxr-serve 启动失败（OXR 未安装时可忽略）"
  sleep 2
  systemctl --user --no-pager status llm-gateway.service | head -5 || true
fi

cat <<EOF

$(say "完成")

下一步：
  1. 把模型放到 ~/models/（或用 tools/ 里的脚本下载）
  2. 编辑 ~/models/llm_gateway/registry.json 里的路径与卸载参数
  3. 启动：  systemctl --user enable --now llm-gateway
  4. 自检：  curl -s localhost:8100/v1/models | jq -r '.data[].id'
  5. 状态：  curl -s localhost:8100/health | jq

如果要 OXR 文档解析，还需要：
  - Python 3.12 venv + torch(CUDA) + transformers
  - ~/models/oxr/ 下的权重（model.safetensors + layout_analysis/last.pth）
  - 详见 docs/architecture.md
EOF
