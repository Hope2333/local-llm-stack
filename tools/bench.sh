#!/usr/bin/env bash
# 单配置基准：起一个临时 llama-server，发一次请求，读 timings 里的 pp/tg 与显存占用。
# 与 docs/benchmarks.md 的测量方法一致。
set -uo pipefail

MODEL=""; MMPROJ=""; CTX=65536; NGL=99; DEVICE="Vulkan1"; THREADS=12
OVERRIDE=""; CACHE=""; PORT=8491; PT=1500; GEN=128; LABEL=""
SERVER="${LLAMA_SERVER:-llama-server}"

usage() {
  cat <<'EOF'
用法: llm-bench --model PATH [选项]

  --model PATH        必填，GGUF 路径
  --mmproj PATH       视觉投影器（纯文本任务不要加）
  --ctx N             上下文长度（默认 65536）
  --ngl N             GPU 卸载层数（默认 99；0 = 纯 CPU）
  --device NAME       设备名（默认 Vulkan1；纯 CPU 用 none）
  --threads N         CPU 线程数（默认 12）
  --override SPEC     -ot 参数，例如 'blk\.(4[0-9]|5[0-9])\.ffn_.*=CPU'
  --cache TYPE        KV 类型，例如 q4_0 / q8_0
  --prompt-tokens N   合成 prompt 的目标 token 数（默认 1500）
  --gen N             生成 token 数（默认 128）
  --port N            临时服务端口（默认 8491）
  --label NAME        输出里的标签
EOF
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --mmproj) MMPROJ="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --ngl) NGL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --override) OVERRIDE="$2"; shift 2;;
    --cache) CACHE="$2"; shift 2;;
    --prompt-tokens) PT="$2"; shift 2;;
    --gen) GEN="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "未知参数: $1"; usage;;
  esac
done

[ -z "$MODEL" ] && usage
[ -f "$MODEL" ] || { echo "模型不存在: $MODEL"; exit 1; }
[ -z "$LABEL" ] && LABEL="$(basename "$MODEL" .gguf)"

LOG="/tmp/llm-bench-$(basename "$MODEL" .gguf).log"
ARGS=(--model "$MODEL" --ctx-size "$CTX" --threads "$THREADS"
      --flash-attn on --jinja --reasoning off --temp 0
      --host 127.0.0.1 --port "$PORT")
[ -n "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")
[ -n "$CACHE" ]  && ARGS+=(--cache-type-k "$CACHE" --cache-type-v "$CACHE")
[ -n "$OVERRIDE" ] && ARGS+=(-ot "$OVERRIDE")
if [ "$DEVICE" = "none" ] || [ "$NGL" = "0" ]; then
  ARGS+=(--n-gpu-layers 0 --device none)
else
  ARGS+=(--n-gpu-layers "$NGL" --device "$DEVICE")
fi

"$SERVER" "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

printf '%-24s ' "$LABEL"
for _ in $(seq 1 1200); do
  sleep 0.5
  kill -0 $PID 2>/dev/null || { echo "启动失败（见 $LOG）"; exit 1; }
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "n/a")

python3 - "$PORT" "$PT" "$GEN" <<'PY'
import json, sys, time, urllib.request
port, pt, gen = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# 中文语料，保证 tokenizer 按预期切分
sent = "三元量化模型把权重压缩到负一、零、正一三个取值，配合分组缩放因子来保持精度。"
body = sent * max(1, pt // 30)
req = {"messages": [{"role": "user", "content": body + "\n\n用一句话总结上面内容。"}],
       "max_tokens": gen, "temperature": 0}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=1800)
d = json.load(r)
t = d.get("timings") or {}
print("pp=%-7s tg=%-7s prompt_n=%-6s gen_n=%-4s 用时=%.1fs" % (
    round(t.get("prompt_per_second", 0), 1),
    round(t.get("predicted_per_second", 0), 2),
    t.get("prompt_n", 0), t.get("predicted_n", 0), time.time() - t0))
PY

echo "                                  显存=${VRAM} MiB"
