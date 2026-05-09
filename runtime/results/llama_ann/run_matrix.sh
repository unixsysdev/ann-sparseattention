#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/marcel/SparseAttention
PROMPT="$ROOT/runtime/prompts/long_prompt.txt"
CPU="$ROOT/runtime/builds/llama-cpu/bin/llama-completion"
GPU="$ROOT/runtime/builds/llama-hip/bin/llama-completion"
COMMON=(-f "$PROMPT" -n 16 -c 1024 -t 8 -fa on --no-warmup --no-display-prompt --no-conversation -s 123 --temp 0)
declare -A MODELS=(
  [base]="$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16.gguf"
  [ann_6layer]="$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf"
  [ann_all32]="$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf"
  [ann_all36]="$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf"
)
for backend in cpu gpu; do
  for name in base ann_6layer ann_all32 ann_all36; do
    out="$ROOT/runtime/results/llama_ann/${backend}_${name}.log"
    echo "[run] $backend $name -> $out"
    if [[ "$backend" == cpu ]]; then
      "$CPU" -m "${MODELS[$name]}" "${COMMON[@]}" > "$out" 2>&1
    else
      "$GPU" -m "${MODELS[$name]}" "${COMMON[@]}" -ngl 99 > "$out" 2>&1
    fi
  done
 done
