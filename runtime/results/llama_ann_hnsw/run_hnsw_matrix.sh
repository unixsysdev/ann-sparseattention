#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/marcel/SparseAttention
OUT=$ROOT/runtime/results/llama_ann_hnsw
PROMPT=$ROOT/runtime/prompts/long_prompt.txt
CPU_BIN=$ROOT/runtime/builds/llama-cpu/bin/llama-completion
HIP_BIN=$ROOT/runtime/builds/llama-hip/bin/llama-completion
MODELS=(
  "ann_6layer:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf"
  "ann_all32:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf"
  "ann_all36:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf"
)
mkdir -p "$OUT"
summary=$OUT/hnsw_summary.md
{
  echo "# llama.cpp Learned Sparse Attention HNSW Smoke Matrix"
  echo
  echo "Prompt: $PROMPT"
  echo
  echo "| backend | model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |"
  echo "|---|---:|---:|---:|---:|---:|---|"
} > "$summary"
run_one() {
  local backend=$1 name=$2 model=$3 bin=$4 extra=$5
  local stem="${backend}_${name}"
  echo "running $stem" >&2
  LLAMA_ANN_SEARCH=hnsw "$bin" -m "$model" -f "$PROMPT" -n 4 -c 1024 -t 8 $extra -fa on --no-warmup --no-display-prompt --no-conversation -s 123 --temp 0 > "$OUT/$stem.out" 2> "$OUT/$stem.err"
  local prompt_tps decode_tps eval_ms total_ms prefix
  prompt_tps=$(perl -ne 'if(/prompt eval time =.*\((?:[^,]*),\s*([0-9.]+) tokens per second\)/){print $1; exit}' "$OUT/$stem.err")
  decode_tps=$(perl -ne 'if(/eval time =.*\((?:\s*([0-9.]+) ms per token),\s*([0-9.]+) tokens per second\)/){print $2; exit}' "$OUT/$stem.err")
  eval_ms=$(perl -ne 'if(/eval time =.*\(\s*([0-9.]+) ms per token,/){print $1; exit}' "$OUT/$stem.err")
  total_ms=$(perl -ne 'if(/total time =\s*([0-9.]+) ms/){print $1; exit}' "$OUT/$stem.err")
  prefix=$(tr '\n' ' ' < "$OUT/$stem.out" | sed 's/|/ /g' | cut -c1-80)
  echo "| $backend | $name | ${prompt_tps:-NA} | ${decode_tps:-NA} | ${eval_ms:-NA} | ${total_ms:-NA} | \`$prefix\` |" >> "$summary"
}
for item in "${MODELS[@]}"; do
  name=${item%%:*}; model=${item#*:}
  run_one CPU "$name" "$model" "$CPU_BIN" ""
  run_one ROCm "$name" "$model" "$HIP_BIN" "-ngl 99"
done
