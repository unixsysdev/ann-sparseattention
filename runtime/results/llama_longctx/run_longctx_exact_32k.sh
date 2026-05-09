#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/marcel/SparseAttention
BIN=$ROOT/runtime/builds/llama-hip/bin/llama-completion
PROMPT=$ROOT/runtime/prompts/long_32k_prompt.txt
OUT=$ROOT/runtime/results/llama_longctx
mkdir -p "$OUT"
MODELS=(
  "base:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16.gguf"
  "ann_6layer_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf"
  "ann_all32_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf"
  "ann_all36_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf"
)
summary=$OUT/longctx_exact_32k_summary.md
{
  echo "# Long Context Exact Decode Test"
  echo
  echo "Prompt: $PROMPT"
  echo "Token count: 33601"
  echo
  echo "| model | prompt tok/s | decode tok/s | eval ms/token | total ms | output prefix |"
  echo "|---|---:|---:|---:|---:|---|"
} > "$summary"
for item in "${MODELS[@]}"; do
  name=${item%%:*}; model=${item#*:}
  echo "running $name" >&2
  unset LLAMA_ANN_SEARCH || true
  "$BIN" -m "$model" -f "$PROMPT" -n 16 -c 65536 -t 8 -ngl 99 -fa on --no-warmup --no-display-prompt --no-conversation -s 7 --temp 0 > "$OUT/$name.32k.out" 2> "$OUT/$name.32k.err"
  prompt_tps=$(perl -ne 'if(/prompt eval time =.*\((?:[^,]*),\s*([0-9.]+) tokens per second\)/){print $1; exit}' "$OUT/$name.32k.err")
  decode_tps=$(perl -ne 'if(/common_perf_print:\s+eval time =.*\(\s*[0-9.]+ ms per token,\s*([0-9.]+) tokens per second\)/){print $1; exit}' "$OUT/$name.32k.err")
  eval_ms=$(perl -ne 'if(/common_perf_print:\s+eval time =.*\(\s*([0-9.]+) ms per token,/){print $1; exit}' "$OUT/$name.32k.err")
  total_ms=$(perl -ne 'if(/total time =\s*([0-9.]+) ms/){print $1; exit}' "$OUT/$name.32k.err")
  prefix=$(tr '\n' ' ' < "$OUT/$name.32k.out" | sed 's/|/ /g' | cut -c1-120)
  echo "| $name | ${prompt_tps:-NA} | ${decode_tps:-NA} | ${eval_ms:-NA} | ${total_ms:-NA} | \`$prefix\` |" >> "$summary"
done
cat "$summary"
