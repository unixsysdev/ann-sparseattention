#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/marcel/SparseAttention
BIN=$ROOT/runtime/builds/llama-hip/bin/llama-completion
PROMPT=$ROOT/runtime/prompts/sample_quality_prompt.txt
OUT=$ROOT/runtime/results/llama_samples
mkdir -p "$OUT"
MODELS=(
  "base:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16.gguf:"
  "ann_6layer_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf:"
  "ann_all32_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all32-k128-v2.gguf:"
  "ann_all36_exact:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-all36-k128-v2.gguf:"
  "ann_6layer_hnsw:$ROOT/runtime/models/Qwen3-4B-Instruct-2507-F16-ann-6layer-k128-v2.gguf:hnsw"
)
summary=$OUT/sample_outputs.md
{
  echo "# Sample Outputs"
  echo
  echo "Prompt:"
  echo '```'
  cat "$PROMPT"
  echo '```'
} > "$summary"
for item in "${MODELS[@]}"; do
  IFS=: read -r name model search <<< "$item"
  echo "running $name" >&2
  if [[ "$search" == hnsw ]]; then export LLAMA_ANN_SEARCH=hnsw; else unset LLAMA_ANN_SEARCH || true; fi
  "$BIN" -m "$model" -f "$PROMPT" -n 96 -c 2048 -t 8 -ngl 99 -fa on --no-warmup --no-display-prompt --no-conversation -s 42 --temp 0 > "$OUT/$name.out" 2> "$OUT/$name.err"
  {
    echo
    echo "## $name"
    echo
    echo '```text'
    cat "$OUT/$name.out"
    echo
    echo '```'
    echo
    grep -E 'prompt eval time|eval time|total time' "$OUT/$name.err" | sed 's/^/- /'
  } >> "$summary"
done
