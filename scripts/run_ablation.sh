#!/usr/bin/env bash
set -u

# Intentionally continue if one config fails so later ablations can still run.
# Credentials should be supplied by the shell environment, not committed here.

declare -A CKPT_DIR=(
  [d64_clean]=/tmp/checkpoints_d64
  [d128]=/tmp/checkpoints_d128
  [d256]=/tmp/checkpoints_d256
)

FINAL_STEP="${FINAL_STEP:-1000}"

for D in d64_clean d128 d256; do
  CKPT="${CKPT_DIR[$D]}/search_step_${FINAL_STEP}.pt"

  echo "=== TRAIN pilot_$D at $(date) ==="
  python -u train.py --config "pilot_$D"
  if [ $? -ne 0 ]; then
    echo "=== TRAIN FAILED pilot_$D - skipping compare ==="
    continue
  fi
  echo "=== TRAIN DONE pilot_$D at $(date) ==="

  if [ -f "$CKPT" ]; then
    echo "=== COMPARE pilot_$D $CKPT at $(date) ==="
    python -u compare_retrieval.py --ckpt "$CKPT" --num-batches 12
    echo "=== COMPARE DONE pilot_$D at $(date) ==="
  else
    echo "=== COMPARE SKIP pilot_$D - no ckpt at $CKPT ==="
  fi
done

echo "=== ABLATION COMPLETE at $(date) ==="
