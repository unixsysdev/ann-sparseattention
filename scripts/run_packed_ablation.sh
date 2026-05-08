#!/usr/bin/env bash
set -u

# Packed high-density d_search ablation. This intentionally uses sequence
# packing to reproduce the original pilot regime quickly; see config caveat.

declare -A CKPT_DIR=(
  [d64_packed]=/tmp/checkpoints_packed_d64
  [d128_packed]=/tmp/checkpoints_packed_d128
  [d256_packed]=/tmp/checkpoints_packed_d256
)

FINAL_STEP="${FINAL_STEP:-1000}"

for D in d64_packed d128_packed d256_packed; do
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

echo "=== PACKED ABLATION COMPLETE at $(date) ==="
