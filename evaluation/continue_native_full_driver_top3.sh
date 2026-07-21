#!/bin/zsh
set -euo pipefail

script_dir=${0:A:h}
repo_root=${script_dir:h}
output_rel="evaluation/runs/native-full-driver-top3-substantive-8192"
output_dir="$repo_root/$output_rel"
reference_root=${RELATIONAL_TRANSFORMER_ROOT:-/Users/henneberger/relational-transformer}
process_pattern="evaluation/train_native_full.py.*native-full-driver-top3-substantive-8192"

cd "$repo_root"

live_pid=$(pgrep -f "$process_pattern" | head -n 1 || true)
if [[ -n "$live_pid" ]]; then
  process_state=$(ps -o state= -p "$live_pid" | tr -d ' ')
  if [[ "$process_state" == T* ]]; then
    kill -CONT "$live_pid"
    echo "Resumed suspended native MPS fine-tuning process $live_pid."
  else
    echo "Native MPS fine-tuning is already running as process $live_pid."
  fi
  exit 0
fi

resume_args=()
if [[ -f "$output_dir/recovery.json" ]]; then
  resume_args=(--resume)
elif [[ -f "$output_dir/train.jsonl" ]]; then
  archive_dir="$output_dir.interrupted-$(date +%Y%m%d-%H%M%S)"
  mv "$output_dir" "$archive_dir"
  echo "No optimizer recovery checkpoint exists; the process ended before its first validation boundary."
  echo "Preserved the incomplete attempt at ${archive_dir#$repo_root/}."
  echo "Restarting from the last valid full-model checkpoint: stanford-star/rt-j/classification."
fi

export PYTHONPATH="$repo_root/python/src:$reference_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$repo_root/.venv-eval/bin/python" evaluation/train_native_full.py \
  --task rel-f1/driver-top3 \
  --checkpoint stanford-star/rt-j/classification \
  --output-dir "$output_rel" \
  --steps 43 \
  --ctx-size 8192 \
  --batch-size 2 \
  --effective-batch-size 32 \
  --eval-every 4 \
  --eval-batch-size 4 \
  --eval-items 10000000 \
  --patience 3 \
  --learning-rate 1e-5 \
  --weight-decay 0 \
  --grad-clip-norm 1 \
  --baseline-run evaluation/runs/native-full-driver-top3-pilot-lr1e-5-8192 \
  --adaptive-lr \
  --lr-backoff-factor 0.2 \
  --lr-backoff-patience 1 \
  --max-lr-backoffs 3 \
  --min-learning-rate 1e-7 \
  "${resume_args[@]}"
