#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Prefix prediction controls: fixed first 50 and 100 environment steps.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/value_calibration
status=results/value_calibration/prefix_model_status.csv
printf 'prefix_steps,seed,gpu,exit_code\n' > "$status"
jobs=()
for seed in 0 1 2 3 4; do
  for prefix in 50 100; do jobs+=("${prefix}:${seed}"); done
done
for start in 0 8; do
  pids=(); labels=()
  for gpu in {0..7}; do
    index=$((start + gpu))
    (( index >= ${#jobs[@]} )) && break
    IFS=: read -r prefix seed <<< "${jobs[$index]}"
    tag=EARLY${prefix}
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python trajectory_quality/quality_model.py \
      --seed "$seed" --prefix-steps "$prefix" --device cuda:0 \
      > "logs/value_calibration/train_${tag}_seed${seed}.log" 2>&1 &
    pids+=("$!"); labels+=("${prefix}:${seed}:${gpu}")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then code=0; else code=$?; fi
    IFS=: read -r prefix seed gpu <<< "${labels[$index]}"
    printf '%s,%s,%s,%s\n' "$prefix" "$seed" "$gpu" "$code" >> "$status"
  done
done
