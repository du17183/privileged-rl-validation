#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Independent offline calibration only. Never steps the environment or updates a policy.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/value_calibration results/value_calibration
status=results/value_calibration/offline_model_status.csv
printf 'method,seed,gpu,exit_code\n' > "$status"
jobs=()
for seed in 0 1 2 3 4; do
  for method in MC TD0 TDMC QUALITY; do jobs+=("${method}:${seed}"); done
done
for start in 0 8 16; do
  pids=()
  labels=()
  for gpu in {0..7}; do
    index=$((start + gpu))
    (( index >= ${#jobs[@]} )) && break
    IFS=: read -r method seed <<< "${jobs[$index]}"
    labels+=("${method}:${seed}:${gpu}")
    if [[ "$method" == QUALITY ]]; then
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python trajectory_quality/quality_model.py \
        --seed "$seed" --device cuda:0 \
        > "logs/value_calibration/train_${method}_seed${seed}.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python value_analysis/value_calibration.py \
        --method "$method" --seed "$seed" --device cuda:0 \
        > "logs/value_calibration/train_${method}_seed${seed}.log" 2>&1 &
    fi
    pids+=("$!")
    echo "offline model started ${method} seed${seed} gpu${gpu} pid=${pids[-1]}" >&2
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then code=0; else code=$?; fi
    IFS=: read -r method seed gpu <<< "${labels[$index]}"
    printf '%s,%s,%s,%s\n' "$method" "$seed" "$gpu" "$code" >> "$status"
    echo "offline model finished ${method} seed${seed} code=${code}" >&2
  done
done
