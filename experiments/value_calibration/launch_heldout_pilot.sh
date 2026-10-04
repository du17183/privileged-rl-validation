#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Twenty independent best/final actor checks; no RL updates.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/value_calibration/online_pilot
status=results/value_calibration/online_pilot/heldout_status.csv
printf 'arm,seed,gpu,exit_code\n' > "$status"
jobs=()
for seed in 0 1 2 3 4; do
  for arm in uniform quality quality_return quality_offline; do jobs+=("${arm}:${seed}"); done
done
for start in 0 8 16; do
  pids=(); labels=()
  for gpu in {0..7}; do
    index=$((start + gpu))
    (( index >= ${#jobs[@]} )) && break
    IFS=: read -r arm seed <<< "${jobs[$index]}"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/value_calibration/heldout_pilot.py \
      --arm "$arm" --seed "$seed" --device cuda:0 \
      > "logs/value_calibration/online_pilot/heldout_${arm}_seed${seed}.log" 2>&1 &
    pids+=("$!"); labels+=("${arm}:${seed}:${gpu}")
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then code=0; else code=$?; fi
    IFS=: read -r arm seed gpu <<< "${labels[$index]}"
    printf '%s,%s,%s,%s\n' "$arm" "$seed" "$gpu" "$code" >> "$status"
    echo "heldout finished ${arm} seed${seed} code=${code}" >&2
  done
done
