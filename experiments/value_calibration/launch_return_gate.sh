#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Exploratory follow-up after success-gated replay could not activate.
# Reuses the completed, matching uniform runs; only quality_return is new.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/value_calibration/online_pilot
status=results/value_calibration/online_pilot/return_gate_status.csv
printf 'arm,seed,gpu,exit_code\n' > "$status"
pids=()
for seed in 0 1 2 3 4; do
  test -s "results/value_calibration/online_pilot/eval_uniform_seed${seed}.csv" || exit 2
  if [[ -e "results/value_calibration/online_pilot/eval_quality_return_seed${seed}.csv" ]]; then
    echo "Refusing to overwrite quality_return seed${seed}" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES=$seed .venv/bin/python experiments/value_calibration/online_pilot.py \
    --arm quality_return --seed "$seed" --device cuda:0 \
    > "logs/value_calibration/online_pilot/quality_return_seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "return-gated pilot started seed${seed} gpu${seed} pid=${pids[-1]}" >&2
done
for seed in 0 1 2 3 4; do
  if wait "${pids[$seed]}"; then code=0; else code=$?; fi
  printf 'quality_return,%s,%s,%s\n' "$seed" "$seed" "$code" >> "$status"
  echo "return-gated pilot finished seed${seed} code=${code}" >&2
done
