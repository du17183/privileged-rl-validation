#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Gate-controlled 2 arms x 5 paired seeds x 100k interactions, one job/GPU.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/value_calibration/online_pilot results/value_calibration/online_pilot
test -s results/value_calibration/online_pilot/offline_gate_passed.json || exit 2
status=results/value_calibration/online_pilot/status.csv
printf 'arm,seed,gpu,exit_code\n' > "$status"
jobs=()
for seed in 0 1 2 3 4; do
  jobs+=("uniform:${seed}" "quality:${seed}")
done
for start in 0 8; do
  pids=(); labels=()
  for gpu in {0..7}; do
    index=$((start + gpu))
    (( index >= ${#jobs[@]} )) && break
    IFS=: read -r arm seed <<< "${jobs[$index]}"
    output="results/value_calibration/online_pilot/eval_${arm}_seed${seed}.csv"
    if [[ -e "$output" ]]; then
      echo "Refusing to overwrite $output" >&2
      exit 3
    fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/value_calibration/online_pilot.py \
      --arm "$arm" --seed "$seed" --device cuda:0 \
      > "logs/value_calibration/online_pilot/${arm}_seed${seed}.log" 2>&1 &
    pids+=("$!"); labels+=("${arm}:${seed}:${gpu}")
    echo "online pilot started ${arm} seed${seed} gpu${gpu} pid=${pids[-1]}" >&2
  done
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then code=0; else code=$?; fi
    IFS=: read -r arm seed gpu <<< "${labels[$index]}"
    printf '%s,%s,%s,%s\n' "$arm" "$seed" "$gpu" "$code" >> "$status"
    echo "online pilot finished ${arm} seed${seed} code=${code}" >&2
  done
done
