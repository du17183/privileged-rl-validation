#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Read-only actor execution sensitivity: same best checkpoint, no optimizer.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
jobs=()
for cap in 0.135 0.05 0.01; do
  for seed in 0 1 2 3 4; do jobs+=("${cap}:${seed}"); done
done
status=results/phase6_stable_online_rl/noise_sweep_status.csv
if [[ ! -e "$status" ]]; then
  printf 'cap,seed,gpu,exit_code\n' > "$status"
fi
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=() labels=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r cap seed <<< "${jobs[$index]}"
    labels+=("${jobs[$index]}")
    suffix=${cap/./p}
    output=results/phase6_stable_online_rl/heldout_P6B3_seed${seed}_best_stochastic_cap${suffix}.csv
    if [[ -s "$output" ]]; then pids+=(0); continue; fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/phase6_stable_online_rl/heldout_eval.py \
      --variant B3 --seed "$seed" --mode best --stochastic --action-std-cap "$cap" \
      --device cuda:0 > "logs/phase6_stable_online_rl/noise_cap${suffix}_seed${seed}.log" 2>&1 &
    pids+=("$!")
    echo "started noise cap=$cap seed=$seed gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in "${!pids[@]}"; do
    [[ ${pids[$gpu]} == 0 ]] && continue
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=: read -r cap seed <<< "${labels[$gpu]}"
    printf '%s,%s,%s,%s\n' "$cap" "$seed" "$gpu" "$code" >> "$status"
    echo "finished noise cap=$cap seed=$seed exit=$code" >&2
  done
done
