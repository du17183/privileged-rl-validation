#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/door_privileged_ablation/heldout_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
variants=(diag_A diag_B B1 B2 B5 B6 E0 E1)
for seed in 0 1 2 3 4; do
  pids=()
  for gpu in "${!variants[@]}"; do
    variant=${variants[$gpu]}
    if [[ -s results/door_privileged_ablation/heldout_${variant}_seed${seed}_final.csv ]] && \
       [[ -s results/door_privileged_ablation/heldout_${variant}_seed${seed}_best.csv ]]; then
      pids+=(0)
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/door_privileged_ablation/heldout_eval.py \
      --variant "$variant" --seed "$seed" --device cuda:0 \
      > logs/door_privileged_ablation/heldout_${variant}_seed${seed}.log 2>&1 &
    pids+=("$!")
    echo "heldout started ${variant}_seed${seed} gpu=$gpu" >&2
  done
  for gpu in "${!variants[@]}"; do
    if [[ ${pids[$gpu]} == 0 ]]; then continue; fi
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    printf '%s,%s,%s,%s\n' "${variants[$gpu]}" "$seed" "$gpu" "$code" >> "$status"
    echo "heldout finished ${variants[$gpu]}_seed${seed} exit=$code" >&2
  done
done
