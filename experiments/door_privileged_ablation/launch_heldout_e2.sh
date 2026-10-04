#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/door_privileged_ablation/heldout_e2_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
pids=()
for seed in 0 1 2 3 4; do
  if [[ -s results/door_privileged_ablation/heldout_E2_seed${seed}_best.csv ]] && \
     [[ -s results/door_privileged_ablation/heldout_E2_seed${seed}_final.csv ]]; then
    pids+=(0)
    continue
  fi
  CUDA_VISIBLE_DEVICES=$seed .venv/bin/python experiments/door_privileged_ablation/heldout_eval.py \
    --variant E2 --seed "$seed" --device cuda:0 \
    > logs/door_privileged_ablation/heldout_E2_seed${seed}.log 2>&1 &
  pids+=("$!")
  echo "heldout started E2_seed${seed} gpu=$seed" >&2
done
for seed in 0 1 2 3 4; do
  if [[ ${pids[$seed]} == 0 ]]; then continue; fi
  if wait "${pids[$seed]}"; then code=0; else code=$?; fi
  printf 'E2,%s,%s,%s\n' "$seed" "$seed" "$code" >> "$status"
  echo "heldout finished E2_seed${seed} exit=$code" >&2
done
