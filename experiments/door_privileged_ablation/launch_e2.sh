#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Targeted follow-up: auxiliary GT prediction only during the first 100k steps.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/door_privileged_ablation/launch_e2_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
pids=()
for seed in 0 1 2 3 4; do
  run=E2_seed${seed}
  if [[ -s results/door_privileged_ablation/eval_${run}.csv ]] && \
     [[ $(tail -n 1 results/door_privileged_ablation/eval_${run}.csv | cut -d, -f3) == 500000 ]]; then
    printf 'E2,%s,%s,already_complete\n' "$seed" "$seed" >> "$status"
    pids+=(0)
    continue
  fi
  CUDA_VISIBLE_DEVICES=$seed .venv/bin/python experiments/door_privileged_ablation/train.py \
    --variant E2 --seed "$seed" --steps 500000 --num-envs 32 \
    --eval-every 25000 --eval-rounds 2 --bc-updates 3000 \
    --offline-critic-updates 1000 --updates-per-vector-step 4 \
    --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
    --aux-weight 0.1 --device cuda:0 \
    > logs/door_privileged_ablation/${run}.log 2>&1 &
  pids+=("$!")
  echo "started $run gpu=$seed pid=${pids[$seed]}" >&2
done
for seed in 0 1 2 3 4; do
  if [[ ${pids[$seed]} == 0 ]]; then continue; fi
  if wait "${pids[$seed]}"; then code=0; else code=$?; fi
  printf 'E2,%s,%s,%s\n' "$seed" "$seed" "$code" >> "$status"
  echo "finished E2_seed${seed} exit=$code" >&2
done
