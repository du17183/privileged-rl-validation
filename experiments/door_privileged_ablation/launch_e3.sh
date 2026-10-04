#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Targeted follow-up: GT prediction in the original BC budget only.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/door_privileged_ablation/launch_e3_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
pids=()
for seed in 0 1 2 3 4; do
  gpu=$((seed+3))
  run=E3_seed${seed}
  if [[ -s results/door_privileged_ablation/eval_${run}.csv ]] && \
     [[ $(tail -n 1 results/door_privileged_ablation/eval_${run}.csv | cut -d, -f3) == 500000 ]]; then
    printf 'E3,%s,%s,already_complete\n' "$seed" "$gpu" >> "$status"
    pids+=(0)
    continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/door_privileged_ablation/train.py \
    --variant E3 --seed "$seed" --steps 500000 --num-envs 32 \
    --eval-every 25000 --eval-rounds 2 --bc-updates 3000 \
    --offline-critic-updates 1000 --updates-per-vector-step 4 \
    --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
    --aux-weight 0.1 --device cuda:0 \
    > logs/door_privileged_ablation/${run}.log 2>&1 &
  pids+=("$!")
  echo "started $run gpu=$gpu pid=${pids[$seed]}" >&2
done
for seed in 0 1 2 3 4; do
  if [[ ${pids[$seed]} == 0 ]]; then continue; fi
  if wait "${pids[$seed]}"; then code=0; else code=$?; fi
  printf 'E3,%s,%s,%s\n' "$seed" "$((seed+3))" "$code" >> "$status"
  echo "finished E3_seed${seed} exit=$code" >&2
done
