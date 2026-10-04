#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Continue independent checkpoint replays on GPU 7 while final runs train.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/stable_privileged_rl/heldout_early2_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
for variant in E50 E100 E200 E100RB; do
  for seed in 0 1 2 3 4; do
    run=${variant}_seed${seed}
    if [[ -s results/stable_privileged_rl/heldout_${run}_best.csv ]] && \
       [[ -s results/stable_privileged_rl/heldout_${run}_final.csv ]]; then
      continue
    fi
    if CUDA_VISIBLE_DEVICES=7 .venv/bin/python experiments/stable_privileged_rl/heldout_eval.py \
         --variant "$variant" --seed "$seed" --device cuda:0 \
         > "logs/stable_privileged_rl/heldout_${run}.log" 2>&1; then code=0; else code=$?; fi
    printf '%s,%s,7,%s\n' "$variant" "$seed" "$code" >> "$status"
    echo "heldout early2 finished $run exit=$code" >&2
  done
done
