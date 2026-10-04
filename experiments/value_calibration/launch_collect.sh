#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 5 source-policy seeds x 8 frozen policies x 125 episodes = 5,000 trajectories.
# No optimizer or replay update occurs in collect_frozen.py.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p results/value_calibration/frozen_rollouts logs/value_calibration
status=results/value_calibration/collection_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,mode,seed,gpu,exit_code\n' > "$status"; fi
policies=(E0:best E0:final E25:final E100:best E100:final E100R1:final Q0:best Q0:final)
for seed in 0 1 2 3 4; do
  pids=(); labels=()
  for gpu in {0..7}; do
    IFS=: read -r variant mode <<< "${policies[$gpu]}"
    run=${variant}_${mode}_seed${seed}
    labels+=("${variant}:${mode}:${seed}")
    if [[ -s results/value_calibration/frozen_rollouts/${run}.h5 ]]; then
      pids+=(0)
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/value_calibration/collect_frozen.py \
      --variant "$variant" --mode "$mode" --seed "$seed" --episodes 125 --device cuda:0 \
      > "logs/value_calibration/${run}.log" 2>&1 &
    pids+=("$!")
    echo "collection started $run gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in {0..7}; do
    [[ ${pids[$gpu]} == 0 ]] && continue
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=: read -r variant mode source_seed <<< "${labels[$gpu]}"
    printf '%s,%s,%s,%s,%s\n' "$variant" "$mode" "$source_seed" "$gpu" "$code" >> "$status"
    echo "collection finished ${variant}_${mode}_seed${source_seed} exit=$code" >&2
  done
done
