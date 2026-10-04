#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Five paired waves, one run per B300. Existing Phase 2 paths are never touched.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p results/door_privileged_ablation logs/door_privileged_ablation checkpoints/door_privileged_ablation
status=results/door_privileged_ablation/launch_tail_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,gpu,exit_code,start_utc,end_utc\n' > "$status"
fi
variants=(diag_A diag_B B1 B2 B5 B6 E0 E1)
for seed in 3 4; do
  pids=()
  starts=()
  for gpu in "${!variants[@]}"; do
    variant=${variants[$gpu]}
    run=${variant}_seed${seed}
    logfile=logs/door_privileged_ablation/${run}.log
    if [[ -s results/door_privileged_ablation/eval_${run}.csv ]] && \
       [[ $(tail -n 1 results/door_privileged_ablation/eval_${run}.csv | cut -d, -f3) == 500000 ]]; then
      printf '%s,%s,%s,already_complete,,\n' "$variant" "$seed" "$gpu" >> "$status"
      pids+=(0)
      starts+=("")
      continue
    fi
    starts+=("$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/door_privileged_ablation/train.py \
      --variant "$variant" --seed "$seed" --steps 500000 --num-envs 32 \
      --eval-every 25000 --eval-rounds 2 --bc-updates 3000 \
      --offline-critic-updates 1000 --updates-per-vector-step 4 \
      --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
      --aux-weight 0.1 --distill-updates 1000 --device cuda:0 \
      > "$logfile" 2>&1 &
    pids+=("$!")
    echo "started $run gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in "${!variants[@]}"; do
    if [[ ${pids[$gpu]} == 0 ]]; then continue; fi
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    printf '%s,%s,%s,%s,%s,%s\n' "${variants[$gpu]}" "$seed" "$gpu" "$code" \
      "${starts[$gpu]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
    echo "finished ${variants[$gpu]}_seed${seed} exit=$code" >&2
  done
done
