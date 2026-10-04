#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Isolated Phase 4 matrix: 10 variants x 5 seeds x 500k interactions.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p results/stable_privileged_rl logs/stable_privileged_rl checkpoints/stable_privileged_rl
status=results/stable_privileged_rl/launch_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,gpu,exit_code,start_utc,end_utc\n' > "$status"
fi
variants=(E0 E25 E50 E100 E200 E100RB E100M Q0 QGT QV)
jobs=()
for variant in "${variants[@]}"; do
  for seed in 0 1 2 3 4; do jobs+=("${variant}:${seed}"); done
done
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=()
  labels=()
  starts=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r variant seed <<< "${jobs[$index]}"
    run=${variant}_seed${seed}
    labels+=("$run")
    if [[ -s results/stable_privileged_rl/eval_${run}.csv ]] && \
       [[ $(tail -n 1 results/stable_privileged_rl/eval_${run}.csv | cut -d, -f3) == 500000 ]]; then
      printf '%s,%s,%s,already_complete,,\n' "$variant" "$seed" "$gpu" >> "$status"
      pids+=(0)
      starts+=("")
      continue
    fi
    starts+=("$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/stable_privileged_rl/train.py \
      --variant "$variant" --seed "$seed" --steps 500000 --num-envs 32 \
      --eval-every 10000 --eval-rounds 1 --bc-updates 3000 \
      --offline-critic-updates 1000 --updates-per-vector-step 4 \
      --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
      --aux-weight 0.1 --rollback-threshold 0.25 --rollback-patience 2 \
      --device cuda:0 > "logs/stable_privileged_rl/${run}.log" 2>&1 &
    pids+=("$!")
    echo "started $run gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in "${!pids[@]}"; do
    [[ ${pids[$gpu]} == 0 ]] && continue
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=_ read -r variant seedword <<< "${labels[$gpu]}"
    seed=${seedword#seed}
    printf '%s,%s,%s,%s,%s,%s\n' "$variant" "$seed" "$gpu" "$code" \
      "${starts[$gpu]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
    echo "finished ${labels[$gpu]} exit=$code" >&2
  done
done
