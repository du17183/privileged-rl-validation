#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Exploratory, separately labeled follow-up to the original rollback rule.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/stable_privileged_rl/launch_followup_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code,start_utc,end_utc\n' > "$status"; fi
pids=()
starts=()
for seed in 0 1 2 3 4; do
  gpu=$((seed+2))
  run=E100R1_seed${seed}
  if [[ -s results/stable_privileged_rl/eval_${run}.csv ]] && \
     [[ $(tail -n 1 results/stable_privileged_rl/eval_${run}.csv | cut -d, -f3) == 500000 ]]; then
    printf 'E100R1,%s,%s,already_complete,,\n' "$seed" "$gpu" >> "$status"
    pids+=(0)
    starts+=("")
    continue
  fi
  starts+=("$(date -u +%Y-%m-%dT%H:%M:%SZ)")
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/stable_privileged_rl/train.py \
    --variant E100R1 --seed "$seed" --steps 500000 --num-envs 32 \
    --eval-every 10000 --eval-rounds 1 --bc-updates 3000 \
    --offline-critic-updates 1000 --updates-per-vector-step 4 \
    --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
    --aux-weight 0.1 --rollback-threshold 0.125 --rollback-patience 1 \
    --device cuda:0 > "logs/stable_privileged_rl/${run}.log" 2>&1 &
  pids+=("$!")
  echo "started $run gpu=$gpu pid=${pids[$seed]}" >&2
done
for seed in 0 1 2 3 4; do
  [[ ${pids[$seed]} == 0 ]] && continue
  if wait "${pids[$seed]}"; then code=0; else code=$?; fi
  printf 'E100R1,%s,%s,%s,%s,%s\n' "$seed" "$gpu" "$code" \
    "${starts[$seed]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
  echo "finished E100R1_seed${seed} exit=$code" >&2
done
