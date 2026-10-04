#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Corrected E1/E2 entropy-coefficient intervention; preserve exploratory runs.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
until [[ -f results/phase7_stable_policy/launch_status.csv ]] && \
      [[ $(wc -l < results/phase7_stable_policy/launch_status.csv) -ge 41 ]]; do
  sleep 30
done
status=results/phase7_stable_policy/low_alpha_launch_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,gpu,exit_code,start_utc,end_utc\n' > "$status"
fi
jobs=()
for variant in E1L E2L; do
  for seed in 0 1 2 3 4; do jobs+=("${variant}:${seed}"); done
done
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=() labels=() starts=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r variant seed <<< "${jobs[$index]}"
    run=P7${variant}_seed${seed}
    labels+=("$run")
    starts+=("$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/phase7_stable_policy/train_low_alpha.py \
      --variant "$variant" --seed "$seed" --steps 300000 --num-envs 32 \
      --eval-every 10000 --eval-rounds 2 --bc-updates 3000 \
      --updates-per-vector-step 4 --batch-size 256 --device cuda:0 \
      > "logs/phase7_stable_policy/${run}.log" 2>&1 &
    pids+=("$!")
    echo "started $run gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in "${!pids[@]}"; do
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=_ read -r variant seedword <<< "${labels[$gpu]}"
    seed=${seedword#seed}
    printf '%s,%s,%s,%s,%s,%s\n' "$variant" "$seed" "$gpu" "$code" \
      "${starts[$gpu]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
    echo "finished ${labels[$gpu]} exit=$code" >&2
  done
done
