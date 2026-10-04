#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Six unique Phase 6 recipes x five paired seeds; B0 is the same run as A2.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p results/phase6_stable_online_rl logs/phase6_stable_online_rl checkpoints/phase6_stable_online_rl
status=results/phase6_stable_online_rl/launch_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,gpu,exit_code,start_utc,end_utc\n' > "$status"
fi
variants=(A0 A1 A2 B1 B2 B3)
jobs=()
for variant in "${variants[@]}"; do
  for seed in 0 1 2 3 4; do jobs+=("${variant}:${seed}"); done
done
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=() labels=() starts=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r variant seed <<< "${jobs[$index]}"
    run=P6${variant}_seed${seed}
    labels+=("$run")
    if [[ -s results/phase6_stable_online_rl/eval_${run}.csv ]] && \
       [[ $(tail -n 1 results/phase6_stable_online_rl/eval_${run}.csv | cut -d, -f3) -ge 500000 ]]; then
      printf '%s,%s,%s,already_complete,,\n' "$variant" "$seed" "$gpu" >> "$status"
      pids+=(0); starts+=("")
      continue
    fi
    starts+=("$(date -u +%Y-%m-%dT%H:%M:%SZ)")
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/phase6_stable_online_rl/train.py \
      --variant "$variant" --seed "$seed" --steps 500000 --num-envs 32 \
      --eval-every 10000 --eval-rounds 2 --bc-updates 3000 \
      --updates-per-vector-step 4 --batch-size 256 --device cuda:0 \
      > "logs/phase6_stable_online_rl/${run}.log" 2>&1 &
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
