#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Independent fixed-reset best/final evaluation, followed by BC step-0 and
# selected stochastic exploration checks. Run after launch.sh finishes.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/phase6_stable_online_rl results/phase6_stable_online_rl
jobs=()
for variant in A0 A1 A2 B1 B2 B3; do
  for seed in 0 1 2 3 4; do
    for mode in best final; do jobs+=("${variant}:${seed}:${mode}:det"); done
  done
done
for seed in 0 1 2 3 4; do
  jobs+=("B1:${seed}:initial:det")
  jobs+=("B1:${seed}:best:stoch")
  jobs+=("B3:${seed}:best:stoch")
done
status=results/phase6_stable_online_rl/heldout_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,mode,stochastic,gpu,exit_code\n' > "$status"
fi
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=() labels=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r variant seed mode stochastic <<< "${jobs[$index]}"
    labels+=("${jobs[$index]}")
    suffix=""
    extra=()
    if [[ "$stochastic" == stoch ]]; then suffix="_stochastic"; extra=(--stochastic); fi
    output=results/phase6_stable_online_rl/heldout_P6${variant}_seed${seed}_${mode}${suffix}.csv
    if [[ -s "$output" ]]; then pids+=(0); continue; fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/phase6_stable_online_rl/heldout_eval.py \
      --variant "$variant" --seed "$seed" --mode "$mode" "${extra[@]}" \
      --device cuda:0 > "logs/phase6_stable_online_rl/heldout_P6${variant}_seed${seed}_${mode}${suffix}.log" 2>&1 &
    pids+=("$!")
    echo "started heldout ${jobs[$index]} gpu=$gpu pid=${pids[$gpu]}" >&2
  done
  for gpu in "${!pids[@]}"; do
    [[ ${pids[$gpu]} == 0 ]] && continue
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=: read -r variant seed mode stochastic <<< "${labels[$gpu]}"
    printf '%s,%s,%s,%s,%s,%s\n' "$variant" "$seed" "$mode" "$stochastic" "$gpu" "$code" >> "$status"
    echo "finished heldout ${labels[$gpu]} exit=$code" >&2
  done
done
