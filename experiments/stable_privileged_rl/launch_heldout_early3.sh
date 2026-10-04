#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Use GPUs 0–1 after the QV training wave exits; R1 still trains on 2–6.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/stable_privileged_rl/heldout_early3_status.csv
if [[ ! -e "$status" ]]; then printf 'variant,seed,gpu,exit_code\n' > "$status"; fi
jobs=()
for variant in Q0 QGT QV; do
  for seed in 0 1 2 3 4; do jobs+=("${variant}:${seed}"); done
done
for ((base=0; base<${#jobs[@]}; base+=2)); do
  pids=(); labels=()
  for gpu in 0 1; do
    index=$((base+gpu))
    if ((index >= ${#jobs[@]})); then break; fi
    IFS=: read -r variant seed <<< "${jobs[$index]}"
    run=${variant}_seed${seed}
    labels+=("$run")
    if [[ -s results/stable_privileged_rl/heldout_${run}_best.csv ]] && \
       [[ -s results/stable_privileged_rl/heldout_${run}_final.csv ]]; then
      pids+=(0); continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/stable_privileged_rl/heldout_eval.py \
      --variant "$variant" --seed "$seed" --device cuda:0 \
      > "logs/stable_privileged_rl/heldout_${run}.log" 2>&1 &
    pids+=("$!")
    echo "heldout early3 started $run gpu=$gpu" >&2
  done
  for gpu in "${!pids[@]}"; do
    [[ ${pids[$gpu]} == 0 ]] && continue
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=_ read -r variant seedword <<< "${labels[$gpu]}"
    printf '%s,%s,%s,%s\n' "$variant" "${seedword#seed}" "$gpu" "$code" >> "$status"
    echo "heldout early3 finished ${labels[$gpu]} exit=$code" >&2
  done
done
