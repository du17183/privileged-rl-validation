#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
source configs/runtime_env.sh
printf 'variant,seed,mode,gpu,exit_code\n' > results/door/heldout_status.csv
jobs=()
for seed in 0 1 2 3 4; do
  for variant in A B C D; do
    for mode in best final; do
      jobs+=("${variant}:${seed}:${mode}")
    done
  done
done
for wave in 0 1 2 3 4; do
  pids=()
  labels=()
  for gpu in 0 1 2 3 4 5 6 7; do
    index=$((wave * 8 + gpu))
    IFS=: read -r variant seed mode <<< "${jobs[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python experiments/door/heldout_eval.py \
      --variant "$variant" --seed "$seed" --mode "$mode" --headless \
      > "logs/door/heldout_${variant}_seed${seed}_${mode}.log" 2>&1 &
    pids+=("$!")
    labels+=("$variant,$seed,$mode,$gpu")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    code=$?
    printf '%s,%s\n' "${labels[$i]}" "$code" >> results/door/heldout_status.csv
  done
done
