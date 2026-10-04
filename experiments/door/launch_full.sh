#!/usr/bin/env bash
# Twenty independent 500k-step runs in three waves across all eight B300 GPUs.
set -u
cd "$(dirname "$0")/../.."
source configs/runtime_env.sh
mkdir -p logs/door results/door checkpoints/door
status_file=results/door/launch_status.csv
printf 'variant,seed,gpu,exit_code\n' > "$status_file"

variants=(A B C D)
jobs=()
for seed in 0 1 2 3 4; do
  for variant in "${variants[@]}"; do
    jobs+=("${variant}:${seed}")
  done
done

for wave in 0 1 2; do
  pids=()
  labels=()
  for gpu in 0 1 2 3 4 5 6 7; do
    index=$((wave * 8 + gpu))
    if (( index >= ${#jobs[@]} )); then
      break
    fi
    IFS=: read -r variant seed <<< "${jobs[$index]}"
    log="logs/door/full_${variant}_seed${seed}.log"
    echo "START wave=$wave gpu=$gpu variant=$variant seed=$seed $(date -u +%FT%TZ)" | tee -a logs/door/launcher.log
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python experiments/door/train.py \
      --variant "$variant" --seed "$seed" --steps 500000 --num-envs 32 \
      --eval-every 25000 --eval-rounds 2 --bc-updates 3000 \
      --offline-critic-updates 1000 --updates-per-vector-step 4 \
      --batch-size 256 --offline-fraction 0.25 --actor-bc-weight 10 \
      --headless > "$log" 2>&1 &
    pids+=("$!")
    labels+=("$variant,$seed,$gpu")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    code=$?
    echo "${labels[$i]},$code" >> "$status_file"
    echo "END ${labels[$i]} exit=$code $(date -u +%FT%TZ)" | tee -a logs/door/launcher.log
  done
done
echo "ALL_FINISHED $(date -u +%FT%TZ)" | tee -a logs/door/launcher.log
