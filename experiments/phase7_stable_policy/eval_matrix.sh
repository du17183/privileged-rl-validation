#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Run only after phase7 launch.sh has finished; no training GPU is shared.
set -u
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
mkdir -p logs/phase7_stable_policy/robustness results/phase7_stable_policy/robustness
status=results/phase7_stable_policy/robustness_status.csv
if [[ ! -e "$status" ]]; then
  printf 'variant,seed,mode,noise,door_angle_deg,cabinet_dy,exit_code,gpu\n' > "$status"
fi
jobs=()
for arm in E0 E1L E2L E3 G1 E1 E2; do
  for seed in 0 1 2 3 4; do
    for noise in 0 0.01 0.05 0.2; do
      jobs+=("$arm:$seed:best:$noise:0:0")
    done
  done
done
for arm in E0 G1 E1; do
  for seed in 0 1 2 3 4; do
    for mode in best final; do
      for setting in '0:0' '2.5:0' '5:0' '0:0.01' '0:-0.01'; do
        IFS=: read -r angle dy <<< "$setting"
        [[ $mode == best && $angle == 0 && $dy == 0 ]] && continue
        jobs+=("$arm:$seed:$mode:0:$angle:$dy")
      done
    done
  done
done
for ((base=0; base<${#jobs[@]}; base+=8)); do
  pids=() descriptors=()
  for gpu in {0..7}; do
    index=$((base+gpu))
    if (( index >= ${#jobs[@]} )); then break; fi
    IFS=: read -r arm seed mode noise angle dy <<< "${jobs[$index]}"
    descriptor="${arm}_${seed}_${mode}_n${noise}_a${angle}_y${dy}"
    descriptors+=("${jobs[$index]}")
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python evaluation/noise_robustness.py \
      --variant "$arm" --seed "$seed" --mode "$mode" \
      --noise "$noise" --door-angle-deg "$angle" --cabinet-dy "$dy" \
      --rounds 2 --device cuda:0 \
      > "logs/phase7_stable_policy/robustness/${descriptor}.log" 2>&1 &
    pids+=("$!")
  done
  for gpu in "${!pids[@]}"; do
    if wait "${pids[$gpu]}"; then code=0; else code=$?; fi
    IFS=: read -r arm seed mode noise angle dy <<< "${descriptors[$gpu]}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$arm" "$seed" "$mode" "$noise" "$angle" "$dy" "$code" "$gpu" >> "$status"
    echo "evaluated ${descriptors[$gpu]} exit=$code" >&2
  done
done
