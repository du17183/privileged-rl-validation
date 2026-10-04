#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/configs/runtime_env.sh"
trap 'printf "%s\n" "$?" > "$ROOT/logs/heldout.exit_code"' EXIT
pids=()
for seed in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES="$seed"
    for variant in A B C; do
      "$ROOT/.venv/bin/python" "$ROOT/evaluation/diagnose_checkpoint.py" \
        "$ROOT/checkpoints/${variant}_seed${seed}/step_200000.pt" \
        --num-envs 64 --output "$ROOT/results/heldout_${variant}_seed${seed}.json" \
        --device cuda:0 --headless
    done
  ) > "$ROOT/logs/heldout_seed${seed}.stdout.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one held-out replay failed" >&2
  exit 1
fi
"$ROOT/.venv/bin/python" "$ROOT/evaluation/aggregate_heldout.py"
