#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trap 'printf "%s\n" "$?" > "$ROOT/logs/launch_all.exit_code"' EXIT
PYTHON_BIN="$ROOT/.venv/bin/python"
DEMO_FILE="$ROOT/datasets/drawer_expert.h5"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Isaac Lab Python environment: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$DEMO_FILE" ]]; then
  echo "Missing expert dataset: $DEMO_FILE" >&2
  exit 1
fi
source "$ROOT/configs/runtime_env.sh"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints" "$ROOT/results"
pids=()
for seed in $(seq 0 7); do
  (
    export CUDA_VISIBLE_DEVICES="$seed"
    for variant in A B C; do
      "$PYTHON_BIN" "$ROOT/train/run_sac.py" \
        --variant "$variant" --seed "$seed" --steps 200000 \
        --num-envs 32 --eval-every 10000 --eval-rounds 2 \
        --demo-file datasets/drawer_expert.h5 --headless
    done
  ) > "$ROOT/logs/seed_${seed}.stdout.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one seed failed; inspect logs/seed_*.stdout.log" >&2
  exit 1
fi
"$PYTHON_BIN" "$ROOT/evaluation/verify_runs.py"
"$PYTHON_BIN" "$ROOT/evaluation/aggregate.py" \
  "$ROOT/results" --output-dir "$ROOT/results"
"$PYTHON_BIN" "$ROOT/evaluation/report_rl.py"
