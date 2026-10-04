#!/usr/bin/env bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Sequentially verify the 40 training runs before occupying GPUs with eval.
set -euo pipefail
cd "$PROJECT_ROOT"
source configs/runtime_env.sh
status=results/phase7_stable_policy/launch_status.csv
until [[ -f "$status" ]] && [[ $(wc -l < "$status") -ge 41 ]]; do
  sleep 30
done
low_status=results/phase7_stable_policy/low_alpha_launch_status.csv
until [[ -f "$low_status" ]] && [[ $(wc -l < "$low_status") -ge 11 ]]; do
  sleep 30
done
until [[ -f results/phase7_stable_policy/duplicate_launch_audit/repair.json ]]; do
  sleep 30
done
if awk -F, 'NR>1 && $4 != 0 && $4 != "already_complete" {bad=1} END{exit !bad}' "$status"; then
  echo 'Training run failed; refusing to summarize incomplete data.' >&2
  exit 1
fi
if awk -F, 'NR>1 && $4 != 0 {bad=1} END{exit !bad}' "$low_status"; then
  echo 'Corrected alpha training run failed.' >&2
  exit 1
fi
.venv/bin/python experiments/phase7_stable_policy/analyze.py \
  > logs/phase7_stable_policy/summary.log 2>&1
.venv/bin/python experiments/phase7_stable_policy/diagnose.py \
  > logs/phase7_stable_policy/diagnostic_summary.log 2>&1
.venv/bin/python experiments/phase7_stable_policy/plot.py \
  > logs/phase7_stable_policy/plot.log 2>&1
bash experiments/phase7_stable_policy/eval_matrix.sh \
  > logs/phase7_stable_policy/robustness_launcher.log 2>&1
if awk -F, 'NR>1 && $7 != 0 {bad=1} END{exit !bad}' \
     results/phase7_stable_policy/robustness_status.csv; then
  echo 'Independent robustness evaluation failed.' >&2
  exit 1
fi
if [[ $(find results/phase7_stable_policy/robustness -maxdepth 1 -name '*.csv' | wc -l) -ne 275 ]]; then
  echo 'Independent robustness matrix has missing or extra CSV results.' >&2
  exit 1
fi
.venv/bin/python experiments/phase7_stable_policy/analyze_robustness.py \
  > logs/phase7_stable_policy/robustness_summary.log 2>&1
.venv/bin/python experiments/phase7_stable_policy/plot_robustness.py \
  > logs/phase7_stable_policy/robustness_plot.log 2>&1
echo 'Phase 7 training, evaluation and summary completed.'
