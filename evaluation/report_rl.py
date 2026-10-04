"""Write the A/B/C experiment report from complete measured CSV results."""

import csv
import json
import statistics
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    lines = [
        "# Panda Drawer BC → online SAC experiment",
        "",
        "Eight paired seeds (0–7) run on eight B300 GPUs. Each A/B/C run uses the same successful planner demonstrations, BC budget, reward, 32 parallel environments, 200,000 online training interactions, 10,000-step evaluation schedule, and 64 episodes per checkpoint. Evaluation interactions are logged separately from training steps. A has robot-only actor/critic; B has robot-only actor and robot+fixture critic; C has robot+fixture actor/critic and is an upper bound rather than the deployment policy.",
        "",
        f"Shared pretraining simulator interactions: {summary.get('shared_conditioning_interactions', 0):,}. These are added to the total-interaction threshold column but are not multiplied by the number of seeds.",
        "A/B actor parameters and BC-only evaluation match exactly within every paired seed; `evaluation/verify_runs.py` checks this before reporting results.",
        "",
        "| Variant | Final drawer success mean ± SD | Final contact-assisted success mean | Peak success mean | Success AUC mean ± SD | Wall time mean |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    final_success = {}
    for variant in "ABC":
        successes, grasped_successes, peaks, times, aucs = [], [], [], [], []
        for seed in range(8):
            with (root / "results" / f"eval_{variant}_seed{seed}.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            successes.append(float(rows[-1]["success_rate"]))
            grasped_successes.append(float(rows[-1].get("contact_assisted_success_rate") or 0.0))
            times.append(float(rows[-1]["wall_time_s"]) / 3600)
            aucs.append(summary["variants"][variant]["seeds"][str(seed)]["success_auc"])
            peaks.append(summary["variants"][variant]["seeds"][str(seed)]["peak_success_rate"])
        final_success[variant] = successes
        lines.append(
            f"| {variant} | {statistics.mean(successes):.3f} ± {statistics.stdev(successes):.3f} | "
            f"{statistics.mean(grasped_successes):.3f} | "
            f"{statistics.mean(peaks):.3f} | "
            f"{statistics.mean(aucs):.3f} ± {statistics.stdev(aucs):.3f} | {statistics.mean(times):.2f} h |"
        )
    lines.extend(["", "## Sample-efficiency thresholds", "", "Values below are median online training steps across seeds that reached the threshold; `not reached` means no seed reached it. Individual crossing points, total interactions including shared pretraining and evaluation, and wall times are in `results/thresholds.csv`.", "", "| Variant | 50% | 80% | 90% |", "|:--|--:|--:|--:|"])
    for variant in "ABC":
        cells = []
        for threshold in (0.5, 0.8, 0.9):
            values = [summary["variants"][variant]["seeds"][str(seed)]["threshold_steps"][str(threshold)] for seed in range(8)]
            reached = [value for value in values if value is not None]
            cells.append(f"{statistics.median(reached):.0f} ({len(reached)}/8 seeds)" if reached else "not reached")
        lines.append(f"| {variant} | {' | '.join(cells)} |")
    lines.extend(["", "## Thresholds retained through the final evaluation", "",
                  "The first listed checkpoint and every later checkpoint meet the threshold; at least two checkpoints are required.",
                  "", "| Variant | 50% | 80% | 90% |", "|:--|--:|--:|--:|"])
    for variant in "ABC":
        cells = []
        for threshold in (0.5, 0.8, 0.9):
            values = [summary["variants"][variant]["seeds"][str(seed)]["retained_threshold_steps"][str(threshold)] for seed in range(8)]
            reached = [value for value in values if value is not None]
            cells.append(f"{statistics.median(reached):.0f} ({len(reached)}/8 seeds)" if reached else "not reached")
        lines.append(f"| {variant} | {' | '.join(cells)} |")
    comparison = summary["paired_comparisons"].get("B_minus_A")
    lines.extend(["", "## Paired A/B comparison", ""])
    if comparison:
        lines.append(
            f"B minus A mean normalized success-AUC gain: {comparison['mean_success_auc_gain']:.4f}; "
            f"paired bootstrap 95% interval [{comparison['paired_bootstrap_95pct_ci'][0]:.4f}, "
            f"{comparison['paired_bootstrap_95pct_ci'][1]:.4f}]; "
            f"exact two-sided paired sign-flip p = {comparison['exact_two_sided_sign_flip_p']:.4f} "
            f"across {comparison['n']} matched seeds."
        )
    else:
        lines.append("Paired A/B AUC could not be computed because completed matched checkpoints were absent.")
    heldout_path = root / "results/heldout_final_summary.json"
    if heldout_path.exists():
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        lines.extend([
            "", "## Independent final-checkpoint replay", "",
            f"Each of 24 final checkpoints ran {heldout['episodes_per_final_checkpoint']} episodes with reset seed {heldout['evaluation_seed']}, adding {heldout['additional_evaluation_interactions']:,} evaluation interactions outside the periodic curves.",
            "", "| Variant | Held-out final success | Contact-assisted success |",
            "|:--|--:|--:|",
        ])
        for variant in "ABC":
            data = heldout["variants"][variant]
            lines.append(f"| {variant} | {data['success_mean']:.3f} | {data['contact_assisted_success_mean']:.3f} |")
        paired = heldout["paired_B_minus_A"]
        lines.extend(["", f"B minus A held-out final-success mean gain: {paired['mean_final_success_gain']:.4f}; exact paired sign-flip p = {paired['exact_two_sided_sign_flip_p']:.4f}."])
    best_final = max(max(values) for values in final_success.values())
    any_crossing = any(
        seed_data["threshold_steps"]["0.5"] is not None
        for variant_data in summary["variants"].values()
        for seed_data in variant_data["seeds"].values()
    )
    if not any_crossing:
        conclusion = "No seed achieved 50% success at any checkpoint. This protocol does not establish whether fixture ground truth improves online RL sample efficiency."
    elif comparison and comparison["mean_success_auc_gain"] > 0 and comparison["exact_two_sided_sign_flip_p"] <= 0.05 and best_final >= 0.5:
        conclusion = "The privileged critic improved measured success AUC under this simulation protocol, with a positive paired effect and exact sign-flip p ≤ 0.05. Check contact-assisted success and per-seed thresholds before translating this finding to a physical fixture."
    elif best_final < 0.5:
        conclusion = "Some checkpoints crossed 50% success, but no run retained it at the final checkpoint. The observed improvement is transient and does not establish a stable sample-efficiency benefit."
    else:
        conclusion = "The completed runs do not provide statistically supported evidence that the privileged critic improves success AUC under this protocol. Inspect per-seed threshold crossings and learning curves before interpreting the sign of the effect."
    lines.extend(["", "## Conclusion", "", conclusion, "", "Figures: `results/success.png` and `results/reward.png`. Per-run CSVs, aggregate curves, thresholds, TensorBoard events, final checkpoints, and the exact experiment configuration are retained in the project.", ""])
    report = "\n".join(lines)
    for path in (root / "docs/rl_experiment_report.md", root / "results/rl_experiment_report.md"):
        path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
