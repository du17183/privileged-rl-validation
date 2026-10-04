"""Audit early recovery anchors and late raw policies without further training."""
import json
from pathlib import Path
import torch
from experiments.phase8_progress_rl.analyze import read, summarize
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def validation(folder, filename):
    return torch.load(folder/filename, map_location="cpu", weights_only=False)["validation"]


def main():
    records = []
    for arm in ("A", "B", "C", "D"):
        for seed in range(5):
            run = f"P8{arm}_seed{seed}"
            folder = ROOT/"checkpoints"/"phase8_progress_rl"/run
            rows = read(OUT/f"eval_{run}.csv")
            diagnostics = read(OUT/f"diagnostics_{run}.csv")
            first = next((row for row in rows if row["guard_event"] == "rollback"), None)
            initial = validation(folder, "step_0.pt")
            raw_final = validation(folder, "raw_step_300000.pt")
            first_raw = validation(folder, f"raw_step_{first['env_steps']}.pt") if first else None
            # These are observations of the implemented guard, not counterfactual runs.
            first_reasons = []
            if first_raw:
                previous = rows[rows.index(first)-1]
                anchor = validation(folder, f"step_{previous['selected_best_step']}.pt")
                for key, threshold, direction in (("success", .25, -1), ("progress", .15, -1),
                                                   ("max_angle", .15, -1), ("regression_amount", .10, 1)):
                    if direction*(first_raw[key]-anchor[key]) > threshold:
                        first_reasons.append(key)
            records.append(dict(arm=arm, seed=seed, initial=initial, raw_final=raw_final,
                first_rollback_step=int(first["env_steps"]) if first else None,
                first_rollback_raw=first_raw, first_rollback_reasons=first_reasons,
                initially_failed=initial["success"] < .5,
                selected_best_step=int(rows[-1]["selected_best_step"]),
                diagnostics_first={k: float(v) for k, v in diagnostics[0].items()},
                diagnostics_last={k: float(v) for k, v in diagnostics[-1].items()}))
    groups = {}
    for arm in ("A", "B", "C", "D"):
        selected = [r for r in records if r["arm"] == arm]
        groups[arm] = dict(raw_final_metrics={key: summarize([r["raw_final"][key] for r in selected])
            for key in ("success", "max_angle", "final_angle", "progress", "regression_amount", "regression_event")},
            late_diagnostics={key: summarize([r["diagnostics_last"][key] for r in selected])
            for key in ("q_mean", "q_variance", "action_drift_mse", "expert_action_mse", "policy_entropy", "policy_std_mean", "alpha")},
            initial_successes=[r["initial"]["success"] for r in selected],
            retained_initial_anchor_seeds=[r["seed"] for r in selected if r["selected_best_step"] == 0])
    result = dict(per_seed=records, groups=groups,
        implementation_guard=dict(success_drop=.25, progress_drop=.15, maximum_angle_drop_rad=.15,
                                  regression_increase_rad=.10),
        limitation="Guard protects measured anchors, including poor initial anchors; no unguarded counterfactual was trained.")
    (OUT/"update_diagnosis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
