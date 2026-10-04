"""Independent best/final and noise tests with paired seed statistics."""
import json
from pathlib import Path
from experiments.phase8_progress_rl.analyze import read, summarize, paired
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def main():
    groups = {}
    for path in (OUT/"heldout").glob("*.csv"):
        row = read(path)[0]
        key = (row["variant"], row["mode"], float(row["noise"]), float(row["initial_angle_deg"]),
               float(row["cabinet_dy"]), float(row["friction_scale"]))
        groups.setdefault(key, {})[int(row["seed"])] = row
    records = []
    for key, seeds in sorted(groups.items()):
        if set(seeds) != set(range(5)):
            raise RuntimeError(f"Incomplete independent group: {key}")
        arm, mode, noise, angle, dy, friction = key
        record = dict(variant=arm, mode=mode, noise=noise, initial_angle_deg=angle,
                      cabinet_dy=dy, friction_scale=friction)
        for metric in ("success", "max_angle", "final_angle", "progress", "regression_amount", "regression_event",
                       "mean_raw_policy_std", "mean_action_std_mc", "mean_xyz_action_std_mc", "mean_gripper_sign_flip_probability"):
            record[metric] = summarize([float(seeds[s][metric]) for s in range(5)])
        nominal = groups.get((arm, mode, 0.0, 0.0, 0.0, 1.0))
        if nominal is not None and key != (arm, mode, 0.0, 0.0, 0.0, 1.0):
            record["paired_vs_nominal"] = paired(record["success"]["per_seed"], [float(nominal[s]["success"]) for s in range(5)])
        if mode == "final":
            best = groups.get((arm, "best", noise, angle, dy, friction))
            if best is not None:
                record["paired_final_minus_best"] = paired(record["success"]["per_seed"], [float(best[s]["success"]) for s in range(5)])
        baseline = groups.get(("A", mode, noise, angle, dy, friction))
        if baseline is not None and arm != "A":
            record["paired_vs_A"] = paired(record["success"]["per_seed"], [float(baseline[s]["success"]) for s in range(5)])
        records.append(record)
    (OUT/"heldout_summary.json").write_text(json.dumps(records, indent=2))
    print([{k: r[k] for k in ("variant", "mode", "noise")} | {"success": r["success"]["mean"]} for r in records])


if __name__ == "__main__":
    main()
