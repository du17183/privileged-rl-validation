"""Aggregate independent 64-episode checkpoint tests by training seed."""

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
DATA = OUT / "robustness"
TCRIT5 = 2.7764451051977987


def summarize(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    h = TCRIT5 * sd / math.sqrt(len(a))
    return {"mean": mean, "sd": sd, "ci95": [mean - h, mean + h],
            "per_seed": a.tolist()}


def paired(values, controls):
    d = np.asarray(values) - np.asarray(controls)
    p = np.mean([abs(np.mean(d * signs)) >= abs(d.mean()) - 1e-12
                 for signs in itertools.product((-1, 1), repeat=5)])
    return {"difference": summarize(d), "exact_signflip_p_two_sided": float(p)}


def main():
    rows = []
    for path in DATA.glob("*.csv"):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    groups = {}
    policy_stds = {}
    for r in rows:
        key = (r["variant"], r["mode"], float(r["noise_pre_tanh_std"]),
               float(r["requested_door_angle_deg"]), float(r["requested_cabinet_dx_m"]),
               float(r["requested_cabinet_dy_m"]))
        groups.setdefault(key, {})[int(r["seed"])] = float(r["success_rate"])
        policy_stds.setdefault(key, {})[int(r["seed"])] = float(r["reset_mean_policy_std"])
    records = []
    for key, seeds in sorted(groups.items()):
        if set(seeds) != set(range(5)):
            raise RuntimeError(f"Incomplete robustness group: {key}: {sorted(seeds)}")
        arm, mode, noise, angle, dx, dy = key
        values = [seeds[s] for s in range(5)]
        nominal_key = (arm, mode, 0.0, 0.0, 0.0, 0.0)
        nominal = groups.get(nominal_key)
        record = {"variant": arm, "mode": mode, "noise_pre_tanh_std": noise,
                  "door_angle_deg": angle, "cabinet_dx_m": dx, "cabinet_dy_m": dy,
                  "success": summarize(values),
                  "reset_mean_policy_std": summarize([policy_stds[key][s] for s in range(5)])}
        if nominal is not None and key != nominal_key:
            record["paired_vs_nominal"] = paired(values, [nominal[s] for s in range(5)])
        records.append(record)
    (OUT / "robustness_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps([{"variant": r["variant"], "mode": r["mode"],
                       "noise": r["noise_pre_tanh_std"], "angle": r["door_angle_deg"],
                       "dy": r["cabinet_dy_m"], "success": round(r["success"]["mean"], 3)}
                      for r in records], indent=2))


if __name__ == "__main__":
    main()
