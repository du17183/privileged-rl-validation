"""Matched checkpoint diagnostics around the first large success drop."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
ARMS = ("E0", "E1L", "E2L", "E3", "G1", "L000", "L001", "L005", "L010", "L050", "E1", "E2")
METRICS = ("critic_loss", "actor_loss", "bc_loss", "q_mean", "q_variance", "q_disagreement",
           "policy_entropy", "alpha", "action_drift_mse", "expert_action_mse")


def rows(arm, seed, kind):
    if arm in ("E0", "L000"):
        old = "B3" if arm == "E0" else "B1"
        path = ROOT / "results" / "phase6_stable_online_rl" / f"{kind}_P6{old}_seed{seed}.csv"
    else:
        path = OUT / f"{kind}_P7{arm}_seed{seed}.csv"
    with path.open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if int(r["env_steps"]) <= 300000]


def main():
    snapshots, drops = [], []
    for arm in ARMS:
        for seed in range(5):
            evals = rows(arm, seed, "eval")
            diags = rows(arm, seed, "diagnostics")
            by_step = {int(r["env_steps"]): r for r in diags}
            last = by_step[300000]
            snapshots.append(dict(arm=arm, seed=seed, env_steps=300000,
                                  **{k: float(last[k]) for k in METRICS}))
            best = float(evals[0]["success_rate"])
            for i, row in enumerate(evals[1:], start=1):
                current = float(row["success_rate"])
                if current < best - 0.25:
                    before_step = int(evals[i - 1]["env_steps"])
                    after_step = int(row["env_steps"])
                    if before_step in by_step and after_step in by_step:
                        before = by_step[before_step]
                        after = by_step[after_step]
                        drops.append(dict(arm=arm, seed=seed, step=after_step,
                                          success_before=float(evals[i-1]["success_rate"]),
                                          success_after=current,
                                          previous_best=best,
                                          **{f"delta_{k}": float(after[k])-float(before[k])
                                             for k in METRICS}))
                    break
                best = max(best, current)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, records in (("final_diagnostics_per_seed.csv", snapshots),
                          ("first_drop_diagnostics.csv", drops)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    for arm in ARMS:
        arm_rows = [r for r in snapshots if r["arm"] == arm]
        print(arm, "entropy", round(np.mean([r["policy_entropy"] for r in arm_rows]), 3),
              "alpha", round(np.mean([r["alpha"] for r in arm_rows]), 3),
              "drift", round(np.mean([r["action_drift_mse"] for r in arm_rows]), 5),
              "first_drops", len([r for r in drops if r["arm"] == arm]))


if __name__ == "__main__":
    main()
