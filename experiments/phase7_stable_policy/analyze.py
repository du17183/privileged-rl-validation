"""Paired five-seed Phase 7 summary at exactly 300k environment steps."""

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
ARMS = ("E0", "E1L", "E2L", "E3", "L000", "L001", "L005", "L010", "L050", "G1")
STEPS = 300000
TCRIT5 = 2.7764451051977987


def read_run(arm, seed):
    if arm in ("E0", "L000"):
        old = "B3" if arm == "E0" else "B1"
        path = ROOT / "results" / "phase6_stable_online_rl" / f"eval_P6{old}_seed{seed}.csv"
    else:
        path = OUT / f"eval_P7{arm}_seed{seed}.csv"
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if int(r["env_steps"]) <= STEPS]
    x = np.asarray([int(r["env_steps"]) for r in rows], dtype=np.int64)
    y = np.asarray([float(r["success_rate"]) for r in rows], dtype=np.float64)
    raw = np.asarray([float(r.get("raw_success_rate", r["success_rate"])) for r in rows], dtype=np.float64)
    if len(x) != 31 or x[0] != 0 or x[-1] != STEPS or np.any(np.abs(np.diff(x) - 10000) > 32):
        raise RuntimeError(f"Incomplete or mismatched evaluation grid: {path}")
    train_episodes = sum(int(r["train_episodes"]) for r in rows)
    train_successes = sum(round(float(r["train_success_rate"]) * int(r["train_episodes"]))
                          for r in rows if int(r["train_episodes"]) > 0)
    return {"arm": arm, "seed": seed,
            "auc": float(np.trapz(y, x) / STEPS),
            "final": float(y[-1]), "best": float(y.max()),
            "gap": float(y.max() - y[-1]),
            "raw_auc": float(np.trapz(raw, x) / STEPS),
            "raw_final": float(raw[-1]),
            "raw_gap": float(raw.max() - raw[-1]),
            "raw_degradation_points": sum(int(raw[i] < raw[:i].max() - 0.25)
                                          for i in range(1, len(raw))),
            "protected_degradation_points": sum(int(y[i] < y[:i].max() - 0.25)
                                                for i in range(1, len(y))),
            "best_step": int(x[y.argmax()]),
            "first_regression_step": int(next((x[i] for i in range(1, len(x))
                                                if y[i] < y[:i].max() - 0.25), -1)),
            "online_train_episodes": train_episodes,
            "online_train_successes": train_successes,
            "cumulative_eval_env_steps": int(rows[-1]["eval_env_steps"]),
            "rollback_count": int(rows[-1]["rollback_count"]) if arm == "G1" else 0}


def summarize(values):
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    half = TCRIT5 * sd / math.sqrt(len(arr))
    return {"mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def paired(x, y):
    delta = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    observed = abs(delta.mean())
    permutations = [abs(np.mean(delta * signs)) for signs in itertools.product((-1, 1), repeat=len(delta))]
    p = float(np.mean(np.asarray(permutations) >= observed - 1e-12))
    return {"difference": summarize(delta), "exact_signflip_p_two_sided": p}


def main():
    per_seed = [read_run(arm, seed) for arm in ARMS for seed in range(5)]
    by_arm = {arm: [r for r in per_seed if r["arm"] == arm] for arm in ARMS}
    summary = {}
    for arm, runs in by_arm.items():
        summary[arm] = {key: summarize([r[key] for r in runs])
                        for key in ("auc", "final", "best", "gap", "raw_auc", "raw_final", "raw_gap",
                                    "raw_degradation_points", "protected_degradation_points",
                                    "online_train_episodes", "online_train_successes",
                                    "cumulative_eval_env_steps", "rollback_count")}
        summary[arm]["paired_vs_E0"] = {key: paired([r[key] for r in runs],
                                                    [r[key] for r in by_arm["E0"]])
                                         for key in ("auc", "final", "gap")}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "phase7_per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_seed[0].keys())
        writer.writeheader()
        writer.writerows(per_seed)
    (OUT / "phase7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    exploratory = {}
    for arm in ("E1", "E2"):
        runs = [read_run(arm, seed) for seed in range(5)]
        exploratory[arm] = {key: summarize([r[key] for r in runs])
                            for key in ("auc", "final", "best", "gap",
                                        "online_train_successes", "cumulative_eval_env_steps")}
        exploratory[arm]["paired_vs_E0"] = {
            key: paired([r[key] for r in runs], [r[key] for r in by_arm["E0"]])
            for key in ("auc", "final", "gap")}
    (OUT / "exploratory_target_entropy_summary.json").write_text(
        json.dumps(exploratory, indent=2), encoding="utf-8")
    print(json.dumps({arm: {k: round(summary[arm][k]["mean"], 4)
                            for k in ("auc", "best", "final", "gap")}
                      for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()
