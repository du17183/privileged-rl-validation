"""Paired progress interventions, with protected and pre-recovery metrics."""
import csv
import itertools
import json
import math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def summarize(values):
    a = np.asarray(values, dtype=float)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = 2.7764451051977987*sd/math.sqrt(5)
    return dict(mean=mean, sd=sd, variance=sd*sd, ci95=[mean-half, mean+half], per_seed=a.tolist())


def paired(x, y):
    difference = np.asarray(x)-np.asarray(y)
    p = np.mean([abs(np.mean(difference*signs)) >= abs(difference.mean())-1e-12
                 for signs in itertools.product((-1, 1), repeat=5)])
    return dict(difference=summarize(difference), exact_signflip_p_two_sided=float(p))


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def first_step(rows, threshold, column="success"):
    return next((int(r["env_steps"]) for r in rows if float(r[column]) >= threshold), None)


def main():
    summaries, records = {}, []
    arms = ("A", "B", "C", "D", "E") if (OUT/"eval_P8E_seed4.csv").exists() else ("A", "B", "C", "D")
    for arm in arms:
        seeds = []
        for seed in range(5):
            run = f"P8{arm}_seed{seed}"
            rows = read(OUT/f"eval_{run}.csv")
            assert len(rows) == 31 and int(rows[-1]["env_steps"]) == 300000, run
            steps = np.array([int(r["env_steps"]) for r in rows])
            success = np.array([float(r["success"]) for r in rows])
            raw = np.array([float(r["raw_success"]) for r in rows])
            last = rows[-1]
            noise = read(OUT/f"noise_{run}.csv")
            initial, best, final = success[0], success.max(), success[-1]
            record = dict(arm=arm, seed=seed, auc=float(np.trapz(success, steps)/300000),
                          raw_auc=float(np.trapz(raw, steps)/300000),
                          initial=initial, best=best, final=final, gap=best-final,
                          raw_best=raw.max(), raw_final=raw[-1], raw_gap=raw.max()-raw[-1],
                          noise_final=float(noise[-1]["success"]),
                          max_angle=float(last["max_angle"]), final_angle=float(last["final_angle"]),
                          progress=float(last["progress"]), regression_amount=float(last["regression_amount"]),
                          regression_rate=float(last["regression_event"]),
                          online_successes=int(last["online_successes"]), rollbacks=int(last["rollbacks"]),
                          selected_best_success=float(last["selected_best_success"]),
                          selected_best_step=int(last["selected_best_step"]),
                          training_plus_evaluation_steps=300000+int(last["cumulative_eval_steps"]),
                          evaluation_steps=int(last["cumulative_eval_steps"]))
            for threshold in (0.5, 0.8, 0.9):
                record[f"first_{int(threshold*100)}_steps"] = first_step(rows, threshold)
                record[f"raw_first_{int(threshold*100)}_steps"] = first_step(rows, threshold, "raw_success")
                reached = next((r for r in rows if float(r["success"]) >= threshold), None)
                record[f"first_{int(threshold*100)}_total_interactions"] = (
                    int(reached["env_steps"])+int(reached["cumulative_eval_steps"]) if reached else None)
            seeds.append(record)
            records.append(record)
        numeric = [key for key in seeds[0] if key not in ("arm", "seed") and "first_" not in key]
        summaries[arm] = {key: summarize([r[key] for r in seeds]) for key in numeric}
        summaries[arm]["threshold_steps"] = {key: [r[key] for r in seeds] for key in seeds[0] if "first_" in key}
    comparisons = {}
    for arm, control in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "B"), ("D", "C")):
        comparisons[f"{arm}-{control}"] = {key: paired(summaries[arm][key]["per_seed"], summaries[control][key]["per_seed"])
                                         for key in ("auc", "raw_auc", "final", "raw_final", "gap", "progress", "online_successes")}
    if "E" in summaries:
        comparisons["E-D"] = {key: paired(summaries["E"][key]["per_seed"], summaries["D"][key]["per_seed"])
                                for key in ("auc", "raw_auc", "final", "raw_final", "gap", "progress", "online_successes")}
    interaction = {}
    for key in ("auc", "raw_auc", "final", "raw_final"):
        d, b, c, a = [np.array(summaries[arm][key]["per_seed"]) for arm in ("D", "B", "C", "A")]
        interaction[key] = paired(d-b-c+a, np.zeros(5))
    (OUT/"summary.json").write_text(json.dumps(summaries, indent=2))
    (OUT/"paired_tests.json").write_text(json.dumps(dict(comparisons=comparisons, factorial_interaction=interaction), indent=2))
    with (OUT/"per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    for arm, s in summaries.items():
        print(arm, {key: round(s[key]["mean"], 4) for key in ("auc", "raw_auc", "initial", "best", "final", "gap", "raw_final", "online_successes", "progress")})


if __name__ == "__main__":
    main()
