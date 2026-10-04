"""Compact progress snapshot; mixed-step scores are not formal comparisons."""
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def main():
    latest = []
    for arm in ("A", "B", "C", "D"):
        for seed in range(5):
            path = OUT/f"eval_P8{arm}_seed{seed}.csv"
            if not path.exists():
                continue
            with path.open(newline="") as stream:
                rows = [r for r in csv.DictReader(stream) if r.get("wall_time_s")]
            if rows:
                last = rows[-1]
                latest.append(dict(arm=arm, seed=seed, steps=int(last["env_steps"]),
                                   raw=float(last["raw_success"]), protected=float(last["success"]),
                                   online_successes=int(last["online_successes"])))
    completed = sum((ROOT/"checkpoints"/"phase8_progress_rl"/f"P8{arm}_seed{seed}"/"completed.json").exists()
                    for arm in ("A", "B", "C", "D") for seed in range(5))
    print(json.dumps(dict(completed=completed, total=20, latest=latest), indent=2))


if __name__ == "__main__":
    main()
