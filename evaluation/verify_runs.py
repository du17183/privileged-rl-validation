"""Fail the experiment if any required full run is absent or incomplete."""

import csv
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    for seed in range(8):
        for variant in "ABC":
            name = f"{variant}_seed{seed}"
            path = root / "results" / f"eval_{name}.csv"
            if not path.exists():
                errors.append(f"missing CSV: {path}")
                continue
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if not rows or int(rows[0]["env_steps"]) != 0:
                errors.append(f"missing BC-only evaluation: {path}")
            if not rows or int(rows[-1]["env_steps"]) < 200000:
                errors.append(f"online budget incomplete: {path}")
            if len(rows) < 21:
                errors.append(f"too few evaluation checkpoints: {path} ({len(rows)})")
            if any(int(row["eval_episodes"]) != 64 for row in rows):
                errors.append(f"evaluation episodes differ from protocol: {path}")
            final_checkpoint = root / "checkpoints" / name / "step_200000.pt"
            if not final_checkpoint.exists():
                errors.append(f"missing final checkpoint: {final_checkpoint}")
            error_log = root / "logs" / f"{name}.error.txt"
            if error_log.exists():
                errors.append(f"run raised exception: {error_log}")
        a_zero = root / "checkpoints" / f"A_seed{seed}" / "step_0.pt"
        b_zero = root / "checkpoints" / f"B_seed{seed}" / "step_0.pt"
        a_eval = root / "results" / f"eval_A_seed{seed}.csv"
        b_eval = root / "results" / f"eval_B_seed{seed}.csv"
        if all(path.exists() for path in (a_zero, b_zero, a_eval, b_eval)):
            a_actor = torch.load(a_zero, map_location="cpu", weights_only=False)["actor"]
            b_actor = torch.load(b_zero, map_location="cpu", weights_only=False)["actor"]
            if a_actor.keys() != b_actor.keys() or any(
                not torch.equal(a_actor[key], b_actor[key]) for key in a_actor
            ):
                errors.append(f"A/B paired seed {seed} starts from different actor parameters")
            with a_eval.open(newline="", encoding="utf-8") as stream:
                a_initial = next(csv.DictReader(stream))
            with b_eval.open(newline="", encoding="utf-8") as stream:
                b_initial = next(csv.DictReader(stream))
            if any(a_initial[field] != b_initial[field] for field in ("success_rate", "mean_return")):
                errors.append(f"A/B paired seed {seed} has different BC-only evaluation")
    if errors:
        raise RuntimeError("\n".join(errors))
    print("Verified 24 complete runs and exact A/B BC actor pairing: 8 seeds × A/B/C, each 200000 online steps")


if __name__ == "__main__":
    main()
