"""Independent 64-episode verification of best and final pilot actors."""

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch

from auxiliary_learning.gt_prediction import EncodedGaussianActor
from experiments.stable_privileged_rl.eval_client import EvalClient

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "value_calibration" / "online_pilot"
CKPT = ROOT / "checkpoints" / "value_calibration" / "online_pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("uniform", "quality", "quality_return", "quality_offline"), required=True)
    parser.add_argument("--seed", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = f"{args.arm}_seed{args.seed}"
    output = OUT / f"heldout_{run}.csv"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite held-out result: {output}")
    with (OUT / f"eval_{run}.csv").open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    if len(source) != 11 or int(source[-1]["env_steps"]) != 100000:
        raise RuntimeError(f"Pilot run incomplete: {run}")
    choices = (("best", max(source, key=lambda row: float(row["success_rate"]))),
               ("final", source[-1]))
    actor = EncodedGaussianActor(26, 7)
    agent = SimpleNamespace(actor=actor)
    variant = {"uniform": "P5HU", "quality": "P5HW",
               "quality_return": "P5HR", "quality_offline": "P5HO"}[args.arm]
    evaluator = EvalClient(ROOT, variant, args.seed, 32, args.device)
    rows = []
    try:
        for mode, choice in choices:
            step = int(choice["env_steps"])
            saved = torch.load(CKPT / run / f"step_{step}.pt", map_location="cpu",
                               weights_only=False)
            actor.load_state_dict(saved["actor"])
            success, mean_return, contact, episodes, interactions = evaluator.evaluate(
                agent, 80000 + args.seed, 2)
            rows.append({"arm": args.arm, "seed": args.seed, "mode": mode,
                         "checkpoint_steps": step,
                         "selection_success_32": float(choice["success_rate"]),
                         "heldout_success_64": success,
                         "heldout_mean_return": mean_return,
                         "heldout_contact": contact, "heldout_episodes": episodes,
                         "heldout_interactions": interactions})
    finally:
        evaluator.close()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{run}: best={rows[0]['heldout_success_64']:.3f} "
          f"final={rows[1]['heldout_success_64']:.3f}", flush=True)


if __name__ == "__main__":
    main()
