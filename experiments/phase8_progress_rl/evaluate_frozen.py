"""Independent heldout best/final tests with measurable fixture inputs."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=("best", "final"), required=True)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--angle-deg", type=float, default=0.0)
parser.add_argument("--cabinet-dy", type=float, default=0.0)
parser.add_argument("--friction-scale", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import csv
import math
from pathlib import Path
import torch
import isaaclab_tasks  # noqa
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from progress_rl.door_env import config, create
from evaluation.progress_metrics import evaluate
ROOT = Path(__file__).resolve().parents[2]


def main():
    folder = ROOT/"checkpoints"/"phase8_progress_rl"/f"P8{args.variant}_seed{args.seed}"
    state = torch.load(folder/("best.pt" if args.mode == "best" else "step_300000.pt"), map_location=args.device or "cuda:0", weights_only=False)
    cfg = config("A", 32, args.device or "cuda:0", 90000+args.seed)
    if args.angle_deg:
        cfg.scene.cabinet.init_state.joint_pos = dict(cfg.scene.cabinet.init_state.joint_pos)
        cfg.scene.cabinet.init_state.joint_pos["door_right_joint"] = math.radians(args.angle_deg)
    if args.cabinet_dy:
        x, y, z = cfg.scene.cabinet.init_state.pos
        cfg.scene.cabinet.init_state.pos = (x, y+args.cabinet_dy, z)
    if args.friction_scale != 1.0:
        # Scale the unchanged cabinet startup physical-material ranges, for evaluation only.
        term = cfg.events.cabinet_physics_material
        term.params = dict(term.params)
        for key in ("static_friction_range", "dynamic_friction_range"):
            term.params[key] = tuple(v*args.friction_scale for v in term.params[key])
    env = create(cfg)
    try:
        actor = EncodedGaussianActor(31 if args.variant in ("C", "D", "E") else 26, 7).to(env.device)
        actor.load_state_dict(state["actor"])
        actor.eval()
        result = evaluate(actor, env, args.variant, 90000+args.seed, noise=args.noise)
        row = dict(variant=args.variant, seed=args.seed, mode=args.mode, checkpoint_step=state["env_steps"],
                   noise=args.noise, initial_angle_deg=args.angle_deg, cabinet_dy=args.cabinet_dy,
                   friction_scale=args.friction_scale, **result)
        out = ROOT/"results"/"phase8_progress_rl"/"heldout"
        out.mkdir(parents=True, exist_ok=True)
        path = out/f"{args.variant}_s{args.seed}_{args.mode}_n{args.noise:g}_a{args.angle_deg:g}_y{args.cabinet_dy:g}_f{args.friction_scale:g}.csv"
        if path.exists():
            raise RuntimeError(f"Independent test already exists: {path}")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        print(row, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
