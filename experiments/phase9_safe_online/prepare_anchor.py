"""Generate one immutable frozen-anchor dataset/seed, never learner successes."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import copy
import hashlib
import json
import os
import traceback
from pathlib import Path
import torch
import h5py
import isaaclab_tasks  # noqa
from safe_online.std_schedule import ControlledActor
from progress_rl.door_env import config, create
from experiments.phase9_safe_online.metrics import evaluate
from progress_rl.progress_monitor import aggregate
ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT/"datasets"/"phase9_safe_online"
    output.mkdir(parents=True, exist_ok=True)
    path = output/f"anchor_seed{args.seed}.h5"
    checkpoint = ROOT/"checkpoints"/"phase8_progress_rl"/f"P8B_seed{args.seed}"/"best.pt"
    state = torch.load(checkpoint, map_location=args.device or "cuda:0", weights_only=False)
    env = create(config("B", 32, args.device or "cuda:0", 92000+args.seed))
    actor = ControlledActor().to(env.device).eval()
    actor.load_state_dict(state["actor"])
    anchor = copy.deepcopy(actor)
    try:
        existing = path.exists()
        with h5py.File(path, "r" if existing else "x") as h5:
            source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if existing:
                if len(h5) != 64 or h5.attrs['checkpoint_sha256'] != source_hash:
                    raise RuntimeError('Existing anchor dataset is incomplete or has wrong source; preserve and inspect')
                print(f'Reusing validated 64-episode anchor dataset {path}', flush=True)
            else:
                h5.attrs.update(source="frozen_phase8_anchor", seed=args.seed,
                    checkpoint_sha256=source_hash,
                    phase8_checkpoint_step=state["env_steps"], optimizer_updates=0)
            def write(trajectory, row):
                group = h5.create_group(f"traj_{len(h5):04d}")
                group.attrs.update(row)
                for key, value in trajectory.items():
                    group.create_dataset(key, data=value, compression="gzip")
            metrics = {}
            if existing:
                metrics['deterministic'] = evaluate(actor, anchor, env, 92000+args.seed)
                metrics['deterministic']['recovered_dataset_unchanged'] = True
            else:
                metrics["deterministic"] = evaluate(actor, anchor, env, 92000+args.seed,
                    trajectory_writer=write)
            print(f"deterministic success={metrics['deterministic']['success']}", flush=True)
            metrics["original_policy"] = evaluate(actor, anchor, env, 92000+args.seed, "policy")
            for cap in (.1, .01):
                actor.cap = cap
                metrics[f"policy_cap_{cap:g}"] = evaluate(actor, anchor, env, 92000+args.seed, "policy")
            successes = int(sum(g.attrs["success"] == 1 for g in h5.values()))
            transitions = sum(len(g["action"]) for g in h5.values() if g.attrs["success"] == 1)
            if successes < 48:
                raise RuntimeError("Frozen anchor did not pass 75% success data gate")
        result = dict(seed=args.seed, metrics=metrics, anchor_success_episodes=successes,
            anchor_success_transitions=transitions, dataset=str(path),
            source_checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            evaluation_steps=sum(m["eval_env_steps"] for m in metrics.values()))
        (ROOT/"results"/"phase9_safe_online"/f"anchor_seed{args.seed}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        error = traceback.format_exc()
        (ROOT/'results'/'phase9_safe_online'/f'anchor_seed{args.seed}_error.txt').write_text(error)
        print(error, flush=True)
        os._exit(1)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        os._exit(1)
    finally:
        app.close()
