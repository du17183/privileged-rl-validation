"""Persistent isolated evaluator, including actual stochastic policy execution."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--ipc-dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import json
import os
import time
from pathlib import Path
import torch
import isaaclab_tasks  # noqa
from safe_online.std_schedule import ControlledActor
from progress_rl.door_env import config, create
from experiments.phase9_safe_online.metrics import evaluate
ROOT = Path(__file__).resolve().parents[2]
IPC = Path(args.ipc_dir)


def atomic(path, result):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result))
    os.replace(temp, path)


def main():
    env = create(config("B", 32, args.device or "cuda:0", 70000+args.seed))
    actor = ControlledActor().to(env.device).eval()
    anchor = ControlledActor().to(env.device).eval()
    old = torch.load(ROOT/"checkpoints"/"phase8_progress_rl"/f"P8B_seed{args.seed}"/"best.pt",
                     map_location=env.device, weights_only=False)
    anchor.load_state_dict(old["actor"])
    try:
        atomic(IPC/"ready.json", dict(pid=os.getpid()))
        sequence = 0
        while not (IPC/"stop").exists():
            path = IPC/f"request_{sequence}.json"
            if not path.exists():
                time.sleep(.025)
                continue
            command = json.loads(path.read_text())
            state = torch.load(command["actor_path"], map_location=env.device, weights_only=False)
            actor.load_state_dict(state["actor"])
            actor.cap = state["std_cap"]
            metrics = evaluate(actor, anchor, env, command["seed"], command["mode"], command["rounds"])
            atomic(IPC/f"response_{sequence}.json", metrics)
            sequence += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        atomic(IPC/"error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        app.close()

