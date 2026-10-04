"""Independent persistent Isaac evaluator; no mutation of the learner sim."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--ipc-dir", required=True)
parser.add_argument("--num-envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json
import os
import time
from pathlib import Path
import torch
import isaaclab_tasks  # noqa
from auxiliary_learning.gt_prediction import EncodedGaussianActor
from progress_rl.door_env import config, create, stage_success
from evaluation.progress_metrics import evaluate
IPC = Path(args.ipc_dir)


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temp, path)


def main():
    # The evaluation reward is always the historical task reward.
    cfg = config("A", args.num_envs, args.device or "cuda:0", 50000+args.seed)
    from isaaclab.managers import TerminationTermCfg
    cfg.terminations.success = TerminationTermCfg(func=stage_success)
    env = create(cfg)
    actor = EncodedGaussianActor(31 if args.variant in ("C", "D", "E") else 26, 7).to(env.device)
    actor.eval()
    try:
        atomic(IPC/"ready.json", {"pid": os.getpid()})
        index = 0
        while not (IPC/"stop").exists():
            request = IPC/f"request_{index}.json"
            if not request.exists():
                time.sleep(0.05)
                continue
            command = json.loads(request.read_text())
            actor.load_state_dict(torch.load(command["actor_path"], map_location=env.device, weights_only=True))
            env.goal = command.get("target", 1.0)
            metrics = evaluate(actor, env, args.variant, command["eval_seed"], command["rounds"], command["noise"])
            atomic(IPC/f"response_{index}.json", metrics)
            index += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        atomic(IPC/"error.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        app.close()
