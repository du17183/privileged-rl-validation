import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--arm', required=True)
parser.add_argument('--ipc-dir', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import json
import os
import time
from pathlib import Path
import torch
import isaaclab_tasks
from safe_online.std_schedule import ControlledActor
from environment_state.normalization import DIMS
from randomized_env.door_randomization import create
from experiments.phase11_parameter_generalization.agent import expanded_model
from experiments.phase11_parameter_generalization.protocol import ROOT, prepared
from experiments.phase11_parameter_generalization.metrics import evaluate
IPC = Path(args.ipc_dir)
def atomic(path, result):
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(result));os.replace(temp, path)
try:
    center = prepared()['handle_center']
    env = create(32, args.device or 'cuda:0', args.seed)
    actor = ControlledActor(DIMS[args.arm], 7).to(env.device).eval()
    anchor = ControlledActor(DIMS[args.arm], 7).to(env.device).eval()
    old = torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt', map_location=env.device, weights_only=False)
    anchor.load_state_dict(expanded_model(old['actor'], anchor.state_dict()))
    anchor.cap = .01
    atomic(IPC/'ready.json', dict(pid=os.getpid()))
    sequence = 0
    while not (IPC/'stop').exists():
        request = IPC/f'request_{sequence}.json'
        if not request.exists(): time.sleep(.025);continue
        command = json.loads(request.read_text())
        state = torch.load(command['actor_path'], map_location=env.device, weights_only=False)
        actor.load_state_dict(state['actor']);actor.cap = state['std_cap']
        results = {}
        for item in command['tests']:
            key = f"{item['condition']}:{item['mode']}:{item['seed']}"
            results[key] = evaluate(actor, anchor, env, args.arm, center, item['seed'], item['condition'], item['mode'], item['rounds'])
        atomic(IPC/f'response_{sequence}.json', results)
        sequence += 1
    env.close()
except BaseException:
    import traceback
    atomic(IPC/'error.json', dict(traceback=traceback.format_exc()))
    raise
finally: app.close()
