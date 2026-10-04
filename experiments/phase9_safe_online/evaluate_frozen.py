"""Independent tests; reset overrides affect only this isolated evaluator."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--arm',required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--pilot',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(headless=True).app
import copy
import json
import math
import os
import traceback
from pathlib import Path
import torch
import isaaclab_tasks
from door_env.door import door_index, door_angle, handle_pose
from progress_rl.door_env import config, create
from safe_online.std_schedule import ControlledActor
from experiments.phase9_safe_online.metrics import evaluate
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/phase9_safe_online/heldout'

def main():
    destination=ROOT/'results/phase9_safe_online/heldout_pilot' if args.pilot else OUT
    destination.mkdir(parents=True,exist_ok=True)
    output = destination/f'{args.arm}_seed{args.seed}.json'
    if output.exists():
        raise RuntimeError('Independent test already exists')
    folder = ROOT/'checkpoints/phase9_safe_online'/(f'P9{args.arm}_seed{args.seed}'+('_smoke' if args.pilot else ''))
    if not (folder/'completed.json').exists():
        raise RuntimeError('Training not complete')
    seed = 190000+args.seed
    env = create(config('B',32,args.device or 'cuda:0',seed))
    actor = ControlledActor().to(env.device).eval()
    anchor = copy.deepcopy(actor)
    source = torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt',map_location=env.device,weights_only=False)
    anchor.load_state_dict(source['actor'])
    cabinet = env.scene['cabinet']
    base_root = cabinet.data.default_root_state.clone()
    base_joints = cabinet.data.default_joint_pos.clone()
    conditions = [('nominal',0.,0.)]
    if args.arm in ('A','B','C','D'):
        conditions += [('angle_2p5',2.5,0.),('angle_5',5.,0.),('handle_y_minus_1cm',0.,-.01),('handle_y_plus_1cm',0.,.01)]
    results = []
    try:
        for label,angle,dy in conditions:
            cabinet.data.default_root_state.copy_(base_root)
            cabinet.data.default_root_state[:,1] += dy
            cabinet.data.default_joint_pos.copy_(base_joints)
            cabinet.data.default_joint_pos[:,door_index(env)] = math.radians(angle)
            env.reset(seed=seed)
            observed_angle = float(door_angle(env).mean())
            observed_handle = (handle_pose(env)[0]-env.scene.env_origins).mean(0).tolist()
            if abs(observed_angle-math.radians(angle))>.001:
                raise RuntimeError('Requested initial angle not physically applied')
            if abs(float(cabinet.data.root_pos_w[:,1].sub(env.scene.env_origins[:,1]).mean())-dy)>.001:
                raise RuntimeError('Requested fixture translation not physically applied')
            modes = ['initial'] if args.pilot else ['best','final']
            if not args.pilot and label=='nominal' and args.arm in ('A','C','CANN'):
                modes += ['initial']
            for selection in modes:
                state = torch.load(folder/('best.pt' if selection=='best' else 'step_300000.pt' if selection=='final' else 'step_0.pt'),map_location=env.device,weights_only=False)
                actor.load_state_dict(state['actor'])
                actor.cap = state['std_cap']
                for mode in (('deterministic','policy','noise001') if label=='nominal' and not args.pilot else ('deterministic','policy')):
                    metrics = evaluate(actor,anchor,env,seed,mode,rounds=1 if args.pilot else 2)
                    row = dict(arm=args.arm,seed=args.seed,checkpoint=selection,checkpoint_step=state['env_steps'],
                        condition=label,requested_angle_deg=angle,requested_handle_dy=dy,
                        observed_angle_rad=observed_angle,observed_handle_xyz=observed_handle,
                        std_cap=actor.cap,**metrics)
                    results.append(row)
                    (destination/f'{args.arm}_seed{args.seed}.partial.json').write_text(json.dumps(results,indent=2))
        output.write_text(json.dumps(results,indent=2))
        print(f'Heldout {args.arm}/{args.seed} complete: {len(results)} tests, {sum(r["episodes"] for r in results)} episodes',flush=True)
    except BaseException:
        error = traceback.format_exc()
        (destination/f'{args.arm}_seed{args.seed}_error.txt').write_text(error)
        print(error,flush=True)
        os._exit(1)
    finally:
        env.close()

if __name__=='__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(),flush=True)
        os._exit(1)
    finally:
        app.close()
