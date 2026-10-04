"""Independent best/final and frozen-source tests; isolated physical overrides."""
import argparse
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--arm',required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--smoke',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(headless=True).app
import copy
import json
import math
import os
import traceback
from pathlib import Path
import torch
import numpy as np
import isaaclab_tasks
from door_env.door import door_index,door_angle,handle_pose,robot_observation
from progress_rl.door_env import config,create
from safe_online.std_schedule import ControlledActor
from safe_online.anchor_policy import AnchoredSAC
from trajectory_quality.quality_model import trajectory_score
from experiments.phase9_safe_online.metrics import evaluate
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'/('heldout_smoke' if args.smoke else 'heldout')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/f'{args.arm}_seed{args.seed}.json'
    if output.exists():raise RuntimeError('Independent test already exists')
    folder=ROOT/'checkpoints/phase10_quality_lwd'/(f'P10{args.arm}_seed{args.seed}'+('_smoke' if args.smoke else ''))
    if not (folder/'completed.json').exists():raise RuntimeError('Training incomplete')
    total_steps=json.loads((folder/'completed.json').read_text())['steps']
    seed=210000+args.seed
    env=create(config('B',32,args.device or 'cuda:0',seed))
    actor=ControlledActor().to(env.device).eval()
    anchor=copy.deepcopy(actor)
    source=torch.load(ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{args.seed}'/'best.pt',map_location=env.device,weights_only=False)
    anchor.load_state_dict(source['actor'])
    cabinet=env.scene['cabinet']
    base_root=cabinet.data.default_root_state.clone()
    base_joints=cabinet.data.default_joint_pos.clone()
    base_material=cabinet.root_physx_view.get_material_properties().clone()
    env_ids=torch.arange(env.num_envs,device='cpu')
    conditions=[('nominal',0.,0.,1.)]
    if not args.smoke and args.arm in ('REFC','B','D10','E'):
        conditions += [('angle_2p5',2.5,0.,1.),('angle_5',5.,0.,1.),
                       ('handle_y_minus_1cm',0.,-.01,1.),('handle_y_plus_1cm',0.,.01,1.),
                       ('friction_0p9',0.,0.,.9),('friction_1p1',0.,0.,1.1)]
    rows=[]
    q_diagnosis=[]
    baseline_obs=None
    baseline_gt=None
    try:
        for label,angle,dy,friction in conditions:
            cabinet.data.default_root_state.copy_(base_root)
            cabinet.data.default_root_state[:,1] += dy
            cabinet.data.default_joint_pos.copy_(base_joints)
            cabinet.data.default_joint_pos[:,door_index(env)] = math.radians(angle)
            materials=base_material.clone()
            materials[:,:,:2] *= friction
            cabinet.root_physx_view.set_material_properties(materials,env_ids)
            observed_material=cabinet.root_physx_view.get_material_properties()
            if not torch.allclose(observed_material,materials,atol=1e-6):raise RuntimeError('Friction override not physically applied')
            env.reset(seed=seed)
            if not torch.allclose(cabinet.root_physx_view.get_material_properties(),materials,atol=1e-6):
                raise RuntimeError('Reset changed requested material override')
            initial_obs=robot_observation(env).clone()
            initial_gt=env.get_tool_state().clone()
            if label=='nominal':
                baseline_obs=initial_obs.clone();baseline_gt=initial_gt.clone()
            obs_delta=float((initial_obs-baseline_obs).abs().max())
            gt_delta=float((initial_gt-baseline_gt).abs().max())
            observed_angle=float(door_angle(env).mean())
            observed_handle=(handle_pose(env)[0]-env.scene.env_origins).mean(0).tolist()
            if abs(observed_angle-math.radians(angle))>.001:raise RuntimeError('Initial angle override failed')
            if abs(float(cabinet.data.root_pos_w[:,1].sub(env.scene.env_origins[:,1]).mean())-dy)>.001:raise RuntimeError('Fixture translation override failed')
            selections=['best','final']+(['initial'] if args.arm=='REFC' and label=='nominal' else [])
            for selection in selections:
                name='best.pt' if selection=='best' else f'step_{total_steps}.pt' if selection=='final' else 'step_0.pt'
                state=torch.load(folder/name,map_location=env.device,weights_only=False)
                actor.load_state_dict(state['actor']);actor.cap=state['std_cap']
                for mode in ('deterministic','policy'):
                    trajectory_writer=None
                    if label=='nominal' and selection=='final' and mode=='policy':
                        frozen=AnchoredSAC(state,env.device,1.)
                        frozen.actor.cap=.01
                        def trajectory_writer(trajectory,episode):
                            with torch.no_grad():
                                obs=torch.as_tensor(trajectory['robot'],device=env.device)
                                actions=torch.as_tensor(trajectory['action'],device=env.device)
                                mean,log_std=frozen.actor.distribution(obs)
                                # Saturated float32 tanh actions cannot recover their
                                # pre-tanh samples. Use conditional expected entropy,
                                # GH12 and the SAME historical epsilon convention as
                                # the learner; never infer logp through lossy atanh.
                                nodes,weights=np.polynomial.hermite.hermgauss(12)
                                nodes=torch.as_tensor(nodes,device=env.device,dtype=mean.dtype)
                                weights=torch.as_tensor(weights/math.sqrt(math.pi),device=env.device,dtype=mean.dtype)
                                z=mean[None]+math.sqrt(2)*log_std.exp()[None]*nodes[:,None,None]
                                jac=(1-z.tanh().square()+1e-6).log()
                                entropy=(log_std+.5*math.log(2*math.pi*math.e)+(weights[:,None,None]*jac).sum(0)).sum(-1)
                                discounts=frozen.cfg.gamma**torch.arange(len(obs),device=env.device)
                                rewards=torch.as_tensor(trajectory['reward'][:,0],device=env.device)
                                mc=float((discounts*rewards).sum())
                                soft_mc=mc+float(frozen.log_alpha.exp()*(discounts[1:]*entropy[1:]).sum())
                                q1,q2=frozen.critic(obs[:1],actions[:1])
                                if not np.isfinite(soft_mc) or not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
                                    raise RuntimeError('Nonfinite frozen value diagnostic')
                                record=trajectory_score(trajectory['privileged'],trajectory['next_privileged'],episode['success'])
                                q_diagnosis.append(dict(arm=args.arm,seed=args.seed,checkpoint='final',
                                    initial_q=float(torch.minimum(q1,q2)),initial_q1=float(q1),initial_q2=float(q2),reward_mc=mc,soft_mc=soft_mc,
                                    return_value=episode['return'],episode_steps=episode['episode_steps'],**record))
                        # Same fixed evaluation; capturing complete trajectories adds no rollout.
                    metrics=evaluate(actor,anchor,env,seed,mode,rounds=2,trajectory_writer=trajectory_writer)
                    rows.append(dict(arm=args.arm,seed=args.seed,checkpoint=selection,
                        checkpoint_step=state['env_steps'],condition=label,requested_angle_deg=angle,
                        requested_fixture_dy=dy,requested_friction_multiplier=friction,
                        observed_initial_angle=observed_angle,observed_handle_xyz=observed_handle,
                        initial_robot_obs_max_abs_delta=obs_delta,initial_gt_max_abs_delta=gt_delta,
                        observed_static_friction_mean=float(observed_material[:,:,0].mean()),
                        observed_dynamic_friction_mean=float(observed_material[:,:,1].mean()),std_cap=actor.cap,**metrics))
                    (OUT/f'{args.arm}_seed{args.seed}.partial.json').write_text(json.dumps(rows,indent=2))
            print(f'{args.arm}/{args.seed} independent {label} complete',flush=True)
        output.write_text(json.dumps(rows,indent=2))
        (OUT/f'q_diagnosis_{args.arm}_seed{args.seed}.json').write_text(json.dumps(q_diagnosis,indent=2))
    except BaseException:
        error=traceback.format_exc()
        (OUT/f'{args.arm}_seed{args.seed}_error.txt').write_text(error)
        print(error,flush=True);os._exit(1)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        print(traceback.format_exc(),flush=True);os._exit(1)
    finally:app.close()
