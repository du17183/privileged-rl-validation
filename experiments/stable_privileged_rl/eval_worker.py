"""Persistent, separate Isaac Lab process for fixed Door validation.

The learner's simulation context is never reset or stepped by this worker.
Requests and responses are atomic JSON files; Actor weights are temporary.
"""

import argparse
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--variant',required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--num-envs',type=int,default=32)
parser.add_argument('--ipc-dir',required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(headless=True).app

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import isaaclab_tasks  # noqa: F401

from auxiliary_learning.gt_prediction import EncodedGaussianActor
from auxiliary_learning.multitask_encoder import MultiTaskActor
from door_env.door import SUCCESS_ANGLE_RAD,make_cfg,robot_observation
from door_env.isaac_env import create_env

IPC=Path(args.ipc_dir)


def atomic_json(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj),encoding='utf-8')
    os.replace(temp,path)


@torch.no_grad()
def evaluate(actor,env,seed,rounds):
    env.reset(seed=seed)
    counts=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    returns=torch.zeros(env.num_envs,device=env.device)
    ever_contact=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    success,rewards,contacts=[],[],[]
    interactions=0
    max_vec_steps=int(env.cfg.episode_length_s/env.step_dt+2)*(rounds+1)
    for _ in range(max_vec_steps):
        action,_=actor(robot_observation(env),deterministic=True)
        _,reward,terminated,truncated,_=env.step(action)
        interactions+=env.num_envs
        returns+=reward
        done=terminated|truncated
        contact=(env.get_tool_state()[:,-2:]>0.5).all(dim=-1)
        if done.any():
            contact[done]=(env.final_privileged[done,-2:]>0.5).all(dim=-1)
        ever_contact|=contact
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if counts[i]<rounds:
                success.append(float(env.final_door_angle[i,0]>SUCCESS_ANGLE_RAD))
                rewards.append(float(returns[i]))
                contacts.append(float(ever_contact[i]))
                counts[i]+=1
            returns[i]=0
            ever_contact[i]=False
        if bool(torch.all(counts>=rounds)):
            break
    expected=env.num_envs*rounds
    if len(success)!=expected:
        raise RuntimeError(f'Worker collected {len(success)}/{expected} evaluation episodes')
    return {'success_rate':float(np.mean(success)),
            'mean_return':float(np.mean(rewards)),
            'contact_rate':float(np.mean(contacts)),
            'episodes':expected,'eval_env_steps':interactions}


def main():
    IPC.mkdir(parents=True,exist_ok=True)
    cfg=make_cfg(args.num_envs,device=args.device or 'cuda:0',seed=50000+args.seed)
    env=create_env(cfg)
    actor=(MultiTaskActor(26,7) if args.variant=='E100M'
           else EncodedGaussianActor(26,7)).to(env.device)
    actor.eval()
    try:
        atomic_json(IPC/'ready.json',{'pid':os.getpid(),'num_envs':env.num_envs})
        sequence=0
        while not (IPC/'stop').exists():
            request=IPC/f'request_{sequence}.json'
            if not request.exists():
                time.sleep(.05)
                continue
            command=json.loads(request.read_text(encoding='utf-8'))
            state=torch.load(command['actor_path'],map_location=env.device,weights_only=True)
            actor.load_state_dict(state,strict=True)
            result=evaluate(actor,env,int(command['eval_seed']),int(command['rounds']))
            atomic_json(IPC/f'response_{sequence}.json',result)
            print(f"EVAL_WORKER sequence={sequence} success={result['success_rate']:.3f}",flush=True)
            sequence+=1
    finally:
        env.close()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        import traceback
        IPC.mkdir(parents=True,exist_ok=True)
        detail=traceback.format_exc()
        atomic_json(IPC/'error.json',{'traceback':detail})
        print(detail,flush=True)
        raise
    finally:
        app.close()
