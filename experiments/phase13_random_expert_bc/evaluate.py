"""Fresh-process rollout evaluation on independent, matched random resets."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
p.add_argument('--episodes',type=int,default=128);p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--seed',type=int,default=13501);p.add_argument('--level',type=int,default=2)
p.add_argument('--stochastic',action='store_true')
p.add_argument('--mask',choices=['none','none_repeat','angle_only','coherent_angle_progress_remaining','handle_pose','contact','all_environment'],default='none')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app

import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
import isaaclab_tasks  # noqa: F401
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase13_random_expert_bc.state import pack,next_environment
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix('.csv').exists():raise FileExistsError(path)
    env=create(a.num_envs,a.device or 'cuda:0',a.seed,level=a.level)
    policy,mean,std,checkpoint=load_checkpoint(a.checkpoint,env.device)
    generator=torch.Generator(device=env.device).manual_seed(a.seed+100000)
    count=np.zeros(env.num_envs,dtype=int);quotas=np.full(env.num_envs,a.episodes//env.num_envs)
    quotas[:a.episodes%env.num_envs]+=1
    maximum=torch.zeros(env.num_envs,device=env.device);contact=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    contacts=torch.zeros(env.num_envs,device=env.device);returns=torch.zeros_like(contacts)
    lengths=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    rows=[];trace=[];ticks=0;started=time.monotonic()
    try:
        with path.with_suffix('.csv').open('x',newline='') as f:
            columns=['clone','episode','success','length','return','max_angle_rad','final_angle_rad','contact_success','contact_fraction',
                     'angle0_rad','offset_x','offset_y','offset_z','friction_scale']
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            while np.any(count<quotas):
                robot=robot_observation(env);context=pack(env.get_environment_state())
                with torch.no_grad():
                    observation=inputs(robot,context,mean,std,checkpoint['arm'],a.mask)
                    action=policy(observation,a.stochastic,generator)
                _,reward,term,trunc,_=env.step(action)
                next_robot,next_gt,done=transition_after_step(env,term,trunc)
                if count[0]==0:
                    trace.append(dict(robot=robot[0].detach().cpu().numpy().copy(),
                                      environment=context[0].detach().cpu().numpy().copy(),
                                      action=action[0].detach().cpu().numpy().copy(),
                                      next_robot=next_robot[0].detach().cpu().numpy().copy(),
                                      next_state=next_gt[0].detach().cpu().numpy().copy(),
                                      done=bool(done[0])))
                maximum=torch.maximum(maximum,next_gt[:,0])
                grasp=(next_gt[:,9:11]>.5).all(-1)
                contact|=grasp;contacts+=grasp;returns+=reward.flatten();lengths+=1
                ids=done.nonzero().flatten().tolist()
                for i in ids:
                    if count[i]<quotas[i]:
                        row=dict(clone=i,episode=int(count[i]),success=int(next_gt[i,0]>1),length=int(lengths[i]),
                                 return_=float(returns[i]),max_angle_rad=float(maximum[i]),final_angle_rad=float(next_gt[i,0]),
                                 contact_success=int(contact[i]),contact_fraction=float(contacts[i]/lengths[i]),
                                 **dict(zip(columns[-5:],[float(x) for x in env.final_parameters[i]])))
                        row['return']=row.pop('return_');rows.append(row);writer.writerow(row);count[i]+=1
                    maximum[i]=0;contact[i]=False;contacts[i]=0;returns[i]=0;lengths[i]=0
                ticks+=1
                if ids:f.flush()
                if ticks%200==0:print(json.dumps(dict(ticks=ticks,episodes=len(rows),successes=sum(r['success'] for r in rows),elapsed_s=time.monotonic()-started)),flush=True)
            result=dict(checkpoint=a.checkpoint,arm=checkpoint['arm'],model_seed=checkpoint['seed'],
                        evaluation_seed=a.seed,level=a.level,mask=a.mask,stochastic=a.stochastic,
                        episodes=len(rows),success=sum(r['success'] for r in rows)/len(rows),
                        mean_max_angle_rad=float(np.mean([r['max_angle_rad'] for r in rows])),
                        mean_final_angle_rad=float(np.mean([r['final_angle_rad'] for r in rows])),
                        contact_success=float(np.mean([r['contact_success'] for r in rows])),
                        interactions=ticks*env.num_envs,vector_steps=ticks,elapsed_s=time.monotonic()-started,
                        physical_reset_checks=env.physical_reset_checks,dataset_sha256=checkpoint['dataset_sha256'])
            with path.open('x') as out:json.dump(result,out,indent=2)
            if trace:
                np.savez_compressed(path.with_suffix('.trace.npz'),**{key:np.asarray([r[key] for r in trace]) for key in trace[0]})
            print('COMPLETE '+json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
