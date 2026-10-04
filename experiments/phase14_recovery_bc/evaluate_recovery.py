"""Same expert-prepared physical perturbations for every tested policy."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--expert',action='store_true')
p.add_argument('--case',choices=['contact_loss','ee_offset','door_regression'],required=True)
p.add_argument('--episodes',type=int,default=64);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=32)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app
import csv,json,time
from pathlib import Path
import numpy as np,torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase13_random_expert_bc.state import pack
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs
from experiments.phase13_random_expert_bc.analyze import wilson
from recovery_expert.recovery_planner import RecoveryPlanner
from recovery_expert.perturbation import Perturbation


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or out.with_suffix('.csv').exists():raise FileExistsError(out)
    env=create(a.num_envs,a.device or 'cuda:0',a.seed,level=2)
    perturb=Perturbation(env.num_envs,env.device,env.step_dt,a.case)
    planner=RecoveryPlanner(env.num_envs,env.device,env.step_dt);planner.reset(list(range(env.num_envs)),env)
    if not a.expert:policy,mean,std,checkpoint=load_checkpoint(a.checkpoint,env.device)
    count=np.zeros(env.num_envs,dtype=int);quota=np.full(env.num_envs,a.episodes//env.num_envs);quota[:a.episodes%env.num_envs]+=1
    attempted=np.zeros_like(count);handoff=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    start_angle=torch.zeros(env.num_envs,device=env.device);start_dist=start_angle.clone();maximum=start_angle.clone()
    contact=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    snapshots={};starts=[];rows=[];ticks=0;started=time.monotonic()
    try:
        with out.with_suffix('.csv').open('x',newline='') as f:
            columns=['clone','episode','valid_perturbation','success','recontact','handoff_tick','handoff_angle','handoff_distance',
                     'max_angle','final_angle','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            while np.any(count<quota):
                robot=robot_observation(env).clone();gt=env.get_tool_state().clone();context=pack(env.get_environment_state()).clone()
                action,finished,valid=perturb.action(env,robot,gt)
                for i in finished.nonzero().flatten().tolist():
                    planner.takeover([i],env);handoff[i]=env.episode_length_buf[i]
                    start_angle[i]=gt[i,0];start_dist[i]=torch.linalg.vector_norm(robot[i,18:21]-gt[i,2:5])
                    snapshots[i]=dict(robot=robot[i].cpu().numpy(),environment=context[i].cpu().numpy(),parameters=env.parameters[i].cpu().numpy())
                active=perturb.stage==2
                if a.expert:chosen=planner.action(env)
                else:
                    with torch.no_grad():chosen=policy(inputs(robot,context,mean,std,checkpoint['arm']))
                action[active]=chosen[active]
                _,reward,term,trunc,_=env.step(action)
                nr,ng,done=transition_after_step(env,term,trunc)
                maximum=torch.maximum(maximum,ng[:,0]);contact|=(ng[:,9:11]>.5).all(-1)&active
                ids=done.nonzero().flatten().tolist()
                for i in ids:
                    if count[i]<quota[i]:
                        isvalid=bool(perturb.valid[i]);row=dict(clone=i,episode=int(attempted[i]),valid_perturbation=int(isvalid),
                            success=int(ng[i,0]>1),recontact=int(contact[i]),handoff_tick=int(handoff[i]),
                            handoff_angle=float(start_angle[i]),handoff_distance=float(start_dist[i]),
                            max_angle=float(maximum[i]),final_angle=float(ng[i,0]),
                            **dict(zip(columns[-5:],[float(x) for x in env.final_parameters[i]])))
                        rows.append(row);writer.writerow(row);attempted[i]+=1
                        if isvalid:
                            count[i]+=1;starts.append(dict(clone=i,episode=row['episode'],**snapshots[i]))
                        if attempted[i]>12:raise RuntimeError('Cannot physically create requested perturbation reliably')
                    snapshots.pop(i,None)
                if ids:
                    perturb.reset(ids);planner.reset(ids,env)
                    for tensor in [handoff,start_angle,start_dist,maximum,contact]:tensor[ids]=0
                    f.flush()
                ticks+=1
                if ticks%200==0:print(json.dumps(dict(ticks=ticks,attempted=len(rows),valid=int(count.sum()),successes=sum(r['success'] for r in rows if r['valid_perturbation']),elapsed_s=time.monotonic()-started)),flush=True)
            selected=[r for r in rows if r['valid_perturbation']]
            result=dict(case=a.case,expert=a.expert,checkpoint=a.checkpoint,seed=a.seed,episodes=len(selected),attempted=len(rows),
                perturbation_valid_fraction=len(selected)/len(rows),success=float(np.mean([r['success'] for r in selected])),
                success_wilson95=wilson(sum(r['success'] for r in selected),len(selected)),
                recontact=float(np.mean([r['recontact'] for r in selected])),mean_final_angle=float(np.mean([r['final_angle'] for r in selected])),
                interactions=ticks*env.num_envs,elapsed_s=time.monotonic()-started,num_envs=env.num_envs,
                no_joint_teleport=True,no_reset_at_handoff=True,invalid_attempts_reported=True)
            np.savez_compressed(out.with_suffix('.starts.npz'),**{k:np.asarray([s[k] for s in starts]) for k in starts[0]})
            out.write_text(json.dumps(result,indent=2));print('COMPLETE '+json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
