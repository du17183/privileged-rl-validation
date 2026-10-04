"""Live policy prefix -> detected deviation -> expert, without episode reset."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validate','collect','audit'],required=True)
p.add_argument('--episodes',type=int,default=64);p.add_argument('--seed',type=int,required=True)
p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--max-attempts',type=int,default=800)
p.add_argument('--save-trajectories',action='store_true')
p.add_argument('--uniform-clone-quota',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app

import csv,json,time
from collections import defaultdict
from pathlib import Path
import h5py,numpy as np,torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase13_random_expert_bc.state import pack,next_environment
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs
from experiments.phase13_random_expert_bc.analyze import wilson
from recovery_expert.failure_detector import FailureDetector
from recovery_expert.recovery_planner import RecoveryPlanner


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'attempts.csv').exists():raise FileExistsError(out)
    env=create(a.num_envs,a.device or 'cuda:0',a.seed,level=2)
    policy,mean,std,checkpoint=load_checkpoint(a.checkpoint,env.device)
    planner=RecoveryPlanner(env.num_envs,env.device,env.step_dt);planner.reset(list(range(env.num_envs)),env)
    detector=FailureDetector(env.num_envs,env.device,env.scene['robot'].data.soft_joint_pos_limits[:,:7].clone())
    triggered=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
    trigger_tick=torch.zeros_like(triggered);prefix=[defaultdict(list) for _ in range(env.num_envs)]
    recovery=[defaultdict(list) for _ in range(env.num_envs)]
    maxangle=torch.zeros(env.num_envs,device=env.device);contacts=torch.zeros_like(triggered,dtype=torch.bool)
    trigger_angle=torch.zeros_like(maxangle);trigger_dist=trigger_angle.clone()
    episode=np.zeros(env.num_envs,dtype=int);completed=np.zeros_like(episode)
    quota=np.full(env.num_envs,a.episodes//env.num_envs);quota[:a.episodes%env.num_envs]+=1
    rows=[];successes=attempts=saved=ticks=0;started=time.monotonic()
    h5=h5py.File(out/'trajectories.h5','x') if a.mode=='collect' or a.save_trajectories else None
    failures=h5py.File(out/'failed_trajectories.h5','x') if h5 is not None else None
    if h5 is not None:
        for f in [h5,failures]:
            f.attrs['seed']=a.seed;f.attrs['prefix_checkpoint']=a.checkpoint
            f.attrs['no_reset_at_takeover']=True;f.attrs['action_source']='expert only after live takeover'
    try:
        with (out/'attempts.csv').open('x',newline='') as f:
            columns=['clone','episode','triggered','trigger','trigger_tick','trigger_angle','trigger_distance','success',
                     'recovery_length','max_angle','final_angle','recontact','final_phase',
                     'angle0_rad','offset_x','offset_y','offset_z','friction_scale']
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
            while True:
                robot=robot_observation(env).clone();gt=env.get_tool_state().clone();context=pack(env.get_environment_state()).clone()
                parameters=env.parameters.clone();age=env.episode_length_buf.clone()
                code,distance=detector.update(robot,gt)
                newly=(triggered==0)&(code>0)
                if newly.any():
                    ids=newly.nonzero().flatten();triggered[ids]=code[ids];trigger_tick[ids]=age[ids]
                    trigger_angle[ids]=gt[ids,0];trigger_dist[ids]=distance[ids]
                    before_robot=env.scene['robot'].data.joint_pos[ids].clone();before_door=gt[ids,0].clone()
                    planner.takeover(ids,env)
                    assert torch.equal(age[ids],env.episode_length_buf[ids])
                    assert torch.equal(before_robot,env.scene['robot'].data.joint_pos[ids])
                    assert torch.equal(before_door,env.get_tool_state()[ids,0])
                active=(triggered>0)&(a.mode!='audit')
                with torch.no_grad():action=policy(inputs(robot,context,mean,std,checkpoint['arm']))
                expert=planner.action(env);action[active]=expert[active]
                phase=planner.state.clone()
                _,reward,term,trunc,_=env.step(action)
                nr,ng,done=transition_after_step(env,term,trunc);ne=next_environment(env,done)
                maxangle=torch.maximum(maxangle,ng[:,0]);contacts|=(ng[:,9:11]>.5).all(-1)&active
                batch=dict(observation=robot,environment_state=context,state=gt,action=action,reward=reward[:,None],
                           next_observation=nr,next_environment_state=ne,next_state=ng,done=done[:,None],
                           terminated=term[:,None],truncated=trunc[:,None],reset_parameters=parameters,
                           episode_tick=age[:,None],planner_phase=phase[:,None])
                arrays={k:v.cpu().numpy() for k,v in batch.items()}
                for i in range(env.num_envs):
                    if a.mode!='collect' and a.uniform_clone_quota and completed[i]>=quota[i]:continue
                    target=recovery[i] if bool(active[i]) else prefix[i]
                    for k,v in arrays.items():target[k].append(v[i].copy())
                ticks+=1
                ids=done.nonzero().flatten().tolist()
                for i in ids:
                    if a.mode!='collect' and a.uniform_clone_quota and completed[i]>=quota[i]:continue
                    ok=bool(ng[i,0]>1);is_trigger=bool(triggered[i])
                    if a.mode=='collect' and saved>=a.episodes:continue
                    if a.mode!='collect' and not a.uniform_clone_quota and attempts>=a.episodes:continue
                    row=dict(clone=i,episode=int(episode[i]),triggered=int(is_trigger),trigger=FailureDetector.NAMES[int(triggered[i])],
                             trigger_tick=int(trigger_tick[i]),trigger_angle=float(trigger_angle[i]),trigger_distance=float(trigger_dist[i]),
                             success=int(ok),recovery_length=len(recovery[i]['action']),max_angle=float(maxangle[i]),final_angle=float(ng[i,0]),
                             recontact=int(contacts[i]),final_phase=int(planner.state[i]),
                             **dict(zip(columns[-5:],[float(v) for v in env.final_parameters[i]])))
                    writer.writerow(row);rows.append(row);episode[i]+=1
                    if is_trigger:
                        completed[i]+=1;attempts+=1;successes+=int(ok)
                        if h5 is not None:
                            dest=h5 if ok else failures;g=dest.create_group(f'traj_{saved:05d}' if ok else f'failure_{attempts:05d}')
                            for key,value in recovery[i].items():g.create_dataset(key,data=np.asarray(value),compression='gzip',compression_opts=1)
                            g['robot_state']=g['observation'];g['expert_action']=g['action']
                            g.create_dataset('success',data=np.full((row['recovery_length'],1),ok))
                            pg=g.create_group('policy_prefix')
                            for key,value in prefix[i].items():pg.create_dataset(key,data=np.asarray(value),compression='gzip',compression_opts=1)
                            for key,value in row.items():g.attrs[key]=value
                            if ok:saved+=1
                    prefix[i]=defaultdict(list);recovery[i]=defaultdict(list)
                if ids:
                    planner.reset(ids,env);detector.reset(ids)
                    for tensor in [triggered,trigger_tick,maxangle,contacts,trigger_angle,trigger_dist]:tensor[ids]=0
                    f.flush()
                    if h5 is not None:h5.flush();failures.flush()
                if ticks%200==0:
                    print(json.dumps(dict(ticks=ticks,completed_episodes=len(rows),recovery_attempts=attempts,recovery_successes=successes,
                          saved=saved,phase=torch.bincount(planner.state,minlength=7).cpu().tolist(),elapsed_s=time.monotonic()-started)),flush=True)
                if a.mode=='collect' and saved>=a.episodes:break
                if a.mode!='collect' and ((a.uniform_clone_quota and np.all(completed>=quota)) or (not a.uniform_clone_quota and attempts>=a.episodes)):break
                if len(rows)>=a.max_attempts:raise RuntimeError('Recovery attempt limit; retain all evidence')
            summary=dict(mode=a.mode,seed=a.seed,checkpoint=a.checkpoint,episodes=len(rows),recovery_attempts=attempts,
                recovery_successes=successes,recovery_success_rate=successes/max(1,attempts),recovery_95_wilson=wilson(successes,attempts),
                saved_trajectories=saved,untriggered_episodes=sum(not r['triggered'] for r in rows),
                trigger_counts={name:sum(r['trigger']==name for r in rows if r['triggered']) for name in FailureDetector.NAMES.values() if name!='none'},
                failures_by_trigger={name:sum(r['trigger']==name and not r['success'] for r in rows) for name in FailureDetector.NAMES.values() if name!='none'},
                interactions=ticks*env.num_envs,vector_steps=ticks,elapsed_s=time.monotonic()-started,
                takeover_resets=0,robot_or_door_teleports=0,num_envs=env.num_envs,physical_reset_checks=env.physical_reset_checks,
                limitation='Trigger is a deviation proxy; successful untriggered policy episodes excluded from recovery denominator. Initial angle can settle under inherited door drive.')
            (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE '+json.dumps(summary),flush=True)
    finally:
        if h5 is not None:h5.close();failures.close()
        env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
