"""Additional initial-state successes, using the same repaired recovery expert.

Ordinary data controls expert labels and controller recipe, not only counts.
Old Phase13 expert datasets and generators stay unchanged.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validate','collect'],required=True)
p.add_argument('--episodes',type=int,required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--max-attempts',type=int,default=600)
p.add_argument('--output',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(headless=True).app
import csv,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch,h5py
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase13_random_expert_bc.state import pack,next_environment
from experiments.phase13_random_expert_bc.analyze import wilson
from recovery_expert.recovery_planner import RecoveryPlanner


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'attempts.csv').exists():raise FileExistsError(out)
    env=create(a.num_envs,a.device or 'cuda:0',a.seed,level=2)
    planner=RecoveryPlanner(env.num_envs,env.device,env.step_dt);planner.reset(list(range(env.num_envs)),env)
    buffers=[defaultdict(list) for _ in range(env.num_envs)]
    count=np.zeros(env.num_envs,dtype=int);quota=np.full(env.num_envs,a.episodes//env.num_envs);quota[:a.episodes%env.num_envs]+=1
    peak=torch.zeros(env.num_envs,device=env.device);rows=[];saved=ticks=0;started=time.monotonic()
    h=h5py.File(out/'trajectories.h5','x') if a.mode=='collect' else None
    if h is not None:h.attrs['source']='ordinary initial-state expert with same Phase14 recovery planner'
    try:
        with (out/'attempts.csv').open('x',newline='') as f:
            fields=['clone','episode','success','length','max_angle','final_angle','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            while True:
                robot=robot_observation(env).clone();gt=env.get_tool_state().clone();context=pack(env.get_environment_state()).clone()
                age=env.episode_length_buf.clone();params=env.parameters.clone()
                take=(age>=16)&~planner.started
                if take.any():planner.takeover(take.nonzero().flatten(),env)
                action=planner.action(env);action[~planner.started,:6]=0;action[~planner.started,6]=1
                phase=planner.state.clone()
                _,reward,term,trunc,_=env.step(action);nr,ng,done=transition_after_step(env,term,trunc);ne=next_environment(env,done)
                peak=torch.maximum(peak,ng[:,0])
                batch=dict(observation=robot,environment_state=context,state=gt,action=action,reward=reward[:,None],next_observation=nr,
                    next_environment_state=ne,next_state=ng,done=done[:,None],terminated=term[:,None],truncated=trunc[:,None],
                    reset_parameters=params,episode_tick=age[:,None],planner_phase=phase[:,None])
                arrays={k:v.cpu().numpy() for k,v in batch.items()}
                for i in range(env.num_envs):
                    if a.mode=='validate' and count[i]>=quota[i]:continue
                    for k,v in arrays.items():buffers[i][k].append(v[i].copy())
                ids=done.nonzero().flatten().tolist();ticks+=1
                for i in ids:
                    if a.mode=='validate' and count[i]>=quota[i]:continue
                    if a.mode=='collect' and (saved>=a.episodes or len(rows)>=a.max_attempts):continue
                    ok=bool(ng[i,0]>1);row=dict(clone=i,episode=int(count[i]),success=int(ok),length=len(buffers[i]['action']),
                        max_angle=float(peak[i]),final_angle=float(ng[i,0]),**dict(zip(fields[-5:],[float(v) for v in env.final_parameters[i]])))
                    rows.append(row);w.writerow(row);count[i]+=1
                    if h is not None and ok:
                        g=h.create_group(f'traj_{saved:05d}')
                        for k,v in buffers[i].items():g.create_dataset(k,data=np.asarray(v),compression='gzip',compression_opts=1)
                        g['robot_state']=g['observation'];g['expert_action']=g['action'];g.create_dataset('success',data=np.ones((row['length'],1),bool))
                        for k,v in row.items():g.attrs[k]=v
                        saved+=1
                    buffers[i]=defaultdict(list)
                if ids:
                    planner.reset(ids,env);peak[ids]=0;f.flush()
                    if h is not None:h.flush()
                if ticks%200==0:print(json.dumps(dict(ticks=ticks,attempted=len(rows),successes=sum(x['success'] for x in rows),saved=saved,elapsed_s=time.monotonic()-started)),flush=True)
                if a.mode=='validate' and np.all(count>=quota):break
                if a.mode=='collect' and (saved>=a.episodes or len(rows)>=a.max_attempts):break
            k=sum(x['success'] for x in rows);s=dict(mode=a.mode,seed=a.seed,episodes=len(rows),successes=k,success_rate=k/len(rows),
                success_95_wilson=wilson(k,len(rows)),saved_trajectories=saved,interactions=ticks*env.num_envs,
                elapsed_s=time.monotonic()-started,num_envs=env.num_envs,same_recovery_expert=True,qp=getattr(env,'phase14_qp',{}))
            (out/'summary.json').write_text(json.dumps(s,indent=2));print('COMPLETE '+json.dumps(s),flush=True)
            if a.mode=='collect' and saved<a.episodes:raise RuntimeError('Ordinary success quota not met')
    finally:
        if h is not None:h.close()
        env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
