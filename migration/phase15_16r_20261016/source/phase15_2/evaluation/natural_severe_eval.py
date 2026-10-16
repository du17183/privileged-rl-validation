"""Independent evaluation-only cohorts from frozen Normal + Recovery rollouts.

Natural snapshots are selected only after that source episode has actually
failed. Graded deviations use recorded legal Cartesian perturbation actions.
No cohort is used as a training dataset. No extra episode time is supplied.
"""
import argparse
import os
import traceback
from pathlib import Path
import torch
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--socket',required=True)
p.add_argument('--model-seed',type=int,required=True);p.add_argument('--reset-seed',type=int,required=True)
p.add_argument('--cases',default='natural_severe');p.add_argument('--episodes',type=int,default=32)
p.add_argument('--natural-episodes',type=int,default=32)
p.add_argument('--resume-from')
p.add_argument('--max-screening-episodes',type=int,default=2048)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app
import time
import h5py
import numpy as np
import torch
import isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase14_2_recovery_bc.model import measured,inputs
from recovery_expert.perturbation_generator import PerturbationGenerator
from phase15_2.common import ROOT,RESULTS,frozen,save_json,sha256,isolated,redirect_calibration
from phase15_1.policy_client import SystemPolicy
from phase15_1.evaluation.cohort_resume import append_previous


def sample(env,i,actions,robot,gt,episode,case,written=-1,assignment=-1):
    return dict(actions=np.asarray(actions,np.float32).reshape(-1,7),params=env.parameters[i].cpu().numpy().copy(),
        robot=robot[i].cpu().numpy().copy(),gt=gt[i].cpu().numpy().copy(),clone=int(i),episode=int(episode),
        case=case,distance=float((robot[i,18:21]-gt[i,2:5]).norm()),written=written,assignment=assignment)


def store(h,record,number,source_success=None):
    g=h.create_group(f'state_{number:04d}')
    for key,value in [('prefix_actions',record['actions']),('reset_parameters',record['params']),
                      ('handoff_robot',record['robot']),('handoff_state',record['gt'])]:
        g.create_dataset(key,data=value,compression='gzip' if key=='prefix_actions' else None)
    g.attrs.update(clone=record['clone'],source_episode=record['episode'],handoff_tick=len(record['actions']),
        door_write_tick=record['written'],assignment=record['assignment'],case=record['case'],distance=record['distance'],
        natural_source_episode_failed=source_success is False,reference_model_seed=a.model_seed,
        reset_seed=a.reset_seed,training_labels=False)
    h.flush()


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=isolated(a.output);out.mkdir(parents=True,exist_ok=True)
    env=create(32,a.device or 'cuda:0',a.reset_seed,level=2)
    redirect_calibration()
    ck=frozen(a.model_seed);job=dict(controller='B',schedule=dict(normal=15,recovery=15),horizons=dict(normal=15,recovery=15))
    client=SystemPolicy(a.socket,env,job);meta=client.load(ck,'D')['normal']
    client.seed=a.reset_seed+300
    mean,std=[torch.tensor(meta[k],device=env.device) for k in ['mean','std']]
    original=env.step
    def observed(action):
        result=original(action);r,g,d=transition_after_step(env,result[2],result[3]);client.observe(r,g,d);return result
    env.step=observed
    try:
        for case_index,case in enumerate(a.cases.split(',')):
            file=out/(case+'.h5');summary_file=out/(case+'.json')
            if summary_file.exists():continue
            if file.exists():raise RuntimeError('Incomplete cohort retained; use a new collection revision')
            client.configure();env.reset(seed=a.reset_seed+case_index*37)
            histories=[[] for _ in range(32)];pending=[None]*32;episodes=np.zeros(32,int)
            written=np.full(32,-1,int);captured=np.zeros(32,bool);perturb_start=np.full(32,-1,int)
            perturb_direction=np.zeros((32,3));saved=completed=failed=source_severe=0;ticks=0;started=time.monotonic()
            rng=np.random.default_rng(a.reset_seed+901);eligible_counts=np.zeros(32,int)
            natural=case=='natural_severe';graded=case.startswith('grade_')
            target_n=a.natural_episodes if natural else a.episodes
            generator=None
            if not natural and not graded:
                generator=PerturbationGenerator(32,env.device,case,a.reset_seed+case_index)
                generator.reset(list(range(32)),list(range(32)))
            with h5py.File(file,'x') as h:
                h.attrs.update(phase15_2=True,evaluation_only=True,source='frozen Normal+Recovery',
                    seed=a.reset_seed,model_seed=a.model_seed,checkpoints=str(ck),rl_updates=0,task_unchanged=True)
                resume=append_previous(h,Path(a.resume_from)/(case+'.h5'),case,target_n) if a.resume_from else dict(imported=0)
                saved=resume['imported']
                while saved<target_n:
                    robot=robot_observation(env).clone();gt=env.get_tool_state().clone();age=env.episode_length_buf.clone()
                    with torch.no_grad():base=client(inputs(robot,measured(env),mean,std,meta['mode']))
                    action=base.clone();distance=(robot[:,18:21]-gt[:,2:5]).norm(dim=-1)
                    if natural:
                        candidate=(age>=120)&(age<=420)&(distance>=.10)&(distance<=.24)&(gt[:,0]<1.)
                        for i in candidate.nonzero().flatten().tolist():
                            eligible_counts[i]+=1
                            if rng.integers(eligible_counts[i])==0:
                                pending[i]=sample(env,i,histories[i],robot,gt,episodes[i],case)
                    elif graded:
                        target=float(case.split('_')[1])/100
                        # Different contact histories: start just before contact or after grasp/progress.
                        desired_start=100+(episodes*17+np.arange(32)*23)%130
                        for i in range(32):
                            if captured[i]:continue
                            if perturb_start[i]<0 and float(distance[i])<.035 and int(age[i])>=desired_start[i] and float(gt[i,0])<.8:
                                perturb_start[i]=int(age[i]);angle=float(gt[i,0])
                                direction=np.array([-np.cos(angle),-np.sin(angle),.2*(-1 if i%2 else 1)])
                                perturb_direction[i]=direction/np.linalg.norm(direction)
                            if perturb_start[i]>=0:
                                action[i]=0;action[i,6]=1
                                if int(age[i])-perturb_start[i]>=12:
                                    action[i,:3]=torch.tensor(perturb_direction[i]*.07,device=env.device)
                                    action[i,3:6]=torch.tensor([0.,.012*(-1 if i%2 else 1),0.],device=env.device)
                                if target-.007<=float(distance[i])<=target+.015 and len(histories[i])<460:
                                    record=sample(env,i,histories[i],robot,gt,episodes[i],case)
                                    if saved<target_n:store(h,record,saved);saved+=1;captured[i]=True
                    else:
                        before=generator.explicit_door_write.clone();action,finished,valid=generator.action(env,robot,gt,base)
                        fresh=generator.explicit_door_write&~before
                        for i in fresh.nonzero().flatten().tolist():written[i]=int(age[i])
                        if fresh.any():gt=env.get_tool_state().clone()
                        for i in (finished&valid).nonzero().flatten().tolist():
                            if not captured[i] and saved<target_n:
                                store(h,sample(env,i,histories[i],robot,gt,episodes[i],case,int(written[i]),int(generator.assignment[i])),saved)
                                saved+=1;captured[i]=True
                    for i in range(32):histories[i].append(action[i].cpu().numpy().copy())
                    result=env.step(action);nr,ng,done=transition_after_step(env,result[2],result[3])
                    ids=done.nonzero().flatten().tolist()
                    for i in ids:
                        completed+=1;ok=bool(ng[i,0]>1.);failed+=int(not ok)
                        if natural and pending[i] is not None:
                            source_severe+=1
                            if not ok and saved<target_n:store(h,pending[i],saved,source_success=False);saved+=1
                        histories[i]=[];pending[i]=None;eligible_counts[i]=0;written[i]=-1;captured[i]=False;episodes[i]+=1;perturb_start[i]=-1
                    if generator and ids:generator.reset(ids,[i+int(episodes[i])*32 for i in ids])
                    ticks+=1
                    if ticks%100==0:
                        save_json(out/(case+'.progress.json'),dict(saved=saved,target=target_n,completed=completed,failed=failed,
                            ticks=ticks,elapsed_s=time.monotonic()-started,rl_updates=0))
                        print('PROGRESS',case,saved,completed,flush=True)
                    if completed>=a.max_screening_episodes and saved<target_n:
                        raise RuntimeError(f'Independent cohort quota not achieved: {case} {saved}/{target_n}; retain all negative evidence')
            save_json(summary_file,dict(case=case,saved=saved,completed_source_episodes=completed,source_failed_episodes=failed,
                resumed_evidence=resume,new_source_episode_counts_exclude_resumed_examples=True,
                source_episodes_with_severe_candidate=source_severe,source_checkpoints=ck,
                source_sha256={k:sha256(v) for k,v in ck.items()},cohort_sha256=sha256(file),
                reset_seed=a.reset_seed,model_seed=a.model_seed,selected_only_after_source_failure=natural, natural_snapshot_rule='uniform reservoir over physical 10-24cm candidates, age120-420; source episode must fail', frozen_source_controller='Phase15.1 E_final B/15/15/graph',
                evaluation_only=True,training_labels=False,interactions=ticks*32,rl_updates=0))
    finally:client.stop();env.step=original;env.close()
    save_json(out/'complete.json',dict(complete=True,rl_updates=0))


if __name__=='__main__':
    try:main()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        save_json(Path(a.output)/'failed.json',dict(error=error));os._exit(1)
    os._exit(0)
