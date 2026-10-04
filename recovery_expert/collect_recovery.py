"""Phase14.1: current policy -> measured disturbance -> GT expert labels."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--case',choices=['ee_offset','contact_loss','door_regression','stagnation'],required=True)
p.add_argument('--episodes',type=int,required=True);p.add_argument('--seed',type=int,required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--max-attempts',type=int,default=1200)
p.add_argument('--expert',choices=['local','historical'],default='local')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app

import csv,json,time,hashlib
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
from recovery_expert.perturbation_generator import PerturbationGenerator
from recovery_expert.failure_classifier import classify
if a.expert=='local':from recovery_expert.phase14_1.recovery_planner import RecoveryPlanner
else:from recovery_expert.recovery_planner import RecoveryPlanner


def main():
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'attempts.csv').exists():raise FileExistsError(out)
    env=create(a.num_envs,a.device or 'cuda:0',a.seed,level=2)
    policy,mean,std,checkpoint=load_checkpoint(a.checkpoint,env.device)
    planner=RecoveryPlanner(env.num_envs,env.device,env.step_dt);planner.reset(list(range(env.num_envs)),env)
    gen=PerturbationGenerator(env.num_envs,env.device,a.case,a.seed)
    quota=np.full(env.num_envs,a.episodes//env.num_envs);quota[:a.episodes%env.num_envs]+=1
    count=np.zeros(env.num_envs,dtype=int);attempt=np.zeros_like(count)
    assigned=np.arange(env.num_envs,dtype=int)
    gen.reset(list(range(env.num_envs)),assigned)
    traces=[defaultdict(list) for _ in count];prefix=[defaultdict(list) for _ in count]
    maximum=torch.zeros(env.num_envs,device=env.device)
    handoff=np.full(env.num_envs,-1);starts={};rows=[];ticks=0;begin=time.monotonic()
    success_file=h5py.File(out/'trajectories.h5','x');failed_file=h5py.File(out/'failed_trajectories.h5','x')
    for file in (success_file,failed_file):
        file.attrs.update(case=a.case,seed=a.seed,prefix_checkpoint=a.checkpoint,
                          robot_teleports=0,no_reset_at_takeover=True,
                          door_regression_injection=a.case=='door_regression')
    fields=['clone','episode','assignment','case','axis','target_mm','actual_dx','actual_dy','actual_dz',
            'perturbation_started','valid_perturbation','success','handoff_tick','handoff_angle',
            'recovery_ticks','max_angle','final_angle','failure_code','failure_label',
            'door_state_writes','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
    try:
        with (out/'attempts.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            while np.any(count<quota):
                robot=robot_observation(env).clone();gt=env.get_tool_state().clone()
                context=pack(env.get_environment_state()).clone();age=env.episode_length_buf.clone()
                params=env.parameters.clone();stage_before=gen.stage.clone()
                gen.enabled=torch.as_tensor(count<quota,device=env.device)
                with torch.no_grad():base_action=policy(inputs(robot,context,mean,std,checkpoint['arm']))
                action,finished,valid=gen.action(env,robot,gt,base_action)
                # Door injection deliberately changes physical state, before
                # observation/action labels are recorded; no hidden teleport in recovery.
                if a.case=='door_regression' and (gen.explicit_door_write&(stage_before==0)).any():
                    gt=env.get_tool_state().clone();context=pack(env.get_environment_state()).clone()
                for i in finished.nonzero().flatten().tolist():
                    if count[i]>=quota[i]:continue
                    before_q=env.scene['robot'].data.joint_pos[i].clone();before_angle=gt[i,0].clone()
                    planner.takeover([i],env)
                    assert torch.equal(age[i],env.episode_length_buf[i])
                    assert torch.equal(before_q,env.scene['robot'].data.joint_pos[i])
                    assert torch.equal(before_angle,env.get_tool_state()[i,0])
                    handoff[i]=int(age[i]);starts[i]=dict(angle=float(gt[i,0]),actual=gen.actual_offset[i].cpu().tolist())
                active=gen.stage==gen.RECOVERY;phase=planner.state.clone()
                expert_action=planner.action(env);action[active]=expert_action[active]
                ee=robot[:,18:21];handle=gt[:,2:5]
                margin=torch.minimum(env.scene['robot'].data.joint_pos[:,:7]-env.scene['robot'].data.soft_joint_pos_limits[:,:7,0],
                                     env.scene['robot'].data.soft_joint_pos_limits[:,:7,1]-env.scene['robot'].data.joint_pos[:,:7]).amin(-1)
                poserr=getattr(planner,'position_error',(ee-handle).norm(dim=-1)).clone()
                roterr=getattr(planner,'rotation_error',torch.zeros_like(poserr)).clone()
                _,reward,term,trunc,_=env.step(action)
                nr,ng,done=transition_after_step(env,term,trunc);ne=next_environment(env,done)
                maximum=torch.maximum(maximum,ng[:,0])
                batch=dict(robot_state=robot,environment_state=context,state=gt,action=action,reward=reward[:,None],
                           next_robot_state=nr,next_environment_state=ne,next_state=ng,done=done[:,None],
                           terminated=term[:,None],truncated=trunc[:,None],door_angle=gt[:,0:1],
                           contact_state=(gt[:,9:11]>.5),episode_tick=age[:,None],planner_phase=phase[:,None],
                           position_error=poserr[:,None],rotation_error=roterr[:,None],joint_margin=margin[:,None],
                           reset_parameters=params,source_stage=gen.stage[:,None])
                arrays={k:v.cpu().numpy() for k,v in batch.items()}
                for i in range(env.num_envs):
                    if count[i]>=quota[i]:continue
                    dest=traces[i] if bool(active[i]) else prefix[i]
                    for k,v in arrays.items():dest[k].append(v[i].copy())
                ids=done.nonzero().flatten().tolist()
                for i in ids:
                    if count[i]>=quota[i]:continue
                    ok=bool(ng[i,0]>1);isvalid=bool(gen.valid[i]);injected=handoff[i]>=0
                    classification=classify(traces[i],ok) if isvalid else dict(code=-1,label='invalid_or_unreached_perturbation',confidence='observed')
                    actual=starts.get(i,{}).get('actual',[0.,0.,0.])
                    row=dict(clone=i,episode=int(attempt[i]),assignment=int(assigned[i]),case=a.case,axis=int(gen.axis[i]),
                             target_mm=float(gen.target_offset[i,gen.axis[i]])*1000,
                             actual_dx=actual[0],actual_dy=actual[1],actual_dz=actual[2],
                             perturbation_started=int(injected),valid_perturbation=int(isvalid),success=int(ok),
                             handoff_tick=int(handoff[i]),handoff_angle=starts.get(i,{}).get('angle',0.),
                             recovery_ticks=len(traces[i]['action']),max_angle=float(maximum[i]),final_angle=float(ng[i,0]),
                             failure_code=classification['code'],failure_label=classification['label'],
                             door_state_writes=int(gen.explicit_door_write[i]),
                             **dict(zip(fields[-5:],[float(v) for v in env.final_parameters[i]])))
                    writer.writerow(row);rows.append(row);attempt[i]+=1
                    # Store every completed attempt, including invalid/unreached states.
                    dest=success_file if isvalid and ok else failed_file
                    g=dest.create_group(f'attempt_{len(rows):05d}')
                    for k,v in traces[i].items():
                        if v:g.create_dataset(k,data=np.asarray(v),compression='gzip',compression_opts=1)
                    if traces[i].get('action'):
                        g['observation']=g['robot_state'];g['expert_action']=g['action']
                        g.create_dataset('success',data=np.full((len(traces[i]['action']),1),isvalid and ok))
                    pg=g.create_group('policy_and_injection_prefix')
                    for k,v in prefix[i].items():
                        if v:pg.create_dataset(k,data=np.asarray(v),compression='gzip',compression_opts=1)
                    for k,v in row.items():g.attrs[k]=v
                    g.attrs['failure_analysis']=json.dumps(classification,ensure_ascii=False)
                    if isvalid:count[i]+=1
                    if count[i]<quota[i]:
                        if isvalid:assigned[i]=i+int(count[i])*env.num_envs
                    traces[i]=defaultdict(list);prefix[i]=defaultdict(list);starts.pop(i,None);handoff[i]=-1
                if ids:
                    planner.reset(ids,env);gen.reset(ids,[assigned[i] for i in ids]);maximum[ids]=0
                    f.flush();success_file.flush();failed_file.flush()
                ticks+=1
                if ticks%200==0 or ids:
                    valid_rows=[r for r in rows if r['valid_perturbation']]
                    progress=dict(case=a.case,seed=a.seed,valid_attempts=len(valid_rows),requested=a.episodes,
                                  successes=sum(r['success'] for r in valid_rows),all_attempts=len(rows),
                                  ticks=ticks,interactions=ticks*env.num_envs,elapsed_s=time.monotonic()-begin)
                    (out/'progress.json').write_text(json.dumps(progress,indent=2))
                    if ticks%200==0:print(json.dumps(progress),flush=True)
                if len(rows)>=a.max_attempts:raise RuntimeError('Perturbation generation budget exhausted; all attempts retained')
            valid_rows=[r for r in rows if r['valid_perturbation']];s=sum(r['success'] for r in valid_rows)
            buckets={}
            if a.case=='ee_offset':
                for axis in range(3):
                    for mm in [5,10,20]:
                        for sign in [-1,1]:
                            group=[r for r in valid_rows if r['axis']==axis and abs(r['target_mm']-sign*mm)<.01]
                            buckets[f'{"xyz"[axis]}_{sign*mm:+d}mm']=dict(n=len(group),success=sum(r['success'] for r in group))
            summary=dict(case=a.case,seed=a.seed,expert=a.expert,requested=a.episodes,valid_attempts=len(valid_rows),
                         success_count=s,success_rate=s/max(1,len(valid_rows)),wilson95=wilson(s,len(valid_rows)),
                         all_attempts=len(rows),invalid_attempts=len(rows)-len(valid_rows),buckets=buckets,
                         interactions=ticks*env.num_envs,elapsed_s=time.monotonic()-begin,
                         no_robot_teleport=True,no_reset_at_takeover=True,door_state_writes=sum(r['door_state_writes'] for r in rows),
                         reward_changed=False,bc_training=False,rl_training=False)
            (out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE '+json.dumps(summary),flush=True)
    finally:
        success_file.close();failed_file.close();env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
