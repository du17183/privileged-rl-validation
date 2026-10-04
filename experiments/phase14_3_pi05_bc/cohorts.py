"""Fresh common pressure states, generated without any recovery expert.

Store legal reset parameters and executed action histories, not learned labels.
Pressure tests replay histories on the same clone slot before policy takeover.
Only door-regression injection writes a door joint (recorded tick).
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--seed-base',type=int,required=True);p.add_argument('--cases',default='ee_offset,contact_loss,door_regression,stagnation,natural_severe');p.add_argument('--episodes',type=int,default=64)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(headless=True).app
import json,time,os
from pathlib import Path
import h5py,numpy as np,torch,isaaclab_tasks
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from experiments.phase13_random_expert_bc.state import pack
from experiments.phase13_random_expert_bc.model import load_checkpoint,inputs
from recovery_expert.perturbation_generator import PerturbationGenerator

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    env=create(32,a.device or 'cuda:0',a.seed_base,level=2);net,mean,std,meta=load_checkpoint(a.checkpoint,env.device)
    try:
        for case in a.cases.split(','):
            file=out/(case+'.h5')
            if file.exists():continue
            index=['ee_offset','contact_loss','door_regression','stagnation','natural_severe'].index(case);seed=a.seed_base+index
            env.reset(seed=seed);torch.manual_seed(seed)
            gen=PerturbationGenerator(32,env.device,case,seed) if case!='natural_severe' else None
            if gen:gen.reset(list(range(32)),list(range(32)))
            actions=[[] for _ in range(32)];written=np.full(32,-1);captured=np.zeros(32,bool);episodes=np.zeros(32,int)
            saved=attempts=invalid=ticks=0;started=time.monotonic();clones=[]
            with h5py.File(file,'x') as h:
                h.attrs.update(seed=seed,reference_checkpoint=a.checkpoint,expert_actions=0,training_labels=False)
                while saved<a.episodes:
                    robot=robot_observation(env).clone();gt=env.get_tool_state().clone();age=env.episode_length_buf.clone()
                    with torch.no_grad():base=net(inputs(robot,pack(env.get_environment_state()),mean,std,meta['arm']))
                    action=base.clone();finished=torch.zeros(32,device=env.device,dtype=torch.bool);valid=finished.clone()
                    if gen:
                        before=gen.explicit_door_write.clone()
                        action,finished,valid=gen.action(env,robot,gt,base)
                        fresh=gen.explicit_door_write&~before
                        for i in fresh.nonzero().flatten().tolist():written[i]=int(age[i])
                        if fresh.any():gt=env.get_tool_state().clone()
                    else:
                        distance=(robot[:,18:21]-gt[:,2:5]).norm(dim=-1)
                        finished=(age==169)&~torch.as_tensor(captured,device=env.device)
                        valid=finished&(distance>=.10)&(distance<=.20)&(gt[:,0]<1.)
                    for i in (finished&valid).nonzero().flatten().tolist():
                        if captured[i] or saved>=a.episodes:continue
                        g=h.create_group(f'state_{saved:04d}')
                        g.create_dataset('prefix_actions',data=np.asarray(actions[i],np.float32),compression='gzip')
                        g.create_dataset('reset_parameters',data=env.parameters[i].cpu().numpy())
                        g.create_dataset('handoff_robot',data=robot[i].cpu().numpy());g.create_dataset('handoff_state',data=gt[i].cpu().numpy())
                        g.attrs.update(clone=i,source_episode=int(episodes[i]),handoff_tick=int(age[i]),door_write_tick=int(written[i]),
                                       distance=float((robot[i,18:21]-gt[i,2:5]).norm()),case=case,
                                       assignment=int(gen.assignment[i]) if gen else -1)
                        clones.append(i);saved+=1;captured[i]=True;h.flush()
                    for i in range(32):
                        if not captured[i]:actions[i].append(action[i].cpu().numpy().copy())
                    _,_,term,trunc,_=env.step(action);done=term|trunc
                    ids=done.nonzero().flatten().tolist()
                    for i in ids:
                        attempts+=1
                        if not captured[i]:invalid+=1
                        actions[i]=[];written[i]=-1;captured[i]=False;episodes[i]+=1
                    if gen and ids:gen.reset(ids,[i+int(episodes[i])*32 for i in ids])
                    ticks+=1
                    if ticks%200==0:print(json.dumps(dict(case=case,saved=saved,attempts=attempts,ticks=ticks,elapsed_s=time.monotonic()-started)),flush=True)
                    if attempts>=2400:raise RuntimeError('Pressure cohort budget exhausted')
            summary=dict(case=case,seed=seed,saved=saved,completed_screening_episodes=attempts,not_selected_completed=invalid,
                         clones=clones,interactions=ticks*32,elapsed_s=time.monotonic()-started,reference_only=True,natural_data_not_trained=True)
            (out/(case+'.json')).write_text(json.dumps(summary,indent=2));print('COMPLETE '+json.dumps(summary),flush=True)
    finally:env.close()

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc();os._exit(1)
    os._exit(0)
