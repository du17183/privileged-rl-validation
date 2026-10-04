"""Three-arm, 50,016-interaction controlled continuation, gate enforced."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','B','C'],required=True);p.add_argument('--seed',type=int,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2]
gate=json.loads((ROOT/'results/phase13_random_expert_bc/anchor_validation.json').read_text())
comparison=json.loads((ROOT/'results/phase13_random_expert_bc/bc_primary_summary.json').read_text())
expert_gate=json.loads((ROOT/'results/phase13_random_expert_bc/expert_validation/summary.json').read_text())
if not (expert_gate['success_rate']>.9 and comparison['paired']['success']['ci95'][0]>0
        and comparison['paired']['success']['positive_seeds']>=4
        and gate['deterministic_summary']['ci95'][0]>.25
        and gate['stochastic_summary']['ci95'][0]>.25):
    raise RuntimeError('User-defined relative BC benefit and independent candidate generalization prerequisites not passed')
app=AppLauncher(headless=True).app
import csv
import hashlib
import subprocess
import sys
import time
import h5py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import isaaclab_tasks  # noqa: F401
from randomized_env.door_randomization import create
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from experiments.phase13_random_expert_bc.state import pack,next_environment
from experiments.phase13_random_expert_bc.model import inputs
from experiments.phase13_random_expert_bc.rl_agent import Agent,Buffer,mix


def append(path,row):
    exists=path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row)


def main():
    torch.set_num_threads(4);torch.manual_seed(14300+a.seed);np.random.seed(14300+a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    name=f'{a.arm}_seed{a.seed}';out=ROOT/'results/phase13_random_expert_bc/rl'/name
    check=ROOT/'checkpoints/phase13_random_expert_bc/rl'/name
    out.mkdir(parents=True,exist_ok=False);check.mkdir(parents=True,exist_ok=False)
    source=ROOT/gate.get('anchor',gate['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=gate['source_sha256']:raise RuntimeError('Anchor hash mismatch')
    value=torch.load(source,map_location=a.device or 'cuda:0',weights_only=False)
    env=create(32,a.device or 'cuda:0',14301+a.seed,level=2)
    mean=torch.tensor(value['mean'],device=env.device);std=torch.tensor(value['std'],device=env.device)
    agent=Agent(value,env.device,.1 if a.arm=='C' else 0.)
    manifest=json.loads((ROOT/'datasets/random_door_expert/split_v1/split.json').read_text())
    rows={key:[] for key in ['observation','next_observation','action','reward','terminated']}
    with h5py.File(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5','r') as h:
        for key in manifest['splits']['train']:
            g=h[key]
            for field in ['observation','next_observation']:
                context='environment_state' if field=='observation' else 'next_environment_state'
                rows[field].append(inputs(torch.tensor(g[field][:],device=env.device),torch.tensor(g[context][:],device=env.device),mean,std,'B'))
            for field in ['action','reward','terminated']:rows[field].append(torch.tensor(g[field][:],dtype=torch.float32,device=env.device))
    expert=Buffer(sum(len(x) for x in rows['action']),env.device)
    expert.add({key:torch.cat(vals) for key,vals in rows.items()})
    online=Buffer(50016,env.device)
    if a.arm!='A':
        for step in range(1000):agent.critic_step(expert.sample(256))
    steps=updates=episodes=successes=0
    returns=torch.zeros(32,device=env.device);lengths=torch.zeros(32,device=env.device,dtype=torch.long)
    best=-1;best_step=0;curves=[];started=time.monotonic();losses=[]
    writer=SummaryWriter(str(ROOT/'logs/phase13_random_expert_bc/rl'/name))
    def save(path):
        torch.save(dict(model=agent.actor.state_dict(),mean=value['mean'],std=value['std'],arm='B',seed=a.seed,
                   update=updates,environment_steps=steps,dataset_sha256=manifest['dataset_sha256'],rl_arm=a.arm,
                   anchor_training_seed=gate['chosen']['seed'],source_sha256=gate['source_sha256'],
                   operational_anchor_gate_passed=gate['operational_anchor_gate_passed']),path)
    def evaluate():
        nonlocal best,best_step
        checkpoint=check/f'step_{steps}.pt';save(checkpoint)
        result=out/f'validation_{steps}.json'
        log=out/f'validation_{steps}.log'
        with log.open('x') as stream:
            run=subprocess.run([sys.executable,'-u','-m','experiments.phase13_random_expert_bc.evaluate',
                '--checkpoint',str(checkpoint),'--output',str(result),'--episodes','64','--num-envs','32',
                '--seed',str(14401+a.seed),'--device',str(env.device)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        if run.returncode:raise RuntimeError(f'Evaluation failed: {log}')
        metrics=json.loads(result.read_text());row=dict(environment_steps=steps,success=metrics['success'],
            contact_success=metrics['contact_success'],final_angle=metrics['mean_final_angle_rad'],
            online_successes=successes,online_episodes=episodes,elapsed_s=time.monotonic()-started)
        append(out/'curve.csv',row);curves.append(row)
        writer.add_scalar('validation/success',row['success'],steps)
        if row['success']>best:best=row['success'];best_step=steps;save(check/'best.pt')
        print('VALIDATION '+json.dumps(row),flush=True)
    try:
        evaluate()
        collect_generator=torch.Generator(device=env.device).manual_seed(14501+a.seed)
        torch.manual_seed(14601+a.seed)
        while steps<50016:
            robot=robot_observation(env).clone();context=pack(env.get_environment_state()).clone()
            observation=inputs(robot,context,mean,std,'B')
            with torch.no_grad():action=agent.actor(observation,stochastic=True,generator=collect_generator)
            _,reward,term,trunc,_=env.step(action)
            next_robot,next_gt,done=transition_after_step(env,term,trunc)
            next_context=next_environment(env,done)
            next_obs=inputs(next_robot,next_context,mean,std,'B')
            online.add(dict(observation=observation,next_observation=next_obs,action=action,
                            reward=reward.reshape(-1,1),terminated=term.float()[:,None]))
            steps+=32;returns+=reward.flatten();lengths+=1
            for i in done.nonzero().flatten().tolist():
                success=int(next_gt[i,0]>1);episodes+=1;successes+=success
                append(out/'episodes.csv',dict(environment_steps=steps,clone=i,success=success,length=int(lengths[i]),
                    return_value=float(returns[i]),final_angle=float(next_gt[i,0]),**{f'parameter_{j}':float(x) for j,x in enumerate(env.final_parameters[i])}))
                returns[i]=0;lengths[i]=0
            if a.arm!='A' and steps>=2048:
                for _ in range(4):losses.append(agent.update(mix(expert,online),expert.sample(256)));updates+=1
            if steps%1024==0:
                row=dict(environment_steps=steps,updates=updates,online_successes=successes,online_episodes=episodes,
                         elapsed_s=time.monotonic()-started)
                if losses:
                    row.update({key:float(np.mean([x[key] for x in losses])) for key in losses[0]});losses=[]
                append(out/'training.csv',row)
                for key,x in row.items():writer.add_scalar('training/'+key,x,steps)
                print(json.dumps(row),flush=True)
            if (steps>=10000 and (steps-32)//10000<steps//10000) or steps==50016:
                # Frozen baseline has no changing policy. Save cadence anyway;
                # its fixed-cohort success is measured initially and finally.
                if a.arm!='A' or steps==50016:evaluate()
                else:save(check/f'step_{steps}.pt')
        save(check/'final.pt')
        state=dict(arm=a.arm,seed=a.seed,interactions=steps,updates=updates,critic_warmup_updates=1000 if a.arm!='A' else 0,
                   curves=curves,best_validation=best,best_step=best_step,final_validation=curves[-1]['success'],
                   online_successes=successes,online_episodes=episodes,incomplete_episodes=int((lengths>0).sum()),
                   elapsed_s=time.monotonic()-started,anchor_training_seed=gate['chosen']['seed'],
                   controls=dict(actor_lr=3e-5,critic_lr=3e-4,std=.01,alpha=1e-4,bc_weight=10.,
                                 anchor_kl_weight=.1 if a.arm=='C' else 0.,replay_expert_fraction=.5,rollbacks=0))
        (out/'summary.json').write_text(json.dumps(state,indent=2));print('COMPLETE '+json.dumps(state),flush=True)
        torch.save(dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),target=agent.target.state_dict(),
                        actor_optimizer=agent.actor_opt.state_dict(),critic_optimizer=agent.critic_opt.state_dict()),check/'training_state.pt')
    finally:writer.close();env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
