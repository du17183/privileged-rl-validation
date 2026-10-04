"""Fixed-expert-state Q audit for robot-only shared-encoder critics."""

import csv
from pathlib import Path

import h5py
import numpy as np
import torch

from auxiliary_learning.multitask_encoder import SharedEncoderSAC

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stable_privileged_rl'
STEPS=(0,100000,200000,300000,400000,500000)
VARIANTS=('Q0','QGT','QV')


def fixed_probe(n=256):
    arrays={key:[] for key in ('observation','action','reward','next_observation','done')}
    rng=np.random.default_rng(20260929)
    with h5py.File(ROOT/'door_dataset'/'door_expert_1000.h5','r') as h5:
        names=sorted(name for name in h5 if name.startswith('traj_'))
        for selected in rng.choice(len(names),size=n,replace=False):
            name=names[int(selected)]
            group=h5[name]
            index=int(rng.integers(len(group['action'])))
            for key in arrays:
                arrays[key].append(group[key][index:index+1])
    return {key:torch.tensor(np.concatenate(value),dtype=torch.float32)
            for key,value in arrays.items()}


def one(variant,seed,step,probe):
    path=ROOT/'checkpoints'/'stable_privileged_rl'/f'{variant}_seed{seed}'/f'step_{step}.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    agent=SharedEncoderSAC(26,11,7,device='cpu')
    agent.actor.load_state_dict(state['actor'])
    agent.critic.load_state_dict(state['critic'])
    agent.target_critic.load_state_dict(state['target_critic'])
    agent.target_encoder.load_state_dict(state['target_encoder'])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'])
        robot,expert_action=probe['observation'],probe['action']
        z=agent.actor.encoder(robot)
        q1,q2=agent.critic(z,expert_action)
        q=torch.minimum(q1,q2)
        actor_action,_=agent.actor(robot,deterministic=True)
        actor_q=torch.minimum(*agent.critic(z,actor_action))
        next_action,_=agent.actor(probe['next_observation'],deterministic=True)
        next_z=agent.target_encoder(probe['next_observation'])
        next_q=torch.minimum(*agent.target_critic(next_z,next_action))
        target=probe['reward'].reshape(-1,1)+agent.cfg.gamma*(1-probe['done'].reshape(-1,1))*next_q
    action=actor_action.detach().requires_grad_(True)
    z=agent.actor.encoder(robot).detach()
    actor_q_for_grad=torch.minimum(*agent.critic(z,action))
    gradient=torch.autograd.grad(actor_q_for_grad.sum(),action)[0]
    expert_direction=expert_action-action.detach()
    cosine=torch.nn.functional.cosine_similarity(gradient,expert_direction,dim=-1)
    return {'variant':variant,'seed':seed,'env_steps':step,'probe_transitions':len(robot),
            'expert_q_mean':float(q.mean()),'expert_q_variance':float(q.var(unbiased=False)),
            'actor_minus_expert_q':float((actor_q-q).mean()),
            'one_step_deterministic_residual_abs':float((target-q).abs().mean()),
            'q_gradient_toward_expert_cosine':float(cosine.mean())}


def main():
    probe=fixed_probe()
    rows=[one(variant,seed,step,probe) for variant in VARIANTS
          for seed in range(5) for step in STEPS]
    path=OUT/'fixed_q_probe.csv'
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} fixed expert-state Q probes to {path}')


if __name__=='__main__': main()
