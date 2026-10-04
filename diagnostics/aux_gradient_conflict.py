"""Policy/BC versus GT-supervision gradients on a fixed expert probe.

This is a local first-order diagnostic at saved checkpoints. It cannot by
itself establish long-horizon causal interference during online learning.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from auxiliary_learning.gt_prediction import AuxiliarySAC
from diagnostics.q_value_analysis import fixed_probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'door_privileged_ablation'


def flattened(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g,p in zip(grads,params)])


def probe(agent,batch):
    actor=agent.actor
    params=list(actor.encoder.parameters())
    robot,gt,expert=batch['robot'],batch['privileged'],batch['action']
    for p in agent.critic.parameters(): p.requires_grad_(False)
    torch.manual_seed(20260929)
    action,logp=actor(robot)
    q1,q2=agent.critic(robot,action)
    bc_action,_=actor(robot,deterministic=True)
    policy_loss=(agent.log_alpha.exp().detach()*logp-torch.minimum(q1,q2)).mean()+\
                10*F.mse_loss(bc_action,expert)
    policy_grad=flattened(torch.autograd.grad(policy_loss,params),params)
    pred=actor.predict_gt(robot)
    aux_loss=F.mse_loss(pred[:,:1],gt[:,:1])+\
             F.binary_cross_entropy_with_logits(pred[:,1:],gt[:,9:11])
    aux_grad=flattened(torch.autograd.grad(aux_loss,params),params)
    cosine=float(F.cosine_similarity(policy_grad[None],aux_grad[None]).item())
    pnorm=float(policy_grad.norm())
    anorm=float(aux_grad.norm())
    return {'policy_bc_loss':float(policy_loss.detach()),
            'aux_loss':float(aux_loss.detach()),
            'encoder_policy_grad_l2':pnorm,'encoder_aux_grad_l2':anorm,
            'aux_to_policy_grad_ratio_at_weight_0_1':.1*anorm/max(pnorm,1e-12),
            'grad_cosine':cosine}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',nargs='+',default=['E0','E1','E2','E3'])
    parser.add_argument('--output',type=Path,default=OUT/'aux_gradient_conflict.csv')
    args=parser.parse_args()
    torch.set_num_threads(4)
    batch=fixed_probe(ROOT/'door_dataset'/'door_expert_1000.h5')
    rows=[]
    for variant in args.variants:
        for seed in range(5):
            for step in (100000,500000):
                path=ROOT/'checkpoints'/'door_privileged_ablation'/f'{variant}_seed{seed}'/f'step_{step}.pt'
                state=torch.load(path,map_location='cpu',weights_only=False)
                agent=AuxiliarySAC(26,11,7,device='cpu',aux_weight=.1)
                agent.actor.load_state_dict(state['actor'])
                agent.critic.load_state_dict(state['critic'])
                agent.log_alpha.data.copy_(state['log_alpha'])
                rows.append({'variant':variant,'seed':seed,'env_steps':step,
                             **probe(agent,batch)})
        print('gradient probe',variant,flush=True)
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
