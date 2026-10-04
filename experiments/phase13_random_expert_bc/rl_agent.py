"""Bounded symmetric SAC from the validated randomized BC policy.

Actor and critic receive exactly the same 39 measured input channels.
Exploration std is fixed .01; no asymmetric privileged critic or quality replay.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase13_random_expert_bc.model import Policy


class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        def network():return nn.Sequential(nn.Linear(46,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q1=network();self.q2=network()
    def forward(self,x,a):
        value=torch.cat((x,a),-1);return self.q1(value),self.q2(value)


class Agent:
    def __init__(self,checkpoint,device,kl_weight):
        self.actor=Policy().to(device);self.actor.load_state_dict(checkpoint['model'])
        self.anchor=copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.critic=TwinQ().to(device);self.target=copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=3e-5)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=3e-4)
        self.kl_weight=kl_weight
        self.alpha=1e-4

    def sample(self,x):
        mean=self.actor.mean(x);sigma=self.actor.log_std.exp().clamp_max(.01)
        z=mean+sigma*torch.randn_like(mean);action=z.tanh()
        logp=(-.5*((z-mean)/sigma)**2-self.actor.log_std-.5*math.log(2*math.pi)).sum(-1,keepdim=True)
        logp-=torch.log(1-action.square()+1e-6).sum(-1,keepdim=True)
        return action,logp

    def critic_step(self,batch):
        with torch.no_grad():
            action,logp=self.sample(batch['next_observation'])
            q1,q2=self.target(batch['next_observation'],action)
            target=batch['reward']+.99*(1-batch['terminated'])*(torch.minimum(q1,q2)-self.alpha*logp)
        q1,q2=self.critic(batch['observation'],batch['action'])
        loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.critic_opt.zero_grad(set_to_none=True);loss.backward();self.critic_opt.step()
        with torch.no_grad():
            for to,source in zip(self.target.parameters(),self.critic.parameters()):to.lerp_(source,.005)
        return dict(critic_loss=float(loss.detach()),q_mean=float(torch.minimum(q1,q2).detach().mean()),
                    q_std=float(torch.minimum(q1,q2).detach().std()),target_q_mean=float(target.mean()))

    def update(self,batch,expert):
        metrics=self.critic_step(batch)
        self.critic.requires_grad_(False)
        action,logp=self.sample(batch['observation'])
        q1,q2=self.critic(batch['observation'],action)
        rl=(self.alpha*logp-torch.minimum(q1,q2)).mean()
        bc=F.mse_loss(self.actor(expert['observation']),expert['action'])
        with torch.no_grad():anchor_mean=self.anchor.mean(batch['observation'])
        mean=self.actor.mean(batch['observation']);sigma=self.actor.log_std.exp().clamp_max(.01)
        kl=(.5*((mean-anchor_mean)/sigma)**2).sum(-1).mean()
        loss=rl+10*bc+self.kl_weight*kl
        self.actor_opt.zero_grad(set_to_none=True);loss.backward();self.actor_opt.step()
        self.critic.requires_grad_(True)
        metrics.update(actor_loss=float(loss.detach()),rl_loss=float(rl.detach()),bc_loss=float(bc.detach()),
                       anchor_kl=float(kl.detach()),entropy=float(-logp.detach().mean()),action_std=float(sigma.mean()))
        return metrics


class Buffer:
    def __init__(self,capacity,device):
        self.values={key:torch.zeros((capacity,dim),device=device) for key,dim in
                     [('observation',39),('next_observation',39),('action',7),('reward',1),('terminated',1)]}
        self.size=0;self.capacity=capacity
    def add(self,values):
        n=len(values['observation'])
        if self.size+n>self.capacity:raise RuntimeError('Bounded replay budget exceeded')
        for key in self.values:self.values[key][self.size:self.size+n]=values[key]
        self.size+=n
    def sample(self,n):
        ids=torch.randint(self.size,(n,),device=self.values['action'].device)
        return {key:value[ids] for key,value in self.values.items()}


def mix(expert,online,n=256):
    left=expert.sample(n//2);right=online.sample(n-n//2)
    return {key:torch.cat((left[key],right[key])) for key in left}
