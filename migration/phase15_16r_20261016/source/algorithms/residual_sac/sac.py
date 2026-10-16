"""SAC over bounded residual coordinates, evaluating the real composition."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

CONTEXT_DIM=121

def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,outputs))

def compose(base, unit, bound):
    delta=unit*bound
    proposed=base[:,:6]+delta
    executed=torch.cat((proposed.clamp(-1,1),base[:,6:7]),-1)
    return delta,executed,proposed

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=mlp(CONTEXT_DIM,12)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad(): self.net[-1].bias[6:]=-4

    def forward(self, context, deterministic=False, noise=None):
        mean,log_std=self.net(context).chunk(2,-1)
        log_std=log_std.clamp(-8,0)
        z=mean if deterministic else mean+log_std.exp()*(torch.randn_like(mean) if noise is None else noise)
        unit=torch.tanh(z)
        # Stable exact tanh change of variables, in unit residual coordinates.
        logp=(-.5*((z-mean)/log_std.exp()).square()-log_std-.5*math.log(2*math.pi)
              -2*(math.log(2)-z-F.softplus(-2*z))).sum(-1,keepdim=True)
        return unit,logp,mean,log_std

class TwinQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1=mlp(CONTEXT_DIM+6+7,1)
        self.q2=mlp(CONTEXT_DIM+6+7,1)
    def forward(self, context,delta,executed):
        value=torch.cat((context,delta,executed),-1)
        return self.q1(value),self.q2(value)

class SAC:
    def __init__(self,seed,bound,device='cuda:0',cfg=None):
        torch.manual_seed(seed)
        self.device=device
        self.cfg=cfg or dict(gamma=.99,tau=.005,lr=.0003,initial_alpha=.0001,target_entropy=-6)
        self.bound=torch.tensor(bound,device=device,dtype=torch.float32)
        self.actor=Actor().to(device)
        self.q=TwinQ().to(device)
        self.target=copy.deepcopy(self.q).requires_grad_(False)
        self.log_alpha=torch.tensor(math.log(self.cfg['initial_alpha']),device=device,requires_grad=True)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.cfg['lr'])
        self.q_opt=torch.optim.Adam(self.q.parameters(),lr=self.cfg['lr'])
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=self.cfg['lr'])
        self.updates=0

    def policy(self,context,base,deterministic=False,noise=None):
        unit,logp,mean,logstd=self.actor(context,deterministic,noise)
        delta,executed,proposed=compose(base,unit,self.bound)
        return dict(unit=unit,logp=logp,delta=delta,executed=executed,proposed=proposed,mean=mean,logstd=logstd)

    @torch.no_grad()
    def target_value(self,b,noise=None):
        p=self.policy(b['next_context'],b['next_base'],noise=noise)
        a,c=self.target(b['next_context'],p['delta'],p['executed'])
        return b['reward']+self.cfg['gamma']*(1-b['done'])*(torch.minimum(a,c)-self.log_alpha.exp()*p['logp'])

    def update(self,b):
        target=self.target_value(b)
        q1,q2=self.q(b['context'],b['delta'],b['executed'])
        loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        self.q_opt.zero_grad(set_to_none=True);loss.backward()
        qgrad=torch.nn.utils.clip_grad_norm_(self.q.parameters(),10)
        self.q_opt.step()
        self.q.requires_grad_(False)
        p=self.policy(b['context'],b['base'])
        q1p,q2p=self.q(b['context'],p['delta'],p['executed'])
        actor_loss=(self.log_alpha.exp().detach()*p['logp']-torch.minimum(q1p,q2p)+10*p['delta'].square().sum(-1,keepdim=True)).mean()
        self.actor_opt.zero_grad(set_to_none=True);actor_loss.backward()
        agrad=torch.nn.utils.clip_grad_norm_(self.actor.parameters(),10)
        self.actor_opt.step();self.q.requires_grad_(True)
        alpha_loss=-(self.log_alpha*(p['logp'].detach()+self.cfg['target_entropy'])).mean()
        self.alpha_opt.zero_grad(set_to_none=True);alpha_loss.backward();self.alpha_opt.step()
        with torch.no_grad():
            for dest,src in zip(self.target.parameters(),self.q.parameters()):
                dest.lerp_(src,self.cfg['tau'])
        self.updates+=1
        values=dict(critic_loss=float(loss.detach()),actor_loss=float(actor_loss.detach()),alpha=float(self.log_alpha.exp().detach()),q_abs_max=float(torch.maximum(q1.abs(),q2.abs()).max()),target_abs_max=float(target.abs().max()),td_abs_mean=float((q1-target).abs().mean()),actor_grad_norm=float(agrad),critic_grad_norm=float(qgrad),clip_fraction=float((p['proposed'].abs()>1).float().mean()),delta_rms=float(p['delta'].square().mean().sqrt()),entropy_unit=float(-p['logp'].mean().detach()))
        if not all(math.isfinite(x) for x in values.values()) or max(values['q_abs_max'],values['target_abs_max'])>10000:
            raise RuntimeError('SAC finite/Q-divergence stop: '+str(values))
        return values

    def state(self):
        return dict(actor=self.actor.state_dict(),q=self.q.state_dict(),target=self.target.state_dict(),log_alpha=self.log_alpha.detach(),actor_opt=self.actor_opt.state_dict(),q_opt=self.q_opt.state_dict(),alpha_opt=self.alpha_opt.state_dict(),updates=self.updates,cfg=self.cfg,bound=self.bound.detach().cpu(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def restore(self,state,restore_rng=False):
        self.actor.load_state_dict(state['actor']);self.q.load_state_dict(state['q']);self.target.load_state_dict(state['target'])
        with torch.no_grad():self.log_alpha.copy_(state['log_alpha'])
        for name in ['actor_opt','q_opt','alpha_opt']:getattr(self,name).load_state_dict(state[name])
        self.updates=state['updates']
        if restore_rng:
            torch.set_rng_state(state['torch_rng'].cpu())
            if state.get('cuda_rng') and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])

class Replay:
    def __init__(self,capacity=50000):
        import numpy as np
        self.capacity=capacity;self.size=0;self.cursor=0;self.data={}
    def add(self,**batch):
        import numpy as np
        n=len(batch['context'])
        if self.size+n>self.capacity:raise RuntimeError('Unexpected replay overwrite')
        for key,v in batch.items():
            v=np.asarray(v,dtype=np.float32)
            if key not in self.data:self.data[key]=np.zeros((self.capacity,*v.shape[1:]),np.float32)
            self.data[key][self.size:self.size+n]=v
        self.size+=n
    def sample(self,n,rng,device):
        ids=rng.integers(0,self.size,n)
        return {k:torch.as_tensor(v[ids],device=device) for k,v in self.data.items()}
    def save(self,path):
        import numpy as np
        np.savez_compressed(path,**{k:v[:self.size] for k,v in self.data.items()})
