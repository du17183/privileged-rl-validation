"""Meaningful SAC equation checks; real replay gate is separately required."""
import copy,math,tempfile
import numpy as np
import torch
from torch import nn
from algorithms.residual_sac.sac import SAC,compose,Actor,CONTEXT_DIM

def numerical_gate(real_batch=None,device='cpu'):
    s=SAC(123,[.04,.04,.04,.02,.02,.02],device)
    ctx=torch.zeros(8,CONTEXT_DIM,device=device)
    base=torch.zeros(8,7,device=device);base[:,6]=-1
    unit,_,_,_=s.actor(ctx,True)
    assert torch.equal(unit,torch.zeros_like(unit))
    _,act,_=compose(base,unit,s.bound);assert torch.equal(act,base)
    b=dict(context=ctx,base=base,next_context=ctx+0.1,next_base=base+.1,delta=torch.zeros(8,6,device=device),executed=base,reward=torch.arange(8,device=device,dtype=torch.float32)[:,None],done=torch.tensor([0,1,0,1,0,1,0,1],device=device,dtype=torch.float32)[:,None])
    noise=torch.ones(8,6,device=device)*.2
    p=s.policy(b['next_context'],b['next_base'],noise=noise)
    with torch.no_grad():
        q1,q2=s.target(b['next_context'],p['delta'],p['executed'])
        expected=b['reward']+.99*(1-b['done'])*(torch.minimum(q1,q2)-s.log_alpha.exp()*p['logp'])
    target=s.target_value(b,noise)
    assert torch.equal(target,expected)
    assert torch.equal(target[b['done'].bool()],b['reward'][b['done'].bool()])
    wrong=dict(b,next_base=b['base']);assert not torch.equal(s.target_value(wrong,noise),target)
    q1params=list(s.q.q1.parameters());q2params=list(s.q.q2.parameters())
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(q1params,q2params))
    unit,lp,mean,ls=s.actor(ctx,noise=noise)
    z=mean+ls.exp()*noise
    direct=torch.distributions.Normal(mean,ls.exp()).log_prob(z)-torch.log(1-torch.tanh(z).square())
    assert torch.allclose(lp,direct.sum(-1,keepdim=True),atol=2e-6)
    raw=torch.tensor([[.2,0,0,0,0,0]],device=device,requires_grad=True)
    _,clipped,_=compose(torch.tensor([[1.,0,0,0,0,0,-1]],device=device),raw,s.bound)
    clipped[:,:6].sum().backward()
    assert raw.grad[0,0]==0 and raw.grad[0,1]>0
    before={k:v.detach().clone() for k,v in s.target.state_dict().items()}
    batch=real_batch if real_batch is not None else b
    result=s.update(batch)
    assert result['actor_grad_norm']>0 and result['critic_grad_norm']>0
    for k,v in s.target.state_dict().items():assert torch.allclose(v,before[k]*.995+s.q.state_dict()[k]*.005,atol=1e-6)
    state=copy.deepcopy(s.state()); restored=SAC(998,s.bound.tolist(),device);restored.restore(state)
    for name in ['actor','q','target']:
        for a,c in zip(getattr(s,name).parameters(),getattr(restored,name).parameters()):assert torch.equal(a,c)
    assert torch.equal(s.log_alpha,restored.log_alpha)
    # Identical stochastic update after restoration checks optimizer state too.
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if device!='cpu' else []
    s.update(batch);torch.set_rng_state(rng)
    if cuda:torch.cuda.set_rng_state_all(cuda)
    restored.update(batch)
    for a,c in zip(s.actor.parameters(),restored.actor.parameters()):assert torch.equal(a,c)
    return dict(passed=True,real_interaction_batch=real_batch is not None,zero_mean_exact=True,Q_target_exact=True,terminal_no_bootstrap=True,next_base_substitution_detected=True,independent_doubleQ=True,tanh_entropy_exact=True,clamp_gradient_handled=True,actor_and_critic_gradients=True,polyak_verified=True,checkpoint_optimizer_restore_exact=True,first_update=result)
