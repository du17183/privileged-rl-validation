"""Identical policy capacity for the robot-only and conditioned BC arms."""
import math
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(39,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.head=nn.Linear(256,7)
        self.log_std=nn.Parameter(torch.full((7,),math.log(.01)),requires_grad=False)

    def mean(self,observation):return self.head(self.encoder(observation))

    def forward(self,observation,stochastic=False,generator=None):
        mean=self.mean(observation)
        if stochastic:
            noise=torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator)
            mean=mean+self.log_std.exp().clamp_max(.01)*noise
        return torch.tanh(mean)


def inputs(robot,environment,mean,std,arm,mask='none'):
    full=torch.cat((robot,environment),-1)
    normalized=(full-mean)/std
    if arm=='A':normalized[...,26:]=0
    if mask=='angle_only':normalized[...,26]=0
    if mask=='coherent_angle_progress_remaining':normalized[...,[26,28,29]]=0
    if mask=='handle_pose':normalized[...,32:39]=0
    if mask=='contact':normalized[...,30:32]=0
    if mask=='all_environment':normalized[...,26:]=0
    return normalized


def load_checkpoint(path,device):
    value=torch.load(path,map_location=device,weights_only=False)
    policy=Policy().to(device);policy.load_state_dict(value['model']);policy.eval()
    return policy,torch.tensor(value['mean'],device=device),torch.tensor(value['std'],device=device),value
