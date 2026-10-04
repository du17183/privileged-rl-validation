"""Same 39->256->256->7 policy and MSE as Phase13.

The target is exactly 1 rad for every episode. Its formerly constant feature
slot carries measured angular velocity, with normalization fitted ONLY on S
training trajectories. Target remains an explicit checkpoint constant and is
also recoverable from angle + remaining angle. No information is discarded
on this fixed-target task; variable target tasks need a different interface.
"""
import torch
from experiments.phase13_random_expert_bc.model import Policy
from experiments.phase13_random_expert_bc.state import pack

FEATURES=['door_angle','door_angular_velocity','progress','remaining_angle',
          'left_contact','right_contact','handle_x','handle_y','handle_z',
          'handle_qw','handle_qx','handle_qy','handle_qz']

def measured(env):
    state=env.get_environment_state()
    if not torch.allclose(state['target_angle'],torch.ones_like(state['target_angle'])):
        raise RuntimeError('This frozen interface requires target_angle=1 rad')
    result=pack(state).clone();result[:,1]=state['door_angular_velocity'].flatten()
    return result

def inputs(robot,environment,mean,std,mode):
    x=(torch.cat((robot,environment),-1)-mean)/std
    if mode=='robot':x[...,26:]=0
    elif mode=='no_handle':x[...,32:39]=0
    elif mode!='full':raise ValueError(mode)
    return x

def load(path,device):
    meta=torch.load(path,map_location=device,weights_only=False)
    net=Policy().to(device);net.load_state_dict(meta['model']);net.eval()
    return net,torch.tensor(meta['mean'],device=device),torch.tensor(meta['std'],device=device),meta
