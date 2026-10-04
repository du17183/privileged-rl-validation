"""Deployment-visible feedback. Actor and critic receive this identical vector.

Velocity is radians/second; remaining angle is radians, with no fitted scaling.
Handle XYZ is workspace metres, centered on the unchanged expert initial mean.
Masks alter only network inputs, never dynamics, rewards, termination or reset.
"""
import torch

PROGRESS = ('door_angle', 'door_angular_velocity', 'target_angle', 'progress', 'remaining_angle')
FEATURES = {'A': (), 'B': PROGRESS, 'C': PROGRESS+('contact_state',),
            'D': PROGRESS+('contact_state','handle_position')}
DIMS = {'A':26, 'B':31, 'C':33, 'D':36}

def encode(robot, state, arm, handle_center):
    fields=[]
    for name in FEATURES[arm]:
        value=state[name]
        if name=='handle_position':
            value=(value-torch.as_tensor(handle_center,device=value.device,dtype=value.dtype))/.1
        fields.append(value)
    result=torch.cat((robot,*fields),-1) if fields else robot
    if result.shape[-1]!=DIMS[arm] or not torch.isfinite(result).all():
        raise ValueError('Invalid feedback observation')
    return result

def mask_state(state, mask, center):
    result={k:v.clone() for k,v in state.items()}
    if mask in ('door_angle','door_angular_velocity','progress','remaining_angle','contact_state'):
        result[mask].zero_()
    elif mask=='handle_position':
        result[mask][:]=torch.as_tensor(center,device=result[mask].device,dtype=result[mask].dtype)
    elif mask in ('angle_family','all_feedback'):
        result['door_angle'].zero_();result['door_angular_velocity'].zero_();result['progress'].zero_()
        result['remaining_angle']=result['target_angle'].clone()
        if mask=='all_feedback':
            result['contact_state'].zero_()
            result['handle_position'][:]=torch.as_tensor(center,device=result['handle_position'].device,dtype=result['handle_position'].dtype)
    else: raise ValueError(mask)
    return result
