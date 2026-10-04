"""Project Cartesian commands using the existing DLS IK map and joint bounds.

This changes expert commands only. It does not replace the environment's IK,
set joint positions/targets, or modify robot limits. Projection is a local
linear approximation, not a global collision-free motion planning guarantee.
"""
import torch
from isaaclab.utils.math import matrix_from_quat,quat_inv,skew_symmetric_matrix


@torch.no_grad()
def posture_command(env,goal=None):
    term=env.action_manager.get_term('arm_action')
    # Read the same frame Jacobian; get_jacobians advanced joint indexing
    # returns a separate tensor, so this does not change physical state.
    j=term._compute_frame_jacobian().clone()
    q=term._asset.data.joint_pos[:,term._joint_ids]
    if goal is None:goal=term._asset.data.default_joint_pos[:,term._joint_ids]
    else:goal=goal[:,term._joint_ids]
    delta=((goal-q)*.12).clamp(-.06,.06)
    damping=term._ik_controller.cfg.ik_params['lambda_val']
    inverse=torch.linalg.solve(j@j.transpose(1,2)+damping*damping*torch.eye(6,device=j.device),j).transpose(1,2)
    scale=term._scale
    mapping=inverse*scale[:,None,:] if scale.ndim==2 else inverse*scale
    # Invert the actual unchanged DLS command-to-joint map. Multiplying by
    # J alone ignores damping and cannot prioritize leaving a joint limit.
    weight=torch.ones_like(q);weight[:,4:6]=3
    h=mapping.transpose(1,2)@(mapping*weight[:,:,None])
    rhs=mapping.transpose(1,2)@(delta*weight)[:,:,None]
    u=torch.linalg.solve(h+1e-5*torch.eye(6,device=j.device),rhs).squeeze(-1)
    u/=u.abs().amax(-1,keepdim=True).clamp_min(1)
    return u


@torch.no_grad()
def project(env,action,active):
    term=env.action_manager.get_term('arm_action')
    if term._ik_controller.cfg.ik_method!='dls':raise RuntimeError('Expected unchanged DLS IK')
    j=term.jacobian_w.clone()
    rotation=matrix_from_quat(quat_inv(term._asset.data.root_quat_w))
    j[:,:3]=rotation@j[:,:3];j[:,3:]=rotation@j[:,3:]
    if term.cfg.body_offset is not None:
        j[:,:3]+=(-skew_symmetric_matrix(term._offset_pos))@j[:,3:]
        j[:,3:]=matrix_from_quat(term._offset_rot)@j[:,3:]
    damping=term._ik_controller.cfg.ik_params['lambda_val']
    inverse=torch.linalg.solve(j@j.transpose(1,2)+damping*damping*torch.eye(6,device=j.device),j).transpose(1,2)
    scale=term._scale
    mapping=inverse*scale[:,None,:] if scale.ndim==2 else inverse*scale
    q=term._asset.data.joint_pos[:,term._joint_ids]
    limits=term._asset.data.soft_joint_pos_limits[:,term._joint_ids]
    lower=torch.minimum(limits[:,:,0]+.005,q)-q
    upper=torch.maximum(limits[:,:,1]-.005,q)-q
    lower=lower.clamp_min(-.12);upper=upper.clamp_max(.12)
    u=action[:,:6].clone()
    for _ in range(8):
        for k in range(mapping.shape[1]):
            row=mapping[:,k];norm=row.square().sum(-1).clamp_min(1e-10)
            value=(row*u).sum(-1)
            correction=(value-upper[:,k]).clamp_min(0)-(lower[:,k]-value).clamp_min(0)
            u-=row*(correction/norm)[:,None];u.clamp_(-1,1)
    result=action.clone();result[active,:6]=u[active]
    return result
