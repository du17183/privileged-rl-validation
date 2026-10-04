"""Expert-only input compensation; the original environment IK is unchanged.

Its translational Jacobian uses a fixed frame offset in the root axes. The
actual tool offset rotates with the hand. Use the physical frame Jacobian
to request bounded commands through that same IK and action interface.
"""
import torch
import numpy as np
from scipy.optimize import minimize,Bounds,LinearConstraint
from isaaclab.utils.math import matrix_from_quat,quat_inv,quat_mul,quat_apply,skew_symmetric_matrix


@torch.no_grad()
def maps(env):
    term=env.action_manager.get_term('arm_action')
    body=term.jacobian_w.clone()
    root_inv=quat_inv(term._asset.data.root_quat_w)
    rotation=matrix_from_quat(root_inv)
    body[:,:3]=rotation@body[:,:3];body[:,3:]=rotation@body[:,3:]
    implemented=body.clone();physical=body.clone()
    if term.cfg.body_offset is not None:
        # PhysX linear body Jacobians are referenced at body COM. Account
        # for COM -> tool, not just link origin -> tool.
        tool,_=term._compute_frame_pose()
        com=quat_apply(root_inv,term._asset.data.body_com_pos_w[:,term._body_idx]-term._asset.data.root_pos_w)
        offset=tool-com
        implemented[:,:3]+=(-skew_symmetric_matrix(term._offset_pos))@body[:,3:]
        implemented[:,3:]=matrix_from_quat(term._offset_rot)@body[:,3:]
        physical[:,:3]+=(-skew_symmetric_matrix(offset))@body[:,3:]
    damping=term._ik_controller.cfg.ik_params['lambda_val']
    inverse=torch.linalg.solve(implemented@implemented.transpose(1,2)+damping*damping*torch.eye(6,device=body.device),implemented).transpose(1,2)
    mapping=inverse*term._scale[:,None,:]
    return implemented,physical,mapping


@torch.no_grad()
def compensate(env,action,active):
    _,physical,mapping=maps(env)
    response=physical@mapping
    desired=action[:,:6]*env.action_manager.get_term('arm_action')._scale
    normal=response.transpose(1,2)@response+1e-6*torch.eye(6,device=response.device)
    u=torch.linalg.solve(normal,response.transpose(1,2)@desired[:,:,None]).squeeze(-1)
    u/=u.abs().amax(-1,keepdim=True).clamp_min(1)
    result=action.clone();result[active,:6]=u[active]
    return result


@torch.no_grad()
def position_task(env,action,active):
    """Position IK with genuinely free orientation and a home-posture prior.

    Solve a six-command convex QP through the unchanged controller. This
    releases three rotation constraints during collision-free reacquisition
    waypoints; it is still a local expert, not a certified global planner.
    """
    if not active.any():return action
    _,physical,mapping=maps(env);term=env.action_manager.get_term('arm_action')
    response=(physical@mapping)[:,:3]
    q=term._asset.data.joint_pos[:,term._joint_ids]
    goal=term._asset.data.default_joint_pos[:,term._joint_ids]
    prior=((goal-q)*.05).clamp(-.03,.03)
    desired=action[:,:3]*term._scale[:,:3]
    h=1000*response.transpose(1,2)@response+mapping.transpose(1,2)@mapping+1e-4*torch.eye(6,device=q.device)
    rhs=(1000*response.transpose(1,2)@desired[:,:,None]+mapping.transpose(1,2)@prior[:,:,None]).squeeze(-1)
    limits=term._asset.data.soft_joint_pos_limits[:,term._joint_ids]
    lower=(torch.minimum(limits[:,:,0]+.005,q)-q).clamp_min(-.12)
    upper=(torch.maximum(limits[:,:,1]-.005,q)-q).clamp_max(.12)
    values=[v.detach().cpu().numpy() for v in [h,rhs,mapping,lower,upper]]
    result=action.clone()
    if not hasattr(env,'phase14_qp'):env.phase14_qp=dict(solves=0,failures=0)
    for i in active.nonzero().flatten().tolist():
        hi,bi,mi,lo,up=[v[i] for v in values];u=np.linalg.solve(hi,bi)
        if np.max(np.abs(u))>1 or np.any(mi@u<lo-1e-7) or np.any(mi@u>up+1e-7):
            solution=minimize(lambda x:.5*x@hi@x-bi@x,np.zeros(6),jac=lambda x:hi@x-bi,
                method='SLSQP',bounds=Bounds(-np.ones(6),np.ones(6)),constraints=LinearConstraint(mi,lo,up),
                options=dict(ftol=1e-8,maxiter=40))
            env.phase14_qp['solves']+=1
            if not solution.success or not np.isfinite(solution.x).all():
                env.phase14_qp['failures']+=1;u=np.zeros(6)
            else:u=solution.x
        result[i,:6]=torch.tensor(u,device=q.device,dtype=q.dtype)
    return result
