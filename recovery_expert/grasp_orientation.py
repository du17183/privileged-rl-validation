"""Choose among equivalent closing-axis grasps using live IK feasibility."""
import torch
from isaaclab.utils.math import quat_mul,quat_inv,quat_from_euler_xyz,quat_apply,compute_pose_error


@torch.no_grad()
def choose(env,ids,reference):
    term=env.action_manager.get_term('arm_action');j=term._compute_frame_jacobian()[ids].clone()
    damping=term._ik_controller.cfg.ik_params['lambda_val']
    inverse=j.transpose(1,2)@torch.linalg.inv(j@j.transpose(1,2)+damping*damping*torch.eye(6,device=j.device))
    angle=env.get_tool_state()[ids,0];zero=torch.zeros_like(angle)
    yaw=quat_from_euler_xyz(zero,zero,angle)
    nominal=quat_mul(yaw,reference)
    rotations=torch.tensor([-30.,-15.,0.,15.,30.,45.],device=j.device)*torch.pi/180
    pitch=quat_from_euler_xyz(torch.zeros_like(rotations),rotations,torch.zeros_like(rotations))
    candidates=quat_mul(nominal[:,None,:].expand(-1,6,-1),pitch[None,:,:].expand(len(ids),-1,-1))
    flip=torch.tensor([0.,0.,0.,1.],device=j.device).expand(len(ids),6,4)
    candidates=torch.cat((candidates,quat_mul(candidates,flip)),1)
    current=env.scene['ee_frame'].data.target_quat_w[ids,0]
    zeros=torch.zeros((len(ids)*12,3),device=j.device)
    _,error=compute_pose_error(zeros,current[:,None,:].expand(-1,12,-1).reshape(-1,4),zeros,candidates.reshape(-1,4),rot_error_type='axis_angle')
    error=error.reshape(len(ids),12,3)
    pose_delta=torch.cat((torch.zeros_like(error),error.clamp(-.075,.075)),-1)
    dq=torch.einsum('njk,nck->ncj',inverse,pose_delta)
    q=term._asset.data.joint_pos[ids][:,term._joint_ids]
    bounds=term._asset.data.soft_joint_pos_limits[ids][:,term._joint_ids]
    proposed=q[:,None,:]+dq
    violation=(bounds[:,None,:,0]+.005-proposed).clamp_min(0)+(proposed-bounds[:,None,:,1]+.005).clamp_min(0)
    toolz=quat_apply(candidates.reshape(-1,4),torch.tensor([0.,0.,1.],device=j.device).expand(len(ids)*12,3)).reshape(len(ids),12,3)
    normal=torch.stack((angle.cos(),angle.sin(),zero),-1)
    admissible=(toolz*normal[:,None,:]).sum(-1)>.15
    admissible&=toolz[:,:,2]<.25
    score=error.square().sum(-1).sqrt()+100*violation.square().sum(-1).sqrt()
    score=torch.where(admissible,score,torch.full_like(score,1e6))
    chosen=candidates[torch.arange(len(ids),device=j.device),score.argmin(-1)]
    return quat_mul(quat_inv(yaw),chosen)
