"""Track FK of a bounded joint-space retreat through the original IK action.

No joint targets/state are written here. The environment remains the sole
actuator controller; all supervised labels are executed Cartesian actions.
"""
import sys,json,h5py
from pathlib import Path
import numpy as np,torch,pinocchio as pin
from isaaclab.utils.math import compute_pose_error


class CartesianJointPath:
    def __init__(self,env):
        urdf=Path(sys.prefix)/'lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
        self.model=pin.buildModelFromUrdf(str(urdf));self.data=self.model.createData()
        self.frame=self.model.getFrameId('panda_hand')
        self.indices=[self.model.joints[self.model.getJointId(name)].idx_q for name in env.scene['robot'].joint_names]
        self.offset=pin.SE3(np.eye(3),np.array([0.,0.,.107]))
        self.start=env.scene['robot'].data.joint_pos.clone()
        base=Path(__file__).resolve().parents[1]/'datasets/random_door_expert'
        keys=json.loads((base/'split_v1/split.json').read_text())['splits']['train'];poses=[];labels=[]
        with h5py.File(base/'collection_v1/trajectories.h5','r') as h:
            for key in keys:
                g=h[key];phase=g['planner_phase'][:].flatten();indices=np.flatnonzero(phase==1)
                if len(indices):poses.append(g['observation'][indices[-1],:9]);labels.append(key)
        values=np.asarray(poses);index=np.argmin(((values[:,:7]-np.median(values[:,:7],axis=0))**2).sum(-1))
        self.goal=torch.tensor(values[index],device=env.device).expand(env.num_envs,-1).clone()
        print('RETREAT_TARGET validated expert CLEAR joint pose '+labels[index],flush=True)
        self.fraction=torch.zeros(env.num_envs,device=env.device)
        pos,quat=self.forward(env,self.start)
        actualp,actualq=env.action_manager.get_term('arm_action')._compute_frame_pose()
        pe,re=compute_pose_error(actualp,actualq,pos,quat,rot_error_type='axis_angle')
        position=float(pe.norm(dim=-1).max());rotation=float(re.norm(dim=-1).max())
        print(f'FK_VALIDATION position_max_m={position:.8f} rotation_max_rad={rotation:.8f}',flush=True)
        if position>.003 or rotation>.03:raise RuntimeError('Bundled Panda URDF does not match unchanged simulated robot')

    def forward(self,env,values):
        arrays=values.detach().cpu().numpy();positions=[];quaternions=[]
        for row in arrays:
            q=np.zeros(self.model.nq);q[self.indices]=row
            pin.forwardKinematics(self.model,self.data,q);pin.updateFramePlacements(self.model,self.data)
            pose=self.data.oMf[self.frame]*self.offset
            positions.append(pose.translation.copy());quat=pin.Quaternion(pose.rotation).coeffs()
            quaternions.append([quat[3],quat[0],quat[1],quat[2]])
        return torch.tensor(np.asarray(positions),device=env.device,dtype=torch.float32),torch.tensor(np.asarray(quaternions),device=env.device,dtype=torch.float32)

    def reset(self,ids,env):
        self.start[ids]=env.scene['robot'].data.joint_pos[ids];self.fraction[ids]=0

    def action(self,env,active):
        self.fraction[active]=(self.fraction[active]+1/120).clamp_max(1)
        virtual=self.start+(self.goal-self.start)*self.fraction[:,None]
        targetp,targetq=self.forward(env,virtual)
        pos,quat=env.action_manager.get_term('arm_action')._compute_frame_pose()
        pe,re=compute_pose_error(pos,quat,targetp,targetq,rot_error_type='axis_angle')
        scale=env.action_manager.get_term('arm_action')._scale
        return (torch.cat((pe,re),-1)/scale).clamp(-1,1),pe.norm(dim=-1),re.norm(dim=-1)
