"""Reacquire the live handle or continue a valid grasp without resetting."""
import torch
import h5py,json
from pathlib import Path
from isaaclab.utils.math import compute_pose_error,quat_from_euler_xyz,quat_mul
from door_env.door import door_angle,handle_pose
from recovery_expert.ik_feasibility import project,posture_command
from recovery_expert.grasp_orientation import choose
from recovery_expert.cartesian_joint_path import CartesianJointPath


class RecoveryPlanner:
    RELEASE,CLEAR,APPROACH,ALIGN,GRASP,OPEN,HOLD=range(7)
    LIFT=7
    RETRACT=8
    ORIENT=9
    POSTURE=10

    def __init__(self,n,device,dt):
        self.state=torch.zeros(n,device=device,dtype=torch.long)
        self.elapsed=torch.zeros(n,device=device);self.dt=dt
        self.home_quat=torch.zeros((n,4),device=device)
        self.release_pos=torch.zeros((n,3),device=device)
        self.release_quat=self.home_quat.clone();self.lift_pos=self.release_pos.clone()
        self.need_lift=torch.zeros(n,device=device,dtype=torch.bool)
        self.flip=torch.zeros(n,device=device,dtype=torch.bool)
        self.fast=torch.zeros(n,device=device,dtype=torch.bool)
        self.path=None
        # Robot orientation calibration from the already frozen expert REST
        # phase, not from policy outcomes or heldout recovery tests.
        rows=[]
        base=Path(__file__).resolve().parents[1]/'datasets/random_door_expert'
        training_ids=json.loads((base/'split_v1/split.json').read_text())['splits']['train']
        with h5py.File(base/'collection_v1/trajectories.h5','r') as h:
            for key in training_ids:
                g=h[key]
                phase=g['planner_phase'][:].flatten();idx=(phase==0).nonzero()[0][-1]
                rows.append(torch.tensor(g['observation'][idx,21:25]))
        reference=torch.stack(rows).to(device);reference=torch.where((reference*reference[:1]).sum(-1,keepdim=True)<0,-reference,reference)
        self.calibrated_quat=reference.mean(0);self.calibrated_quat/=self.calibrated_quat.norm()
        self.started=torch.zeros(n,device=device,dtype=torch.bool)

    def reset(self,ids,env):
        self.started[ids]=False;self.state[ids]=self.RELEASE;self.elapsed[ids]=0
        self.home_quat[ids]=self.calibrated_quat

    def takeover(self,ids,env):
        gt=env.get_tool_state()
        frame=env.scene['ee_frame'].data;handle,_=handle_pose(env)
        near_handle=(frame.target_pos_w[ids,0]-handle[ids]).norm(dim=-1)<.035
        close_command=env.action_manager.action[ids,6]<0
        continue_grasp=(gt[ids,9:11]>.5).all(-1)&(gt[ids,0]>.05)&near_handle&close_command
        self.state[ids]=torch.where(continue_grasp,self.OPEN,self.RELEASE)
        self.fast[ids]=env.episode_length_buf[ids]<90
        self.elapsed[ids]=0;self.started[ids]=True
        frame=env.scene['ee_frame'].data
        self.release_pos[ids]=frame.target_pos_w[ids,0]
        self.release_quat[ids]=frame.target_quat_w[ids,0]
        self.home_quat[ids]=self.calibrated_quat
        angle=gt[ids,0];zero=torch.zeros_like(angle)
        nominal=quat_mul(quat_from_euler_xyz(zero,zero,angle),self.home_quat[ids])
        symmetric=quat_mul(nominal,torch.tensor([0.,0.,0.,1.],device=gt.device).expand(len(ids),4))
        self.flip[ids]=False
        term=env.action_manager.get_term('arm_action');q=term._asset.data.joint_pos[ids][:,term._joint_ids]
        bounds=term._asset.data.soft_joint_pos_limits[ids][:,term._joint_ids]
        near=((q-bounds[:,:,0])<.04)|((bounds[:,:,1]-q)<.04)
        self.state[ids]=torch.where(near.any(-1)&~continue_grasp,torch.full_like(self.state[ids],self.POSTURE),self.state[ids])
        awkward=(q[:,5]>3.35)|(q[:,1]<-1.35)
        self.state[ids]=torch.where(awkward&~continue_grasp,torch.full_like(self.state[ids],self.POSTURE),self.state[ids])
        if self.path is None:self.path=CartesianJointPath(env)
        self.path.reset(ids,env)
        gripper_open=env.scene['robot'].data.joint_pos[ids,7:9].sum(-1)>.065
        self.state[ids]=torch.where(self.fast[ids]&gripper_open&~near.any(-1)&~awkward,torch.full_like(self.state[ids],self.CLEAR),self.state[ids])
        handle,_=handle_pose(env);distance=torch.linalg.vector_norm(self.release_pos[ids]-handle[ids],dim=-1)
        self.need_lift[ids]=(distance<.17)&(self.release_pos[ids,2]<handle[ids,2]+.04)
        self.lift_pos[ids]=self.release_pos[ids]
        angle=gt[ids,0];normal=torch.stack((angle.cos(),angle.sin(),torch.zeros_like(angle)),-1)
        self.lift_pos[ids,:2]-=.10*normal[:,:2]
        self.lift_pos[ids,2]=torch.maximum(self.release_pos[ids,2],handle[ids,2]+.07)

    def advance(self,mask,phase):self.state[mask]=phase;self.elapsed[mask]=0

    @torch.no_grad()
    def action(self,env):
        f=env.scene['ee_frame'].data;ee,quat=f.target_pos_w[:,0],f.target_quat_w[:,0]
        handle,_=handle_pose(env);cab=env.scene['cabinet']
        hinge=cab.data.body_pos_w[:,cab.body_names.index('door_right_nob_link')]
        angle=door_angle(env).flatten();z=torch.zeros_like(angle)
        normal=torch.stack((angle.cos(),angle.sin(),z),-1)
        target=handle.clone();orientation=quat_mul(quat_from_euler_xyz(z,z,angle),self.home_quat)
        symmetric=quat_mul(orientation,torch.tensor([0.,0.,0.,1.],device=ee.device).expand(env.num_envs,4))
        orientation[self.flip]=symmetric[self.flip]
        masks=[self.state==i for i in range(7)];release,clear,approach,align,grasp,opening,hold=masks
        target[release]=self.release_pos[release]-.015*normal[release];orientation[release]=self.release_quat[release]
        retracting=self.state==self.RETRACT
        target[retracting]=self.lift_pos[retracting];target[retracting,2]=self.release_pos[retracting,2]
        orientation[retracting]=quat[retracting]
        lifting=self.state==self.LIFT
        target[lifting]=self.lift_pos[lifting];orientation[lifting]=quat[lifting]
        target[clear]-=.15*normal[clear];target[clear,2]+=.08
        target[clear,0]=torch.maximum(target[clear,0],env.scene.env_origins[clear,0]+.25)
        orienting=self.state==self.ORIENT
        target[orienting]-=.15*normal[orienting];target[orienting,2]+=.08
        target[orienting,0]=torch.maximum(target[orienting,0],env.scene.env_origins[orienting,0]+.25)
        target[approach]-=.08*normal[approach]
        free=(clear|approach)&~self.fast
        orientation[free]=quat[free]
        target[align|grasp]+=.03*normal[align|grasp]
        c,s=torch.cos(torch.tensor(.15,device=ee.device)),torch.sin(torch.tensor(.15,device=ee.device))
        dx,dy=handle[:,0]-hinge[:,0],handle[:,1]-hinge[:,1]
        target[opening,0]=hinge[opening,0]+c*dx[opening]-s*dy[opening]+.03
        target[opening,1]=hinge[opening,1]+s*dx[opening]+c*dy[opening]
        target[hold]=ee[hold];orientation[opening|hold]=quat[opening|hold]
        pe,re=compute_pose_error(ee,quat,target,orientation,rot_error_type='axis_angle')
        arm=torch.cat((pe/.05,re/.3),-1).clamp(-1,1)
        arm[:,3:]=arm[:,3:].clamp(-.25,.25)
        closed=grasp|opening|hold
        action=torch.cat((arm,torch.where(closed,-torch.ones_like(angle),torch.ones_like(angle))[:,None]),-1)
        posture=self.state==self.POSTURE
        if posture.any():
            path_action,path_pe,path_re=self.path.action(env,posture)
            action[posture,:6]=posture_command(env,self.path.goal)[posture]
        dist=torch.linalg.vector_norm(ee-target,dim=-1);self.elapsed+=self.dt
        active=self.started
        finished_posture=active&posture&((self.elapsed>2.8)|((self.elapsed>2)&(path_pe<.03)&(path_re<.20))) if posture.any() else torch.zeros_like(active)
        if finished_posture.any():
            ids=finished_posture.nonzero().flatten()
            self.home_quat[ids]=self.calibrated_quat
            self.flip[ids]=False
        self.advance(finished_posture,self.CLEAR)
        self.advance(active&release&(self.elapsed>.25)&~self.fast,self.RETRACT)
        self.advance(active&release&(self.elapsed>.25)&self.fast,self.CLEAR)
        self.advance(active&retracting&((dist<.025)|(self.elapsed>.8)),self.LIFT)
        self.advance(active&lifting&((dist<.025)|(self.elapsed>.8)),self.CLEAR)
        self.advance(active&clear&(dist<.025)&(self.elapsed>.15)&~self.fast,self.ORIENT)
        self.advance(active&clear&(dist<.025)&(self.elapsed>.15)&self.fast,self.APPROACH)
        self.advance(active&orienting&(torch.linalg.vector_norm(re,dim=-1)<.2)&(dist<.04)&(self.elapsed>.15),self.APPROACH)
        self.advance(active&approach&(dist<.02)&(self.elapsed>.15),self.ALIGN)
        self.advance(active&align&(dist<.02)&(self.elapsed>.15),self.GRASP)
        self.advance(active&grasp&(self.elapsed>.5),self.OPEN)
        self.advance(active&opening&(angle>1.04),self.HOLD)
        return project(env,action,active)
