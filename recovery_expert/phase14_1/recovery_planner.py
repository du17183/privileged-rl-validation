"""GT waypoint recovery through the unchanged Cartesian DLS action interface."""
import torch
from recovery_expert.recovery_planner import RecoveryPlanner as HistoricalPlanner
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, quat_mul
from door_env.door import handle_pose, door_angle
from recovery_expert.kinematic_compensation import compensate


class RecoveryPlanner(HistoricalPlanner):
    def __init__(self,n,device,dt):
        super().__init__(n,device,dt)
        self.retry=torch.zeros(n,device=device,dtype=torch.long)
        self.position_error=torch.zeros(n,device=device)
        self.rotation_error=torch.zeros(n,device=device)
        self.target_pose=torch.zeros((n,7),device=device)
        self.reentry=torch.ones(n,device=device,dtype=torch.long)

    def reset(self,ids,env):
        super().reset(ids,env);self.retry[ids]=0

    def takeover(self,ids,env):
        f=env.scene['ee_frame'].data
        self.release_pos[ids]=f.target_pos_w[ids,0]
        self.release_quat[ids]=f.target_quat_w[ids,0]
        self.home_quat[ids]=self.calibrated_quat
        self.state[ids]=self.RELEASE;self.elapsed[ids]=0;self.started[ids]=True
        handle,_=handle_pose(env)
        distance=(self.release_pos[ids]-handle[ids]).norm(dim=-1)
        self.reentry[ids]=torch.where(distance<.055,self.ALIGN,
                                     torch.where(distance<.10,self.APPROACH,self.CLEAR))
        # A regressed door leaves the tool at a different tangential position.
        # Retreat outside the panel before rotating or approaching the handle.
        self.reentry[ids]=torch.where(env.get_tool_state()[ids,0]>.2,
                                     torch.full_like(self.reentry[ids],self.CLEAR),self.reentry[ids])

    @torch.no_grad()
    def action(self,env):
        f=env.scene['ee_frame'].data;ee,quat=f.target_pos_w[:,0],f.target_quat_w[:,0]
        handle,_=handle_pose(env);angle=door_angle(env).flatten();z=torch.zeros_like(angle)
        normal=torch.stack((angle.cos(),angle.sin(),z),-1)
        orientation=quat_mul(quat_from_euler_xyz(z,z,angle),self.home_quat)
        target=handle.clone()
        release,clear,approach,align,grasp,opening,hold=[self.state==i for i in range(7)]
        target[release]=self.release_pos[release];orientation[release]=self.release_quat[release]
        target[clear]-=.10*normal[clear];target[clear,2]+=.055
        target[clear,0]=torch.maximum(target[clear,0],env.scene.env_origins[clear,0]+.25)
        orientation[clear]=self.release_quat[clear]
        orienting=self.state==self.ORIENT
        target[orienting]-=.10*normal[orienting];target[orienting,2]+=.055
        target[orienting,0]=torch.maximum(target[orienting,0],env.scene.env_origins[orienting,0]+.25)
        target[approach]-=.06*normal[approach]
        target[align|grasp]+=.03*normal[align|grasp]
        cab=env.scene['cabinet'];hinge=cab.data.body_pos_w[:,cab.body_names.index('door_right_nob_link')]
        c,s=torch.cos(torch.tensor(.15,device=ee.device)),torch.sin(torch.tensor(.15,device=ee.device))
        dx,dy=handle[:,0]-hinge[:,0],handle[:,1]-hinge[:,1]
        target[opening,0]=hinge[opening,0]+c*dx[opening]-s*dy[opening]+.03
        target[opening,1]=hinge[opening,1]+s*dx[opening]+c*dy[opening]
        orientation[opening|hold]=quat[opening|hold];target[hold]=ee[hold]
        pe,re=compute_pose_error(ee,quat,target,orientation,rot_error_type='axis_angle')
        self.position_error=pe.norm(dim=-1);self.rotation_error=re.norm(dim=-1)
        self.target_pose=torch.cat((target-env.scene.env_origins,orientation),-1)
        action=torch.cat(((pe/.05).clamp(-.6,.6),(re/.3).clamp(-.3,.3),
                          torch.where(grasp|opening|hold,-torch.ones_like(angle),torch.ones_like(angle))[:,None]),-1)
        action[opening,:3]=(pe[opening]/.05).clamp(-1,1)
        action[clear|orienting,:3]=(pe[clear|orienting]/.05).clamp(-1,1)
        action[orienting,3:6]=(re[orienting]/.3).clamp(-.5,.5)
        self.elapsed+=self.dt;active=self.started
        gt=env.get_tool_state();near=(ee-handle).norm(dim=-1)<.06
        actual_grasp=(gt[:,9:11]>.5).all(-1)&near&(env.action_manager.action[:,6]<0)
        released=active&release&(self.elapsed>.15)
        for destination in [self.CLEAR,self.APPROACH,self.ALIGN]:
            self.advance(released&(self.reentry==destination),destination)
        self.advance(active&clear&(self.position_error<.025)&(self.elapsed>.1),self.ORIENT)
        self.advance(active&orienting&(self.position_error<.03)&(self.rotation_error<.2)&(self.elapsed>.1),self.APPROACH)
        self.advance(active&approach&(self.position_error<.018)&(self.rotation_error<.25)&(self.elapsed>.1),self.ALIGN)
        self.advance(active&align&(self.position_error<.018)&(self.rotation_error<.25)&(self.elapsed>.1),self.GRASP)
        self.advance(active&grasp&actual_grasp&(self.elapsed>.45),self.OPEN)
        retry=active&grasp&~actual_grasp&(self.elapsed>1.)&(self.retry<2)
        self.retry[retry]+=1;self.advance(retry,self.CLEAR)
        self.advance(active&opening&(angle>1.04),self.HOLD)
        return compensate(env,action,active)
