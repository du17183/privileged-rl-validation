"""GT Cartesian motion plan for randomized fixtures; existing IK controller.

The clear waypoint keeps the open gripper in front of and above the handle
before descending. Orientation is calibrated to each episode's door yaw.
No robot teleportation, grasp attachment, joint forcing or task edits.
"""
import torch
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, quat_mul
from door_env.door import door_angle, handle_pose


class RandomDoorPlanner:
    REST, CLEAR, APPROACH, ALIGN, GRASP, OPEN, HOLD = range(7)

    def __init__(self, num_envs, device, step_dt, release_orientation=False):
        self.state = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.orientation = torch.zeros((num_envs,4),device=device)
        self.step_dt = step_dt
        self.release_orientation = release_orientation

    def reset(self, ids):
        self.state[ids] = self.REST
        self.elapsed[ids] = 0

    def advance(self, mask, phase):
        self.state[mask] = phase
        self.elapsed[mask] = 0

    @torch.no_grad()
    def action(self, env):
        frame = env.scene['ee_frame'].data
        ee, quat = frame.target_pos_w[:,0], frame.target_quat_w[:,0]
        handle, _ = handle_pose(env)
        cab = env.scene['cabinet']
        hinge = cab.data.body_pos_w[:,cab.body_names.index('door_right_nob_link')]
        angle = door_angle(env).flatten()
        rest = self.state == self.REST
        zeros = torch.zeros_like(angle)
        self.orientation[rest] = quat_mul(quat_from_euler_xyz(zeros,zeros,angle)[rest],quat[rest])
        normal = torch.stack((angle.cos(),angle.sin(),zeros),dim=-1)
        target = handle.clone()
        target_quat = self.orientation.clone()
        target[rest] = ee[rest]; target_quat[rest] = quat[rest]
        clear = self.state == self.CLEAR
        approach = self.state == self.APPROACH
        align = self.state == self.ALIGN
        grasp = self.state == self.GRASP
        opening = self.state == self.OPEN
        holding = self.state == self.HOLD
        if self.release_orientation:
            target_quat[opening | holding] = quat[opening | holding]
        target[clear] -= .15*normal[clear]
        target[clear,2] += .08
        target[approach] -= .08*normal[approach]
        target[align | grasp] += .03*normal[align | grasp]
        lead = .15
        c,s = torch.cos(torch.tensor(lead,device=ee.device)),torch.sin(torch.tensor(lead,device=ee.device))
        dx,dy = handle[:,0]-hinge[:,0],handle[:,1]-hinge[:,1]
        target[opening,0] = hinge[opening,0]+c*dx[opening]-s*dy[opening]+.03
        target[opening,1] = hinge[opening,1]+s*dx[opening]+c*dy[opening]
        target[holding] = ee[holding]
        pe,re = compute_pose_error(ee,quat,target,target_quat,rot_error_type='axis_angle')
        arm = torch.cat((pe/.05,re/.3),-1).clamp(-1,1)
        close = grasp | opening | holding
        action = torch.cat((arm,torch.where(close,-torch.ones_like(angle),torch.ones_like(angle))[:,None]),-1)
        distance = torch.linalg.vector_norm(ee-target,dim=-1)
        self.elapsed += self.step_dt
        self.advance(rest & (self.elapsed > .25),self.CLEAR)
        self.advance(clear & (distance < .025) & (self.elapsed > .2),self.APPROACH)
        self.advance(approach & (distance < .02) & (self.elapsed > .2),self.ALIGN)
        self.advance(align & (distance < .02) & (self.elapsed > .2),self.GRASP)
        self.advance(grasp & (self.elapsed > .5),self.OPEN)
        self.advance(opening & (angle > 1.04),self.HOLD)
        return action
