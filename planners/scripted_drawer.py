"""Task-space waypoint planner with differential IK actions for Panda drawer.

Approach, align, close fingers, pull, release. It uses handle ground truth only
to generate demonstrations; learned A/B actors never receive that ground truth.
"""

import torch

from isaaclab.utils.math import compute_pose_error

from envs.drawer import drawer_position


class DrawerWaypointPlanner:
    REST, APPROACH, ALIGN, GRASP, PULL, RELEASE = range(6)

    def __init__(self, num_envs, device, step_dt, position_scale=0.05, rotation_scale=0.3):
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.step_dt = step_dt
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.device = device

    def reset(self, env_ids):
        self.state[env_ids] = self.REST
        self.elapsed[env_ids] = 0

    @torch.no_grad()
    def action(self, env):
        ee = env.scene["ee_frame"].data
        cabinet = env.scene["cabinet_frame"].data
        ee_pos = ee.target_pos_w[:, 0, :]
        ee_quat = ee.target_quat_w[:, 0, :]
        handle_pos = cabinet.target_pos_w[:, 0, :]
        handle_quat = cabinet.target_quat_w[:, 0, :]
        drawer = drawer_position(env).squeeze(-1)

        target_pos = handle_pos.clone()
        target_quat = handle_quat.clone()
        target_pos[self.state == self.REST] = ee_pos[self.state == self.REST]
        target_quat[self.state == self.REST] = ee_quat[self.state == self.REST]
        target_pos[self.state == self.APPROACH, 0] -= 0.10
        target_pos[self.state == self.ALIGN, 0] += 0.025
        target_pos[self.state == self.GRASP, 0] += 0.025
        target_pos[self.state == self.PULL, 0] -= 0.015
        target_pos[self.state == self.RELEASE] = ee_pos[self.state == self.RELEASE]
        target_quat[self.state == self.RELEASE] = ee_quat[self.state == self.RELEASE]

        pos_error, rot_error = compute_pose_error(
            ee_pos, ee_quat, target_pos, target_quat, rot_error_type="axis_angle"
        )
        arm = torch.cat((pos_error / self.position_scale, rot_error / self.rotation_scale), dim=-1)
        arm = arm.clamp(-1.0, 1.0)
        gripper = torch.where(
            (self.state == self.GRASP) | (self.state == self.PULL),
            -torch.ones_like(self.elapsed), torch.ones_like(self.elapsed),
        )
        action = torch.cat((arm, gripper[:, None]), dim=-1)

        distance = torch.linalg.vector_norm(ee_pos - target_pos, dim=-1)
        ready = distance < 0.025
        self.elapsed += self.step_dt
        self._advance((self.state == self.REST) & (self.elapsed > 0.5), self.APPROACH)
        self._advance((self.state == self.APPROACH) & ready & (self.elapsed > 0.25), self.ALIGN)
        self._advance((self.state == self.ALIGN) & ready & (self.elapsed > 0.25), self.GRASP)
        self._advance((self.state == self.GRASP) & (self.elapsed > 0.6), self.PULL)
        self._advance((self.state == self.PULL) & ((drawer > 0.31) | (self.elapsed > 3.0)), self.RELEASE)
        return action

    def _advance(self, mask, next_state):
        self.state[mask] = next_state
        self.elapsed[mask] = 0
