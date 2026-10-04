"""Closed-loop Cartesian waypoints for grasping and rotating the cabinet door.

The planner reads fixture truth only for generating offline expert labels. A/B/D
learned actors receive solely ``robot_observation``.
"""

import torch
from isaaclab.utils.math import compute_pose_error
from door_env.door import door_angle, handle_pose


class DoorWaypointPlanner:
    REST, APPROACH, ALIGN, GRASP, OPEN, HOLD = range(6)

    def __init__(self, num_envs, device, step_dt, lead_angle=0.15):
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.step_dt = step_dt
        self.lead_angle = lead_angle

    def reset(self, env_ids):
        self.state[env_ids] = self.REST
        self.elapsed[env_ids] = 0

    @torch.no_grad()
    def action(self, env):
        ee = env.scene["ee_frame"].data
        ee_pos = ee.target_pos_w[:, 0]
        ee_quat = ee.target_quat_w[:, 0]
        handle_pos, _ = handle_pose(env)
        cabinet = env.scene["cabinet"]
        hinge = cabinet.data.body_pos_w[:, cabinet.body_names.index("door_right_nob_link")]
        q = door_angle(env).squeeze(-1)
        target_pos = handle_pos.clone()
        target_quat = ee_quat.clone()
        rest = self.state == self.REST
        approach = self.state == self.APPROACH
        align = self.state == self.ALIGN
        grasp = self.state == self.GRASP
        opening = self.state == self.OPEN
        holding = self.state == self.HOLD
        target_pos[rest] = ee_pos[rest]
        target_pos[approach, 0] -= 0.06
        target_pos[align, 0] += 0.03
        target_pos[grasp, 0] += 0.03
        # Follow a circular end-effector path about the physical hinge. The
        # target leads the measured handle so the gripper transmits torque.
        lead = torch.full_like(q, self.lead_angle)
        dx = handle_pos[:, 0] - hinge[:, 0]
        dy = handle_pos[:, 1] - hinge[:, 1]
        target_pos[opening, 0] = hinge[opening, 0] + torch.cos(lead[opening]) * dx[opening] - torch.sin(lead[opening]) * dy[opening]
        target_pos[opening, 1] = hinge[opening, 1] + torch.sin(lead[opening]) * dx[opening] + torch.cos(lead[opening]) * dy[opening]
        target_pos[opening, 0] += 0.03
        target_pos[holding] = ee_pos[holding]

        pos_error, rot_error = compute_pose_error(
            ee_pos, ee_quat, target_pos, target_quat, rot_error_type="axis_angle"
        )
        arm = torch.cat((pos_error / 0.05, rot_error / 0.3), dim=-1).clamp(-1, 1)
        close = grasp | opening | holding
        action = torch.cat((arm, torch.where(close, -torch.ones_like(q), torch.ones_like(q))[:, None]), dim=-1)

        distance = torch.linalg.vector_norm(ee_pos - target_pos, dim=-1)
        self.elapsed += self.step_dt
        self._advance(rest & (self.elapsed > 0.25), self.APPROACH)
        self._advance(approach & (distance < 0.025) & (self.elapsed > 0.2), self.ALIGN)
        self._advance(align & (distance < 0.02) & (self.elapsed > 0.2), self.GRASP)
        self._advance(grasp & (self.elapsed > 0.5), self.OPEN)
        self._advance(opening & (q > 1.04), self.HOLD)
        return action

    def _advance(self, mask, target):
        self.state[mask] = target
        self.elapsed[mask] = 0
