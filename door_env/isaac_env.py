"""Door environment with correct terminal snapshots across automatic resets."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from .door import door_angle, privileged_state, robot_observation


class PandaDoorEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self.capture_terminal = False
        self.final_robot = None
        self.final_privileged = None
        self.final_door_angle = None
        super().__init__(cfg=cfg, **kwargs)

    def get_tool_state(self):
        return privileged_state(self)

    def _reset_idx(self, env_ids):
        if self.capture_terminal and len(env_ids):
            robot = robot_observation(self)
            gt = privileged_state(self)
            angle = door_angle(self)
            if self.final_robot is None:
                self.final_robot = torch.empty_like(robot)
                self.final_privileged = torch.empty_like(gt)
                self.final_door_angle = torch.empty_like(angle)
            self.final_robot[env_ids] = robot[env_ids].clone()
            self.final_privileged[env_ids] = gt[env_ids].clone()
            self.final_door_angle[env_ids] = angle[env_ids].clone()
        super()._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()


def create_env(cfg):
    env = PandaDoorEnv(cfg=cfg)
    env.reset(seed=cfg.seed)
    env.capture_terminal = True
    return env


@torch.no_grad()
def transition_after_step(env, terminated, truncated):
    done = terminated | truncated
    next_robot = robot_observation(env).clone()
    next_gt = privileged_state(env).clone()
    if done.any():
        next_robot[done] = env.final_robot[done]
        next_gt[done] = env.final_privileged[done]
    return next_robot, next_gt, done
