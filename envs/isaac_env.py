"""Isaac Lab environment that preserves terminal next states before auto reset."""

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .drawer import drawer_position, privileged_state, robot_observation


class PandaDrawerEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self.capture_terminal = False
        self.final_robot = None
        self.final_privileged = None
        self.final_drawer_position = None
        super().__init__(cfg=cfg, **kwargs)

    def get_privileged_state(self):
        """Fixture-state interface used only by data collection and training critics."""
        return privileged_state(self)

    def get_tool_state(self):
        """Use the same fixture-state call site planned for the real tool controller."""
        return self.get_privileged_state()

    def _reset_idx(self, env_ids):
        if self.capture_terminal and len(env_ids) > 0:
            robot = robot_observation(self)
            privileged = privileged_state(self)
            drawer = drawer_position(self)
            if self.final_robot is None:
                self.final_robot = torch.empty_like(robot)
                self.final_privileged = torch.empty_like(privileged)
                self.final_drawer_position = torch.empty_like(drawer)
            self.final_robot[env_ids] = robot[env_ids].clone()
            self.final_privileged[env_ids] = privileged[env_ids].clone()
            self.final_drawer_position[env_ids] = drawer[env_ids].clone()
        super()._reset_idx(env_ids)
        # Isaac Lab's automatic reset computes observations immediately after
        # _reset_idx, without the forward() used by its explicit reset().
        # Forward here so frame-transformer poses match the just-reset joint
        # state before the next policy/fixture read.
        self.scene.write_data_to_sim()
        self.sim.forward()


def create_env(cfg):
    env = PandaDrawerEnv(cfg=cfg)
    env.reset(seed=cfg.seed)
    env.capture_terminal = True
    return env


@torch.no_grad()
def transition_after_step(env, terminated, truncated):
    """Return true next state even when Isaac Lab reset a finished env in this step."""
    done = torch.logical_or(terminated, truncated)
    robot_next = robot_observation(env).clone()
    privileged_next = privileged_state(env).clone()
    if done.any():
        if env.final_robot is None:
            raise RuntimeError("Terminal snapshot was not captured before auto reset")
        robot_next[done] = env.final_robot[done]
        privileged_next[done] = env.final_privileged[done]
    return robot_next, privileged_next, done
