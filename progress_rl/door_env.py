"""Phase 8 extension; all Phase 1--7 task files remain immutable."""
import torch
from door_env.door import door_angle, door_velocity, contact_force, CONTACT_FORCE_N, make_cfg
from door_env.isaac_env import PandaDoorEnv
from progress_rl.progress_state import from_measurements, features
from progress_rl.progress_reward import reward


def stage_success(env):
    return (door_angle(env) > env.episode_target).squeeze(-1)


def stage_reward(env):
    value = reward(env)
    # Success remains the same per-second scale; only the curriculum target changes.
    full = (door_angle(env).squeeze(-1) > 1.0).float()
    return value+600.0*(stage_success(env).float()-full)


def config(variant, num_envs=32, device="cuda:0", seed=0):
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg
    cfg = make_cfg(num_envs, device=device, seed=seed)
    if variant in ("B", "D"):
        cfg.rewards.door = RewardTermCfg(func=reward, weight=1.0)
    if variant == "E":
        cfg.rewards.door = RewardTermCfg(func=stage_reward, weight=1.0)
        cfg.terminations.success = TerminationTermCfg(func=stage_success)
    return cfg


class ProgressDoorEnv(PandaDoorEnv):
    def __init__(self, cfg, **kwargs):
        self.goal = 1.0
        self.episode_start = None
        self.episode_target = None
        self.previous_angle = None
        self.final_progress = None
        self.final_start = None
        self.final_target = None
        super().__init__(cfg, **kwargs)

    def get_progress_state(self):
        contact = ((contact_force(self, "door_left_contact") > CONTACT_FORCE_N) &
                   (contact_force(self, "door_right_contact") > CONTACT_FORCE_N))
        return from_measurements(door_angle(self), door_velocity(self), contact,
                                 self.episode_start, self.episode_target)

    def _reset_idx(self, env_ids):
        if self.capture_terminal and len(env_ids):
            state = features(self.get_progress_state())
            if self.final_progress is None:
                self.final_progress = torch.empty_like(state)
                self.final_start = torch.empty_like(self.episode_start)
                self.final_target = torch.empty_like(self.episode_target)
            self.final_progress[env_ids] = state[env_ids].clone()
            self.final_start[env_ids] = self.episode_start[env_ids].clone()
            self.final_target[env_ids] = self.episode_target[env_ids].clone()
        super()._reset_idx(env_ids)
        if self.episode_start is None:
            self.episode_start = torch.zeros_like(door_angle(self))
            self.episode_target = torch.ones_like(self.episode_start)
        self.episode_start[env_ids] = door_angle(self)[env_ids]
        self.episode_target[env_ids] = self.goal

    def step(self, action):
        self.previous_angle = door_angle(self).clone()
        return super().step(action)


def create(cfg):
    env = ProgressDoorEnv(cfg)
    env.reset(seed=cfg.seed)
    env.capture_terminal = True
    return env


def live_observation(env, variant):
    from door_env.door import robot_observation
    robot = robot_observation(env)
    return torch.cat((robot, features(env.get_progress_state())), dim=-1) if variant in ("C", "D", "E") else robot


def terminal_observation(env, variant, done):
    obs = live_observation(env, variant).clone()
    if done.any():
        final = torch.cat((env.final_robot, env.final_progress), dim=-1) if variant in ("C", "D", "E") else env.final_robot
        obs[done] = final[done]
    return obs
