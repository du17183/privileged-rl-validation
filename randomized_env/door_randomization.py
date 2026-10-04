"""Per-episode legal angle, fixture translation and material randomization.

The cabinet is translated rigidly: geometry/control/reward are unchanged.
Each clone owns an independent RNG stream; learner/evaluator RNGs are separate.
"""
import math
import numpy as np
import torch
from progress_rl.door_env import ProgressDoorEnv, config
from door_env.door import door_index
from environment_state.state_interface import get_environment_state, clone_state

LEVELS = {0: (0., 0., 0.), 1: (2.5, .005, 0.), 2: (5., .01, 0.), 3: (5., .01, .1)}


class RandomizedDoorEnv(ProgressDoorEnv):
    def __init__(self, cfg, **kwargs):
        self.level = 0
        self.preset = None
        self.final_environment = None
        self.final_parameters = None
        self.parameters = None
        self.episode_level = self.final_level = None
        self.nominal_root = self.nominal_joints = self.nominal_material = None
        self.random_seed = cfg.seed
        self.streams = None
        self.physical_reset_checks = 0
        super().__init__(cfg, **kwargs)

    def get_environment_state(self):
        return get_environment_state(self)

    def set_randomization(self, level, preset=None):
        if level not in LEVELS:
            raise ValueError(level)
        self.level, self.preset = level, preset

    def seed_streams(self, seed):
        self.random_seed = int(seed)
        self.streams = [np.random.default_rng(np.random.SeedSequence([int(seed), i, 11011])) for i in range(self.num_envs)]

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed_streams(seed)
        return super().reset(seed=seed, **kwargs)

    def _ensure_defaults(self):
        if self.nominal_root is not None:
            return
        cabinet = self.scene['cabinet']
        self.nominal_root = cabinet.data.default_root_state.clone()
        self.nominal_joints = cabinet.data.default_joint_pos.clone()
        self.nominal_material = cabinet.root_physx_view.get_material_properties().clone()
        self.parameters = torch.zeros((self.num_envs, 5), device=self.device)
        self.final_parameters = torch.zeros_like(self.parameters)
        self.episode_level = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.final_level = self.episode_level.clone()
        if self.streams is None:
            self.seed_streams(self.random_seed)

    def _sample(self, ids):
        angle_max, offset, friction = LEVELS[self.level]
        values = []
        for i in ids:
            rng = self.streams[i]
            row = [math.radians(rng.uniform(0, angle_max)), *rng.uniform(-offset, offset, 3), rng.uniform(1-friction, 1+friction)]
            if self.preset:
                # Always consume the same RNG draws before fixed/stratified overrides.
                if 'angle_deg' in self.preset: row[0] = math.radians(self.preset['angle_deg'])
                if 'angle_range_deg' in self.preset: row[0] = math.radians(rng.uniform(*self.preset['angle_range_deg']))
                if 'offset_xyz' in self.preset: row[1:4] = self.preset['offset_xyz']
                if 'offset_bound' in self.preset: row[1:4] = list(rng.uniform(-self.preset['offset_bound'], self.preset['offset_bound'], 3))
                if 'offset_linf_range' in self.preset:
                    lower, upper = self.preset['offset_linf_range']
                    while True:
                        draw = rng.uniform(-upper, upper, 3)
                        if lower <= np.max(np.abs(draw)) <= upper:
                            row[1:4] = list(draw);break
                if 'friction_scale' in self.preset: row[4] = self.preset['friction_scale']
            values.append(row)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _reset_idx(self, env_ids):
        self._ensure_defaults()
        if self.capture_terminal and len(env_ids):
            old = clone_state(self.get_environment_state())
            if self.final_environment is None:
                self.final_environment = {k: torch.zeros_like(v) for k, v in old.items()}
            for k, value in old.items():
                self.final_environment[k][env_ids] = value[env_ids]
            self.final_parameters[env_ids] = self.parameters[env_ids]
            self.final_level[env_ids] = self.episode_level[env_ids]
        if len(env_ids):
            cabinet = self.scene['cabinet']
            cpu_ids = env_ids.cpu().to(torch.int64)
            samples = self._sample(cpu_ids.tolist())
            self.parameters[env_ids] = samples
            self.episode_level[env_ids] = self.level
            cabinet.data.default_root_state[env_ids] = self.nominal_root[env_ids]
            cabinet.data.default_root_state[env_ids, :3] += samples[:, 1:4]
            cabinet.data.default_joint_pos[env_ids] = self.nominal_joints[env_ids]
            cabinet.data.default_joint_pos[env_ids, door_index(self)] = samples[:, 0]
            material = self.nominal_material.clone()
            material[cpu_ids, :, :2] *= samples[:, 4].cpu()[:, None, None]
            cabinet.root_physx_view.set_material_properties(material, cpu_ids)
        super()._reset_idx(env_ids)
        if len(env_ids):
            # Check actual state after the original scene reset/forward, not only
            # requested config values. Fail early if an event undoes overrides.
            angle = cabinet.data.joint_pos[env_ids, door_index(self)]
            xyz = cabinet.data.root_pos_w[env_ids]-self.scene.env_origins[env_ids]
            expected_xyz = self.nominal_root[env_ids, :3]+samples[:, 1:4]
            actual_mat = cabinet.root_physx_view.get_material_properties()[cpu_ids, :, :2]
            if not torch.allclose(angle, samples[:, 0], atol=1e-5) or not torch.allclose(xyz, expected_xyz, atol=3e-5) or not torch.allclose(actual_mat, material[cpu_ids, :, :2], atol=1e-6):
                raise RuntimeError('Physical randomization was not applied')
            self.physical_reset_checks += len(env_ids)


def create(num_envs, device, seed, level=2, preset=None):
    env = RandomizedDoorEnv(config('B', num_envs, device, seed))
    env.set_randomization(level, preset)
    env.reset(seed=seed)
    env.capture_terminal = True
    return env
