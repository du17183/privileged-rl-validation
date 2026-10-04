import torch
from door_env.door import robot_observation
from door_env.isaac_env import transition_after_step
from environment_state.state_interface import from_measurements
from environment_state.normalization import encode
from offline_rl.baseline import load_demonstrations
from progress_rl.agent import ExpertView


class MeasuredExpert:
    def __init__(self, path, device, dt, arm, center):
        self.source = ExpertView(load_demonstrations(path, device), 'B', dt)
        self.data = dict(self.source.prepared)
        for key, gt_key in (('robot', 'privileged'), ('next_robot', 'next_privileged')):
            gt = self.data[gt_key]
            state = from_measurements(gt, torch.zeros_like(gt[:, :1]), torch.ones_like(gt[:, :1]))
            self.data[key] = encode(self.data[key], state, arm, center)
        self.device = device

    def __len__(self): return self.data['action'].shape[0]

    def sample(self, count):
        indices = torch.randint(len(self), (count,), device=self.device)
        return {key: value[indices] for key, value in self.data.items()}


def live_observation(env, arm, center):
    return encode(robot_observation(env), env.get_environment_state(), arm, center)


def transition(env, terminated, truncated, arm, center):
    robot, gt, done = transition_after_step(env, terminated, truncated)
    state = {k: v.clone() for k, v in env.get_environment_state().items()}
    if done.any():
        for k in state:
            state[k][done] = env.final_environment[k][done]
    return encode(robot, state, arm, center), gt, done
