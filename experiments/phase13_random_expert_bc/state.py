"""Phase 13 policy inputs. All poses use the calibrated robot workspace."""
import torch

FIELDS = ("door_angle", "target_angle", "progress", "remaining_angle",
          "contact_state", "handle_position", "handle_orientation")
ENV_DIM = 13
ROBOT_DIM = 26


def pack(state):
    return torch.cat([state[k] for k in FIELDS], dim=-1)


def next_environment(env, done):
    values = pack(env.get_environment_state()).clone()
    if done.any():
        terminal = pack(env.final_environment)
        values[done] = terminal[done]
    return values
