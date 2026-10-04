"""Fixed, versioned units; no statistics fitted to evaluation data."""
import torch

FEATURES = {
    'A': (),
    'B': ('door_angle', 'target_angle', 'progress'),
    'C': ('door_angle', 'target_angle', 'progress', 'contact_state'),
    'D': ('door_angle', 'target_angle', 'progress', 'contact_state', 'handle_position'),
}
DIMS = {'A': 26, 'B': 29, 'C': 31, 'D': 34}
# Coordinates are metres in the robot workspace. This constant is replaced by
# the read-only expert's mean initial pose in protocol.json before training.


def encode(robot, state, arm, handle_center):
    fields = []
    for name in FEATURES[arm]:
        value = state[name]
        if name == 'handle_position':
            value = (value-torch.as_tensor(handle_center, device=value.device, dtype=value.dtype))/0.1
        fields.append(value)
    result = torch.cat((robot, *fields), -1) if fields else robot
    if result.shape[-1] != DIMS[arm] or not torch.isfinite(result).all():
        raise ValueError('Invalid measured observation')
    return result
