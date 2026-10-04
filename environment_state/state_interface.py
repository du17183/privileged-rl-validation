"""Simulator adapter for signals expected from encoders/contact/pose sensors.

Pose is expressed in the calibrated robot workspace, not global clone origins.
Quaternion is wxyz; orientation is exposed but excluded from the core matrix.
"""
import torch


def from_measurements(gt, start, target):
    angle = gt[:, :1]
    return dict(door_angle=angle, door_angular_velocity=gt[:, 1:2],
                target_angle=target, progress=((angle-start)/(target-start).clamp_min(1e-6)).clamp(0, 1),
                remaining_angle=target-angle, contact_state=gt[:, 9:11],
                handle_position=gt[:, 2:5], handle_orientation=gt[:, 5:9])


def get_environment_state(env):
    gt = env.get_tool_state()
    start = env.episode_start if env.episode_start is not None else gt[:, :1]
    target = env.episode_target if env.episode_target is not None else torch.ones_like(start)
    return from_measurements(gt, start, target)


def clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}
