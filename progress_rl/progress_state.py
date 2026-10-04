"""Fixture measurements available during both learning and deployment."""
import torch

FEATURE_NAMES = ("door_angle", "door_angular_velocity", "target_angle", "progress", "contact_state")
FULL_TARGET = 1.0


def from_measurements(angle, velocity, contact, start_angle=0.0, target_angle=FULL_TARGET):
    start = torch.as_tensor(start_angle, device=angle.device, dtype=angle.dtype).expand_as(angle)
    target = torch.as_tensor(target_angle, device=angle.device, dtype=angle.dtype).expand_as(angle)
    if bool((target <= start).any()):
        raise ValueError("Target angle must exceed the episode start angle")
    return dict(door_angle=angle, door_angular_velocity=velocity, target_angle=target,
                progress=((angle-start)/(target-start)).clamp(0.0, 1.0),
                remaining_angle=(target-angle).clamp_min(0), contact_state=contact.float())


def features(state):
    # radians; angular velocity / 5 rad/s; binary bilateral handle contact.
    return torch.cat((state["door_angle"], state["door_angular_velocity"]/5.0,
                      state["target_angle"], state["progress"], state["contact_state"]), dim=-1).float()


def from_stored_state(gt, target=FULL_TARGET):
    contact = (gt[:, 9:11] > 0.5).all(dim=-1, keepdim=True)
    return from_measurements(gt[:, :1], gt[:, 1:2], contact, target_angle=target)


def observation(robot, gt, variant, target=FULL_TARGET):
    return torch.cat((robot, features(from_stored_state(gt, target))), dim=-1) if variant in ("C", "D", "E") else robot
