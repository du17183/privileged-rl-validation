"""Panda drawer task adapter for Isaac Lab 2.3.0.

Only ``robot_observation`` may enter a deployment actor in variants A/B.
``privileged_state`` is recorded separately and is never part of the A/B policy group.
"""

import torch


SUCCESS_DISTANCE_M = 0.30
CONTACT_FORCE_N = 0.5


def drawer_index(env):
    return env.scene["cabinet"].joint_names.index("drawer_top_joint")


def drawer_position(env):
    return env.scene["cabinet"].data.joint_pos[:, drawer_index(env):drawer_index(env) + 1]


def drawer_velocity(env):
    return env.scene["cabinet"].data.joint_vel[:, drawer_index(env):drawer_index(env) + 1]


def contact_force(env, name):
    matrix = env.scene[name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"Filtered contact forces unavailable for {name}")
    force = torch.linalg.vector_norm(matrix, dim=-1).reshape(env.num_envs, -1).amax(dim=-1, keepdim=True)
    # PhysX may still report the preceding episode's impulse immediately
    # after reset, before the first new physics step.
    return torch.where(env.episode_length_buf[:, None] == 0, torch.zeros_like(force), force)


def robot_observation(env):
    """Deployable signals: robot proprioception, TCP pose, and reset-controller clock.

    The clock is elapsed episode time, available from the automatic reset
    controller on a physical robot. It never reads the drawer or fixture.
    """
    robot = env.scene["robot"].data
    ee = env.scene["ee_frame"].data
    episode_progress = (
        env.episode_length_buf.float().unsqueeze(-1) / float(env.max_episode_length)
    ).clamp(0.0, 1.0)
    return torch.cat(
        (
            robot.joint_pos,
            robot.joint_vel,
            ee.target_pos_w[:, 0, :] - env.scene.env_origins,
            ee.target_quat_w[:, 0, :],
            episode_progress,
        ), dim=-1,
    ).float()


def privileged_state(env):
    """Simulated fixture: drawer displacement/speed, handle pose, two contact flags."""
    handle = env.scene["cabinet_frame"].data
    left = (contact_force(env, "left_contact") > CONTACT_FORCE_N).float()
    right = (contact_force(env, "right_contact") > CONTACT_FORCE_N).float()
    return torch.cat(
        (
            drawer_position(env),
            drawer_velocity(env),
            handle.target_pos_w[:, 0, :] - env.scene.env_origins,
            handle.target_quat_w[:, 0, :],
            left, right,
        ), dim=-1,
    ).float()


def success(env):
    return (drawer_position(env).squeeze(-1) > SUCCESS_DISTANCE_M)


def drawer_reward(env):
    tcp = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    handle = env.scene["cabinet_frame"].data.target_pos_w[:, 0, :]
    reach = 1.0 - torch.tanh(5.0 * torch.linalg.vector_norm(tcp - handle, dim=-1))
    grasp = (
        (contact_force(env, "left_contact") > CONTACT_FORCE_N)
        & (contact_force(env, "right_contact") > CONTACT_FORCE_N)
    ).float().squeeze(-1)
    # Isaac Lab RewardManager multiplies each reward term by env.step_dt.
    # Integrated positive velocity therefore measures positive drawer travel.
    progress_rate = torch.clamp(drawer_velocity(env).squeeze(-1), min=0.0)
    # The manager multiplies this by 1/60 s; 600 yields a terminal bonus of 10.
    # Opening must outscore hovering near the handle for the full 8 s timeout.
    return reach + 0.5 * grasp + 20.0 * progress_rate + 600.0 * success(env).float()


def make_cfg(num_envs=32, device="cuda:0", seed=0):
    """Construct one common task configuration for all A/B/C variants."""
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg("Isaac-Open-Drawer-Franka-IK-Rel-v0", device=device, num_envs=num_envs)
    cfg.seed = seed
    # The stock policy includes cabinet GT. Strip it even though the trainer builds
    # robot-only input directly, so accidental use of obs['policy'] cannot leak GT.
    cfg.observations.policy.cabinet_joint_pos = None
    cfg.observations.policy.cabinet_joint_vel = None
    cfg.observations.policy.rel_ee_drawer_distance = None
    cfg.observations.policy.enable_corruption = False
    cfg.scene.cabinet_frame.debug_vis = False
    # The fixture protocol starts every episode from the same Panda home pose.
    # The stock task adds random joint offsets at reset, so disable that event.
    cfg.events.reset_robot_joints = None
    cfg.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.3, 0.3, 0.3)
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.cabinet.spawn.activate_contact_sensors = True
    cfg.scene.left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/drawer_handle_top"],
    )
    cfg.scene.right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/drawer_handle_top"],
    )
    for name in list(cfg.rewards.__dict__):
        if not name.startswith("_"):
            setattr(cfg.rewards, name, None)
    cfg.rewards.drawer = RewTerm(func=drawer_reward, weight=1.0)
    cfg.terminations.success = DoneTerm(func=success)
    return cfg
