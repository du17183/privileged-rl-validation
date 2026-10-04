"""Isaac Lab Panda right-door task with a deployable/fixture state boundary."""

import torch
from isaaclab.utils.math import quat_apply


SUCCESS_ANGLE_RAD = 1.0
CONTACT_FORCE_N = 0.5
# Sektion door_right_nob_link visual centre, measured relative to its rigid-body
# frame at the closed pose. The body frame rotates about the cabinet hinge.
HANDLE_LOCAL_OFFSET = (0.13, -0.35, 0.185)


def door_index(env):
    return env.scene["cabinet"].joint_names.index("door_right_joint")


def door_angle(env):
    i = door_index(env)
    return env.scene["cabinet"].data.joint_pos[:, i:i + 1]


def door_velocity(env):
    i = door_index(env)
    return env.scene["cabinet"].data.joint_vel[:, i:i + 1]


def handle_pose(env):
    cabinet = env.scene["cabinet"]
    i = cabinet.body_names.index("door_right_nob_link")
    pos = cabinet.data.body_pos_w[:, i]
    quat = cabinet.data.body_quat_w[:, i]
    offset = torch.tensor(HANDLE_LOCAL_OFFSET, device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + quat_apply(quat, offset), quat


def contact_force(env, name):
    matrix = env.scene[name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"Filtered contact forces unavailable: {name}")
    force = torch.linalg.vector_norm(matrix, dim=-1).reshape(env.num_envs, -1).amax(dim=-1, keepdim=True)
    return torch.where(env.episode_length_buf[:, None] == 0, torch.zeros_like(force), force)


def robot_observation(env):
    robot = env.scene["robot"].data
    ee = env.scene["ee_frame"].data
    progress = (env.episode_length_buf.float().unsqueeze(-1) / float(env.max_episode_length)).clamp(0, 1)
    return torch.cat((robot.joint_pos, robot.joint_vel,
                      ee.target_pos_w[:, 0] - env.scene.env_origins,
                      ee.target_quat_w[:, 0], progress), dim=-1).float()


def privileged_state(env):
    pos, quat = handle_pose(env)
    left = (contact_force(env, "door_left_contact") > CONTACT_FORCE_N).float()
    right = (contact_force(env, "door_right_contact") > CONTACT_FORCE_N).float()
    return torch.cat((door_angle(env), door_velocity(env), pos - env.scene.env_origins,
                      quat, left, right), dim=-1).float()


def success(env):
    return door_angle(env).squeeze(-1) > SUCCESS_ANGLE_RAD


def door_reward(env):
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0]
    handle_pos, _ = handle_pose(env)
    reach = 1.0 - torch.tanh(5.0 * torch.linalg.vector_norm(ee_pos - handle_pos, dim=-1))
    grasp = ((contact_force(env, "door_left_contact") > CONTACT_FORCE_N) &
             (contact_force(env, "door_right_contact") > CONTACT_FORCE_N)).float().squeeze(-1)
    return reach + 0.5 * grasp + 20.0 * torch.clamp(door_velocity(env).squeeze(-1), min=0.0) + 600.0 * success(env).float()


def make_cfg(num_envs=32, device="cuda:0", seed=0):
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg("Isaac-Open-Drawer-Franka-IK-Rel-v0", device=device, num_envs=num_envs)
    cfg.seed = seed
    cfg.observations.policy.cabinet_joint_pos = None
    cfg.observations.policy.cabinet_joint_vel = None
    cfg.observations.policy.rel_ee_drawer_distance = None
    cfg.observations.policy.enable_corruption = False
    cfg.scene.cabinet_frame.debug_vis = False
    cfg.events.reset_robot_joints = None
    cfg.scene.cabinet.init_state.pos = (0.9, 0.0, 0.4)
    cfg.scene.cabinet.spawn.usd_path = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "assets" / "panda_door_cabinet.usd")
    cfg.actions.arm_action.scale = (0.05, 0.05, 0.05, 0.3, 0.3, 0.3)
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.cabinet.spawn.activate_contact_sensors = True
    for name, finger in (("door_left_contact", "panda_leftfinger"),
                         ("door_right_contact", "panda_rightfinger")):
        setattr(cfg.scene, name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger}", update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/door_right_nob_link"],
        ))
    for name in list(cfg.rewards.__dict__):
        if not name.startswith("_"):
            setattr(cfg.rewards, name, None)
    cfg.rewards.door = RewTerm(func=door_reward, weight=1.0)
    cfg.terminations.success = DoneTerm(func=success)
    cfg.episode_length_s = 10.0
    return cfg
