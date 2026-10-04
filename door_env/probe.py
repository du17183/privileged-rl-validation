"""One-shot fixture geometry probe for the Panda door task."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
import isaaclab_tasks  # noqa: F401
from envs.drawer import make_cfg
from envs.isaac_env import create_env

cfg = make_cfg(2, device="cuda:0", seed=123)
env = create_env(cfg)
cab = env.scene["cabinet"]
print("JOINT_NAMES", cab.joint_names, flush=True)
print("BODY_NAMES", cab.body_names, flush=True)
print("LIMITS", cab.data.soft_joint_pos_limits[0].tolist(), flush=True)
print("CAB_ROOT", cab.data.root_pos_w[0].tolist(), flush=True)
print("EE", env.scene["ee_frame"].data.target_pos_w[0, 0].tolist(), env.scene["ee_frame"].data.target_quat_w[0, 0].tolist(), flush=True)
for name in cab.body_names:
    if "door" in name or "handle" in name:
        idx = cab.body_names.index(name)
        print("BODY", name, cab.data.body_pos_w[0, idx].tolist(), cab.data.body_quat_w[0, idx].tolist(), flush=True)
print("ROB_HOME", env.scene["robot"].data.joint_pos[0].tolist(), flush=True)
env.close()
app.close()
