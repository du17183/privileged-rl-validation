"""Physical fixture checks: 20 parallel success and timeout automatic resets."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import json
from pathlib import Path
import torch
import isaaclab_tasks  # noqa: F401
from door_env.door import make_cfg, door_angle, privileged_state
from door_env.isaac_env import create_env

env = create_env(make_cfg(20, seed=321))
path = Path(__file__).resolve().parents[1] / "results/door/reset_validation.json"
action = torch.zeros((env.num_envs, 7), device=env.device)
action[:, -1] = 1.0
robot = env.scene["robot"]
home = robot.data.default_joint_pos.clone()

def restored():
    q = door_angle(env).abs() < 1e-5
    joints = (robot.data.joint_pos - home).abs().amax(dim=-1) < 1e-4
    contacts = privileged_state(env)[:, -2:].sum(dim=-1) == 0
    return q.squeeze(-1) & joints & contacts

try:
    cab = env.scene["cabinet"]
    j = cab.joint_names.index("door_right_joint")
    q = cab.data.default_joint_pos.clone()
    v = torch.zeros_like(q)
    q[:, j] = 1.2
    cab.write_joint_state_to_sim(q, v)
    env.scene.write_data_to_sim()
    env.sim.forward()
    _, _, term, trunc, _ = env.step(action)
    forced_success = int((term & ~trunc & restored()).sum())
    success_angle = env.final_door_angle[:, 0].tolist()

    env.reset(seed=322)
    env.episode_length_buf[:] = env.max_episode_length - 1
    _, _, term, trunc, _ = env.step(action)
    forced_timeout = int((trunc & ~term & restored()).sum())
    result = {"num_envs": env.num_envs, "success_reset_pass": forced_success,
              "timeout_reset_pass": forced_timeout,
              "success_final_angles_rad": success_angle,
              "closed_threshold_rad": 1e-5, "home_tolerance_rad": 1e-4}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESET_VALIDATION",json.dumps(result),flush=True)
    if forced_success != env.num_envs or forced_timeout != env.num_envs:
        raise RuntimeError("Automatic fixture reset validation failed")
finally:
    env.close()
    app.close()
