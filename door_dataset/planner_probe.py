"""Run door planner for one episode and print contact/angle diagnostics."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
import isaaclab_tasks  # noqa: F401
from door_env.door import make_cfg, door_angle, handle_pose, privileged_state
from door_env.isaac_env import create_env
from door_dataset.planner import DoorWaypointPlanner

env = create_env(make_cfg(4, seed=42))
planner = DoorWaypointPlanner(env.num_envs, env.device, env.step_dt)
for step in range(601):
    action = planner.action(env)
    _, reward, terminated, truncated, _ = env.step(action)
    if step % 30 == 0 or (terminated | truncated).any():
        ee=env.scene['ee_frame'].data.target_pos_w[0,0]
        print('TRACE',step,'mode',planner.state[0].item(),'q',door_angle(env)[0,0].item(),
              'handle',handle_pose(env)[0][0].tolist(),'ee',ee.tolist(),
              'contacts',privileged_state(env)[0,-2:].tolist(),
              'gripper',env.scene['robot'].data.joint_pos[0,-2:].tolist(),flush=True)
    if (terminated | truncated).any():
        print('FINAL',env.final_door_angle[:,0].tolist(),flush=True)
        break
env.close()
app.close()
