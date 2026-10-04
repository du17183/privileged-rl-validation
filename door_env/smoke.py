"""Validate task construction, state interface, and automatic reset."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
import isaaclab_tasks  # noqa: F401
from door_env.door import make_cfg, robot_observation, privileged_state, door_angle, handle_pose
from door_env.isaac_env import create_env

env = create_env(make_cfg(4, seed=41))
print('INITIAL', robot_observation(env).shape, privileged_state(env).shape,
      door_angle(env)[:, 0].tolist(), handle_pose(env)[0][0].tolist(),flush=True)
rob = env.scene['robot']
print('EE_QUAT',env.scene['ee_frame'].data.target_quat_w[0,0].tolist(),flush=True)
print('DRAWER_HANDLE_QUAT',env.scene['cabinet_frame'].data.target_quat_w[0,0].tolist(),flush=True)
for name in ('panda_leftfinger','panda_rightfinger','panda_hand'):
    idx=rob.body_names.index(name)
    print('ROBOT_BODY',name,rob.data.body_pos_w[0,idx].tolist(),flush=True)
action = torch.zeros((env.num_envs, 7), device=env.device)
action[:, -1] = 1.0
for t in range(610):
    _, reward, terminated, truncated, _ = env.step(action)
    if (terminated | truncated).any():
        print('RESET',t,'FINAL',env.final_door_angle[:,0].tolist(),
              'AFTER',door_angle(env)[:,0].tolist(),
              'ROBOT',robot_observation(env)[0,:9].tolist(),flush=True)
        break
print('FORCES',privileged_state(env)[:,-2:].tolist(),flush=True)
env.close()
app.close()
