# Drawer expert dataset

- File: `/DATA/disk1/home/xiaolong/privileged_rl_validation/datasets/drawer_expert_1000.h5`
- Generation: Cartesian waypoint planner, differential IK, Panda gripper control
- Successful trajectories: 1000
- Total attempted episodes: 1012
- Collection success rate: 0.988
- Mean / median / min / max episode length: 300.6 / 301 / 289 / 324 steps
- Total saved transitions: 300645
- Total collection interactions including failed/partial episodes: 308256
- Robot observation dimension: 26
- Privileged state dimension: 11
- Action dimension: 7
- Drawer displacement range in saved states: 0.0000–0.3000 m
- Contact flag activation, left / right: 0.516 / 0.513
- Source runs:
  - 500 trajectories, action noise std 0.0, seed 100, 512 attempts
  - 500 trajectories, action noise std 0.08, seed 110, 500 attempts

Each trajectory contains `observation`, `state`, `action`, `reward`, `next_observation`, `next_state`, `done`, `terminated`, and `truncated`. The `state` columns are drawer position, drawer velocity, handle position, handle quaternion (wxyz), left contact, and right contact. Only successful trajectories are saved; collection success rate uses all attempts.
