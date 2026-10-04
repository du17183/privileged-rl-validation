# Drawer expert dataset

- File: `/DATA/disk1/home/xiaolong/privileged_rl_validation/datasets/drawer_smoke.h5`
- Generation: Cartesian waypoint planner, differential IK, Panda gripper control
- Successful trajectories: 10
- Total attempted episodes: 16
- Collection success rate: 0.625
- Mean / median / min / max episode length: 300.8 / 301 / 300 / 301 steps
- Total saved transitions: 3008
- Robot observation dimension: 25
- Privileged state dimension: 11
- Action dimension: 7
- Drawer displacement range in saved states: 0.0000–0.2998 m
- Contact flag activation, left / right: 0.522 / 0.519

Each trajectory contains `observation`, `state`, `action`, `reward`, `next_observation`, `next_state`, `done`, `terminated`, and `truncated`. The `state` columns are drawer position, drawer velocity, handle position, handle quaternion (wxyz), left contact, and right contact. Only successful trajectories are saved; collection success rate uses all attempts.
