# Isaac Sim / Isaac Lab environment report

- Host: `discover2` (Linux-6.9.12-060912-generic-x86_64-with-glibc2.39)
- Python: `3.11.15` at `/DATA/disk1/home/xiaolong/privileged_rl_validation/.venv/bin/python`
- Isaac Sim: `5.1.0.0`
- Isaac Lab: `0.47.2`; commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`
- PyTorch: `2.7.0+cu128`; CUDA runtime `12.8`; CUDA available `True`
- B300 compute capability: SM 10.3. PyTorch's CUDA 12.8 runtime is retained, while this project's virtual environment uses NVRTC 12.9.86 (supporting SM 10.3). The original NVRTC libraries are saved under `third_party/nvrtc128_backup/`.
- Missing system `libGLU.so.1` is supplied by a project-local Ubuntu `libglu1-mesa` package under `third_party/system_libs/`; `configs/runtime_env.sh` sets its library path and the accepted EULA environment variable.
- NumPy: `1.26.0`; HDF5 / h5py: `3.16.0`; TensorBoard: `2.21.0`
- glibc: `ldd (Ubuntu GLIBC 2.39-0ubuntu8.4) 2.39`
- GPUs and driver:

```text
0, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
1, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
2, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
3, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
4, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
5, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
6, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
7, NVIDIA B300 SXM6 AC, 275040 MiB, 580.82.07
```

## Panda drawer task

Isaac Lab `Isaac-Open-Drawer-Franka-IK-Rel-v0` is adapted with separate robot and fixture state, finger-handle contact sensors, a 0.30 m success threshold, and deterministic reset to Panda home plus closed drawer. The cabinet drawer is a prismatic articulation; the action is six relative IK coordinates and one binary gripper command.

- Robot observation / fixture GT / action dimensions: 26 / 11 / 7
- Drawer joint: `drawer_top_joint`
- Reset restoration: 20/20 (100.0%)
- Mean / p95 reset wall time: 0.015 / 0.006 s for 8 parallel environments
- Filtered left/right contact tensor shapes: `{'left_contact': [8, 1, 1, 3], 'right_contact': [8, 1, 1, 3]}`
- Forced automatic reset: all done `True`; closed drawer `True`; Panda home `True`
- Automatic reset frame mismatch, TCP / handle: 0.000000 / 0.000000 m
- Timeout failure reset: all truncated `True` after 480 steps; closed `True`; Panda home `True`

## State interface

`get_privileged_state()` and `get_tool_state()` return drawer position (m), velocity (m/s), handle position (m), handle quaternion (wxyz), and left/right contact flags. Only the B/C training critic and the C upper-bound actor can consume these data. The deployable A/B actor consumes only Panda joint position, joint velocity, gripper state, robot-derived TCP pose, and the episode clock supplied by the reset controller. The frame sensor is refreshed before automatic reset observations are used.
`evaluation/check_actor_boundary.py` verifies that A/B inference returns identical actions with and without a supplied fixture tensor, while C requires that tensor. Its result is saved in `results/actor_gt_isolation.json`.
