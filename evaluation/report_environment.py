"""Create a factual environment report from runtime preflight JSON files."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "results/environment.json").read_text(encoding="utf-8"))
    runtime_path = root / "results/drawer_preflight.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else None
    packages = meta["packages"]
    text = [
        "# Isaac Sim / Isaac Lab environment report",
        "",
        f"- Host: `{meta['hostname']}` ({meta['platform']})",
        f"- Python: `{meta['python']}` at `{meta['python_executable']}`",
        f"- Isaac Sim: `{packages['isaacsim']}`",
        f"- Isaac Lab: `{packages['isaaclab']}`; commit `{meta['isaaclab_commit']}`",
        f"- PyTorch: `{packages['torch']}`; CUDA runtime `{meta.get('torch_cuda_runtime')}`; CUDA available `{meta.get('torch_cuda_available')}`",
        "- B300 compute capability: SM 10.3. PyTorch's CUDA 12.8 runtime is retained, while this project's virtual environment uses NVRTC 12.9.86 (supporting SM 10.3). The original NVRTC libraries are saved under `third_party/nvrtc128_backup/`.",
        "- Missing system `libGLU.so.1` is supplied by a project-local Ubuntu `libglu1-mesa` package under `third_party/system_libs/`; `configs/runtime_env.sh` sets its library path and the accepted EULA environment variable.",
        f"- NumPy: `{packages['numpy']}`; HDF5 / h5py: `{packages['h5py']}`; TensorBoard: `{packages['tensorboard']}`",
        f"- glibc: `{meta['glibc']}`",
        "- GPUs and driver:",
        "",
        "```text",
        meta["gpus"],
        "```",
        "",
        "## Panda drawer task",
        "",
        "Isaac Lab `Isaac-Open-Drawer-Franka-IK-Rel-v0` is adapted with separate robot and fixture state, finger-handle contact sensors, a 0.30 m success threshold, and deterministic reset to Panda home plus closed drawer. The cabinet drawer is a prismatic articulation; the action is six relative IK coordinates and one binary gripper command.",
        "",
    ]
    if runtime:
        text.extend([
            f"- Robot observation / fixture GT / action dimensions: {runtime['robot_observation_dim']} / {runtime['privileged_state_dim']} / {runtime['action_dim']}",
            f"- Drawer joint: `{runtime['drawer_joint_name']}`",
            f"- Reset restoration: {runtime['reset_successes']}/{runtime['reset_attempts']} ({runtime['reset_success_rate']:.1%})",
            f"- Mean / p95 reset wall time: {runtime['reset_wall_time_mean_s']:.3f} / {runtime['reset_wall_time_p95_s']:.3f} s for {runtime['num_envs']} parallel environments",
            f"- Filtered left/right contact tensor shapes: `{runtime['contact_filter_shapes']}`",
            f"- Forced automatic reset: all done `{runtime.get('automatic_reset_done_all')}`; closed drawer `{runtime.get('automatic_reset_closed')}`; Panda home `{runtime.get('automatic_reset_home')}`",
            f"- Automatic reset frame mismatch, TCP / handle: {runtime.get('automatic_reset_ee_frame_error_m', float('nan')):.6f} / {runtime.get('automatic_reset_handle_frame_error_m', float('nan')):.6f} m",
            f"- Timeout failure reset: all truncated `{runtime.get('timeout_reset_truncated_all')}` after {runtime.get('timeout_reset_step_count')} steps; closed `{runtime.get('timeout_reset_closed')}`; Panda home `{runtime.get('timeout_reset_home')}`",
            "",
        ])
    else:
        text.append("Runtime contact, articulation, and reset checks have not completed.\n")
    text.extend([
        "## State interface",
        "",
        "`get_privileged_state()` and `get_tool_state()` return drawer position (m), velocity (m/s), handle position (m), handle quaternion (wxyz), and left/right contact flags. Only the B/C training critic and the C upper-bound actor can consume these data. The deployable A/B actor consumes only Panda joint position, joint velocity, gripper state, robot-derived TCP pose, and the episode clock supplied by the reset controller. The frame sensor is refreshed before automatic reset observations are used.",
        "`evaluation/check_actor_boundary.py` verifies that A/B inference returns identical actions with and without a supplied fixture tensor, while C requires that tensor. Its result is saved in `results/actor_gt_isolation.json`.",
        "",
    ])
    report = "\n".join(text)
    for path in (root / "docs/environment_setup.md", root / "results/environment_report.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
