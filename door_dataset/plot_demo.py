"""Plot one physical expert trajectory for environment/data audit."""
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
with h5py.File(ROOT / "door_dataset/door_expert_1000.h5", "r") as h5:
    traj = h5["traj_00000"]
    state = traj["state"][:]
    robot = traj["observation"][:]
    n = len(state)
    terminal_angle = float(traj["next_state"][-1, 0])

time = np.arange(n) / 60
distance = np.linalg.norm(robot[:, 18:21] - state[:, 2:5], axis=1)
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1]})
axes[0].plot(np.arange(n + 1) / 60, np.r_[state[:, 0], terminal_angle],
             color="#1769aa", linewidth=2)
axes[0].axhline(1.0, color="#d97706", linestyle="--", label="Success threshold")
axes[0].set_ylabel("Right-door angle (rad)")
axes[0].legend(frameon=False)
axes[1].plot(time, distance, color="#555555")
axes[1].set_ylabel("TCP-handle distance (m)")
axes[2].step(time, state[:, 9], where="post", label="Left finger", color="#2e8b57")
axes[2].step(time, state[:, 10], where="post", label="Right finger", color="#b34b2a", alpha=0.7)
axes[2].set_ylim(-0.1, 1.1)
axes[2].set_ylabel("Contact flag")
axes[2].set_xlabel("Episode time (s)")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(ROOT / "results/door/expert_trajectory.png", dpi=180)
plt.close(fig)
print(f"PLOT length={n} terminal_angle={terminal_angle:.4f}")
