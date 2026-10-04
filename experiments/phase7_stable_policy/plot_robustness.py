"""Plot independent noise-dose and small fixture-shift success rates."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
FIGURES = OUT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def matching(records, arm, mode, noise, angle=0.0, dy=0.0):
    return next(r for r in records if r["variant"] == arm and r["mode"] == mode
                and r["noise_pre_tanh_std"] == noise and r["door_angle_deg"] == angle
                and r["cabinet_dy_m"] == dy)


def main():
    records = json.loads((OUT / "robustness_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    for arm in ("E0", "E1L", "E2L", "E3", "G1", "E1", "E2"):
        items = [matching(records, arm, "best", noise) for noise in (0.0, 0.01, 0.05, 0.2)]
        x = np.arange(4)
        y = [i["success"]["mean"] for i in items]
        ax.plot(x, y, marker="o", label=arm, lw=1.6)
    ax.set_xticks(np.arange(4), ("0", "0.01", "0.05", "0.2"))
    ax.set(xlabel="Added pre-tanh Gaussian action std", ylabel="Held-out success rate",
           ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(ncol=4, frameon=False)
    fig.savefig(FIGURES / "noise_dose_response.png", dpi=180)
    plt.close(fig)

    labels = (("nominal", 0.0, 0.0), ("angle +2.5°", 2.5, 0.0),
              ("angle +5°", 5.0, 0.0), ("handle y +1cm", 0.0, 0.01),
              ("handle y -1cm", 0.0, -0.01))
    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout="constrained")
    x = np.arange(len(labels))
    for j, (arm, mode) in enumerate((("E0", "best"), ("E0", "final"),
                                      ("G1", "best"), ("G1", "final"),
                                      ("E1", "best"), ("E1", "final"))):
        y = [matching(records, arm, mode, 0.0, angle, dy)["success"]["mean"]
             for _, angle, dy in labels]
        ax.plot(x, y, marker="o", label=f"{arm} {mode}", lw=1.5)
    ax.set_xticks(x, [label for label, _, _ in labels], rotation=20, ha="right")
    ax.set(ylabel="Held-out success rate", ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.savefig(FIGURES / "initial_state_shift.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
