"""Standalone scientific figures from complete matched runs."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
FIG = OUT/"figures"
LABELS = dict(A="Baseline", B="Progress reward", C="Progress state", D="State + reward", E="Curriculum")


def curves(arm, column, kind="eval"):
    xs, ys = [], []
    for seed in range(5):
        with (OUT/f"{kind}_P8{arm}_seed{seed}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        xs.append(np.array([float(r["env_steps"])/1000 for r in rows]))
        ys.append(np.array([float(r[column]) for r in rows]))
    assert all(np.array_equal(xs[0], x) for x in xs)
    return xs[0], np.stack(ys)


def main():
    FIG.mkdir(exist_ok=True)
    arms = ("A", "B", "C", "D", "E") if (OUT/"eval_P8E_seed4.csv").exists() else ("A", "B", "C", "D")
    for column, name, ylabel, kind in (("success", "protected_success", "Protected success", "eval"),
        ("raw_success", "raw_success", "Success before recovery", "eval"),
        ("progress", "progress", "Mean completion fraction", "eval"),
        ("max_angle", "max_angle", "Mean maximum door angle (rad)", "eval"),
        ("regression_amount", "regression", "Mean cumulative backward angle (rad)", "eval"),
        ("policy_std_mean", "policy_std", "Mean pre-tanh policy std", "diagnostics"),
        ("success", "noise001_success", "Success with pre-tanh noise 0.01", "noise")):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        for arm in arms:
            x, y = curves(arm, column, kind)
            mean = y.mean(0)
            half = 2.7764451051977987*y.std(0, ddof=1)/np.sqrt(5)
            line, = ax.plot(x, mean, label=LABELS[arm])
            low, high = mean-half, mean+half
            if column in ("success", "raw_success", "progress"):
                low, high = low.clip(0, 1), high.clip(0, 1)
            ax.fill_between(x, low, high, alpha=0.12, color=line.get_color())
        ax.set(xlabel="Training environment interactions (thousands)", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        if column in ("success", "raw_success", "progress"):
            ax.set_ylim(-0.02, 1.02)
        fig.savefig(FIG/f"{name}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
