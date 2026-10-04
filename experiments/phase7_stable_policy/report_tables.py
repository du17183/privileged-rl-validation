"""Render reproducible Markdown tables from five-seed Phase 7 summaries."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"


def formatted(item, digits=3):
    mean = item["mean"]
    low, high = item["ci95"]
    low, high = max(0.0, low), min(1.0, high)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main():
    main_summary = json.loads((OUT / "phase7_summary.json").read_text())
    exploratory = json.loads((OUT / "exploratory_target_entropy_summary.json").read_text())
    robustness_path = OUT / "robustness_summary.json"
    robustness = json.loads(robustness_path.read_text()) if robustness_path.exists() else None
    lines = ["| 组 | AUC [95% CI] | Best [95% CI] | Final [95% CI] | Gap [95% CI] | Seed SD (Final) | vs E0 AUC Δ / exact p |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("E0", "E1L", "E2L", "E3", "L000", "L001", "L005", "L010", "L050", "G1"):
        s = main_summary[arm]
        difference = s["paired_vs_E0"]["auc"]
        lines.append(f"| {arm} | {formatted(s['auc'])} | {formatted(s['best'])} | "
                     f"{formatted(s['final'])} | {formatted(s['gap'])} | {s['final']['sd']:.3f} | "
                     f"{difference['difference']['mean']:+.3f} / {difference['exact_signflip_p_two_sided']:.4f} |")
    lines += ["", "| 探索性组 | AUC [95% CI] | Best [95% CI] | Final [95% CI] | Gap [95% CI] | vs E0 AUC Δ / exact p |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ("E1", "E2"):
        s = exploratory[arm]
        difference = s["paired_vs_E0"]["auc"]
        lines.append(f"| {arm} | {formatted(s['auc'])} | {formatted(s['best'])} | "
                     f"{formatted(s['final'])} | {formatted(s['gap'])} | "
                     f"{difference['difference']['mean']:+.3f} / {difference['exact_signflip_p_two_sided']:.4f} |")
    if robustness is not None:
        lines += ["", "| 最佳 checkpoint | Deterministic | Noise 0.01 | Noise 0.05 | Noise 0.2 |",
                  "|---|---:|---:|---:|---:|"]
        for arm in ("E0", "E1L", "E2L", "E3", "G1", "E1", "E2"):
            cells = []
            for noise in (0.0, 0.01, 0.05, 0.2):
                row = next(r for r in robustness if r["variant"] == arm and r["mode"] == "best"
                           and r["noise_pre_tanh_std"] == noise and r["door_angle_deg"] == 0.0
                           and r["cabinet_dy_m"] == 0.0)
                cells.append(formatted(row["success"]))
            lines.append(f"| {arm} | {' | '.join(cells)} |")
    lines += ["", "| 组/检查点 | 标称 | 门 +2.5° | 门 +5° | 把手 y −1cm | 把手 y +1cm |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ("E0", "G1", "E1"):
        for mode in ("best", "final"):
            cells = []
            for angle, dy in ((0.0, 0.0), (2.5, 0.0), (5.0, 0.0), (0.0, -0.01), (0.0, 0.01)):
                row = next(r for r in robustness if r["variant"] == arm and r["mode"] == mode
                           and r["noise_pre_tanh_std"] == 0.0 and r["door_angle_deg"] == angle
                           and r["cabinet_dy_m"] == dy)
                cells.append(formatted(row["success"]))
            lines.append(f"| {arm}/{mode} | {' | '.join(cells)} |")
    lines += ["", "| 组/检查点/干预 | 相对标称配对差 [95% CI] | exact p |",
              "|---|---:|---:|"]
    for row in robustness:
        pair = row.get("paired_vs_nominal")
        if pair is None:
            continue
        difference = pair["difference"]
        low, high = difference["ci95"]
        intervention = f"noise={row['noise_pre_tanh_std']},angle={row['door_angle_deg']},dy={row['cabinet_dy_m']}"
        lines.append(f"| {row['variant']}/{row['mode']}/{intervention} | "
                     f"{difference['mean']:+.3f} [{low:+.3f}, {high:+.3f}] | "
                     f"{pair['exact_signflip_p_two_sided']:.4f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
