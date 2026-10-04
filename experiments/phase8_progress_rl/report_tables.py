"""Reproducible report tables; preserve unreached seed thresholds."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def fmt(value, bounded=False, digits=3):
    low, high = value["ci95"]
    if bounded:
        low, high = max(0, low), min(1, high)
    return f"{value['mean']:.{digits}f} [{low:.{digits}f},{high:.{digits}f}]"


def main():
    summary = json.loads((OUT/"summary.json").read_text())
    pairs = json.loads((OUT/"paired_tests.json").read_text())["comparisons"]
    heldout = json.loads((OUT/"heldout_summary.json").read_text())
    lines = ["| 组 | 受保护 AUC [95% CI] | Raw AUC [95% CI] | 曲线最高 [95% CI] | Final [95% CI] | 曲线最高−Final [95% CI] | Raw Final [95% CI] | Final seed SD | 在线成功/seed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, s in summary.items():
        lines.append(f"| {arm} | {fmt(s['auc'],True)} | {fmt(s['raw_auc'],True)} | {fmt(s['best'],True)} | {fmt(s['final'],True)} | {fmt(s['gap'],True)} | {fmt(s['raw_final'],True)} | {s['final']['sd']:.3f} | {s['online_successes']['mean']:.1f} |")
    lines += ["", "| 组 | 保存的 best.pt 验证成功率 [95% CI] | best.pt 步数，seed 0–4 |",
              "|---|---:|---|"]
    for arm, s in summary.items():
        steps = " / ".join(str(int(v)) for v in s["selected_best_step"]["per_seed"])
        lines.append(f"| {arm} | {fmt(s['selected_best_success'],True)} | {steps} |")
    lines += ["", "| 组 | 训练交互/seed | 保护与评估交互/seed [95% CI] | 合计交互/seed [95% CI] | 平均回退次数 [95% CI] |",
              "|---|---:|---:|---:|---:|"]
    for arm, s in summary.items():
        lines.append(f"| {arm} | 300,000 | {fmt(s['evaluation_steps'],digits=0)} | {fmt(s['training_plus_evaluation_steps'],digits=0)} | {fmt(s['rollbacks'],digits=1)} |")
    lines += ["", "| 组 | 最终最大门角 rad [95% CI] | 最终门角 rad [95% CI] | 完成比例 [95% CI] | 倒退量 rad [95% CI] | 倒退回合比例 [95% CI] |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, s in summary.items():
        lines.append(f"| {arm} | {fmt(s['max_angle'])} | {fmt(s['final_angle'])} | {fmt(s['progress'],True)} | {fmt(s['regression_amount'])} | {fmt(s['regression_rate'],True)} |")
    lines += ["", "| 组 | 首次 50% 步数，seed 0–4 | 首次 80% | 首次 90% |",
              "|---|---|---|---|"]
    for arm, s in summary.items():
        for kind, label in (("", "受保护"), ("raw_", "回退前")):
            cells = []
            for threshold in (50, 80, 90):
                values = s["threshold_steps"][f"{kind}first_{threshold}_steps"]
                cells.append(" / ".join("未达到" if v is None else str(v) for v in values))
            lines.append(f"| {arm} {label} | {' | '.join(cells)} |")
    lines += ["", "| 配对比较 | 受保护 AUC 差 [95% CI] / exact p | Raw AUC 差 [95% CI] / exact p | Final 差 [95% CI] / exact p |",
              "|---|---:|---:|---:|"]
    for label, tests in pairs.items():
        cells = [f"{fmt(tests[k]['difference'])} / {tests[k]['exact_signflip_p_two_sided']:.4f}" for k in ("auc", "raw_auc", "final")]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += ["", "| 独立复测 | 最佳确定性 [95% CI] | 最终确定性 [95% CI] | 最佳 std=0.01 [95% CI] | 最终 std=0.01 [95% CI] |",
              "|---|---:|---:|---:|---:|"]
    for arm in summary:
        cells = []
        for mode, noise in (("best",0.0), ("final",0.0), ("best",0.01), ("final",0.01)):
            row = next(r for r in heldout if r["variant"]==arm and r["mode"]==mode and r["noise"]==noise
                       and r["initial_angle_deg"]==0 and r["cabinet_dy"]==0 and r["friction_scale"]==1)
            cells.append(fmt(row["success"],True))
        lines.append(f"| {arm} | {' | '.join(cells)} |")
    result = "\n".join(lines)+"\n"
    (OUT/"report_tables.md").write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
