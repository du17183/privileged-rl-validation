"""Print report-ready tables from analyzer CSVs without manual transcription."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase6_stable_online_rl"
VARIANTS = ("A0", "A1", "A2", "B1", "B2", "B3")


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def formatted(row):
    return f"{float(row['mean']):.3f} [{float(row['ci95_low']):.3f}, {float(row['ci95_high']):.3f}]"


def main():
    aggregate = {(r["variant"], r["metric"]): r for r in read("aggregate.csv")}
    heldout = {(r["variant"], r["mode"]): r
               for r in read("heldout_aggregate.csv")}
    paired = {(r["treatment"], r["control"], r["metric"]): r
              for r in read("paired_comparisons.csv")}
    print("| 组 | AUC [95% CI] | 最佳固定评估 | 最终固定评估 | 最佳独立复测 | 最终独立复测 | 独立复测退化 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for variant in VARIANTS:
        cells = [variant, formatted(aggregate[(variant, "auc")]),
                 formatted(aggregate[(variant, "best")]),
                 formatted(aggregate[(variant, "final")]),
                 formatted(heldout[(variant, "best")]),
                 formatted(heldout[(variant, "final")]),
                 formatted(heldout[(variant, "best_minus_final")])]
        print("| " + " | ".join(cells) + " |")
    print("\n| 配对差 | AUC 差 [95% CI] | 精确 p | 最终成功率差 [95% CI] | 精确 p |")
    print("|---|---:|---:|---:|---:|")
    for treatment, control in (("A1", "A0"), ("A2", "A0"), ("B1", "A2"),
                               ("B2", "B1"), ("B3", "B1"),
                               ("B3", "A0"), ("B3", "A2")):
        auc, final = paired[(treatment, control, "auc")], paired[(treatment, control, "final")]
        def diff(r):
            return (f"{float(r['mean_difference']):+.3f} "
                    f"[{float(r['ci95_low']):+.3f}, {float(r['ci95_high']):+.3f}]")
        print(f"| {treatment} − {control} | {diff(auc)} | {float(auc['exact_two_sided_signflip_p']):.4f} | "
              f"{diff(final)} | {float(final['exact_two_sided_signflip_p']):.4f} |")
    print("\n| 组 | 初始接触率 | 首次约10k接触率 | 末期接触率 |")
    print("|---|---:|---:|---:|")
    for variant in VARIANTS:
        print(f"| {variant} | {formatted(aggregate[(variant, 'initial_contact')])} | "
              f"{formatted(aggregate[(variant, 'first_eval_contact')])} | "
              f"{formatted(aggregate[(variant, 'final_contact')])} |")


if __name__ == "__main__":
    main()
