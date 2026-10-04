"""One-time audited recovery of accidentally duplicated G1 seed1/2 file writes.

The long-running original learners were not stopped. Their late checkpoints and
in-memory replay remain untouched. Preserve raw CSVs before filtering the two
short-lived duplicate learners' nonmonotonic early rows. Run only after both
original evaluations have reached 300k and the batch launcher has exited.
"""

import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
CHECKPOINTS = ROOT / "checkpoints" / "phase7_stable_policy"
LOGS = ROOT / "logs" / "phase7_stable_policy"
AUDIT = OUT / "duplicate_launch_audit"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    temporary = path.with_suffix(".repair_tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def monotonic(rows):
    kept, rejected = [], []
    last = -1
    for row in rows:
        step = int(row["env_steps"])
        if step > last:
            kept.append(row)
            last = step
        else:
            rejected.append(row)
    return kept, rejected


def main():
    if AUDIT.exists():
        raise RuntimeError("Duplicate launch audit directory already exists; refusing repeated repair")
    # Validate every condition before editing any file.
    plans = []
    for seed in (1, 2):
        run = f"P7G1_seed{seed}"
        for kind, expected in (("eval", 31), ("diagnostics", 30)):
            path = OUT / f"{kind}_{run}.csv"
            fields, original = read_csv(path)
            kept, rejected = monotonic(original)
            if len(kept) != expected or int(kept[-1]["env_steps"]) != 300000 or not rejected:
                raise RuntimeError(f"Unexpected {path}: kept={len(kept)}, rejected={len(rejected)}")
            plans.append((path, fields, original, kept, rejected))
    status_path = OUT / "launch_status.csv"
    status_fields, status_rows = read_csv(status_path)
    if len(status_rows) != 40:
        raise RuntimeError(f"Expected 40 launcher records, got {len(status_rows)}")
    canceled = []
    for seed in (1, 2):
        matches = [r for r in status_rows if r["variant"] == "P7G1" and int(r["seed"]) == seed]
        if len(matches) != 1 or matches[0]["exit_code"] in ("0", "already_complete"):
            raise RuntimeError(f"Expected one canceled duplicate status for G1 seed{seed}: {matches}")
        canceled.append(dict(matches[0]))
    best_steps = {}
    for seed in (1, 2):
        eval_kept = next(kept for path, _, _, kept, _ in plans
                         if path.name == f"eval_P7G1_seed{seed}.csv")
        best = max(eval_kept, key=lambda r: float(r["success_rate"]))
        step = int(best["env_steps"])
        if step <= 20000:
            raise RuntimeError(f"Best checkpoint could have been overwritten: G1 seed{seed} step{step}")
        if not (CHECKPOINTS / f"P7G1_seed{seed}" / f"step_{step}.pt").exists():
            raise RuntimeError(f"Selected best checkpoint missing: G1 seed{seed} step{step}")
        best_steps[seed] = step
    AUDIT.mkdir(parents=True)
    for path, fields, original, kept, rejected in plans:
        shutil.copy2(path, AUDIT / f"{path.stem}.duplicate_raw.csv")
        write_csv(path, fields, kept)
    shutil.copy2(status_path, AUDIT / "launch_status.duplicate_raw.csv")
    for row in status_rows:
        if row["variant"] == "P7G1" and int(row["seed"]) in (1, 2):
            row["exit_code"] = "already_complete"
    write_csv(status_path, status_fields, status_rows)
    for seed, step in best_steps.items():
        base = CHECKPOINTS / f"P7G1_seed{seed}"
        shutil.copy2(base / "best.pt", AUDIT / f"P7G1_seed{seed}.overwritten_best.pt")
        shutil.copy2(base / f"step_{step}.pt", base / "best.pt")
        for early_step in (0, 10016, 20000):
            suspect = base / f"step_{early_step}.pt"
            if suspect.exists():
                shutil.copy2(suspect, AUDIT / f"P7G1_seed{seed}.suspect_step_{early_step}.pt")
        text_log = LOGS / f"P7G1_seed{seed}.log"
        if text_log.exists():
            shutil.copy2(text_log, AUDIT / f"P7G1_seed{seed}.interleaved.log")
    for seed, duplicate_pid in ((1, 2820478), (2, 2820483)):
        events = list((LOGS / f"P7G1_seed{seed}").glob(f"*{duplicate_pid}*"))
        if len(events) != 1:
            raise RuntimeError(f"Expected one duplicate TensorBoard event file for G1 seed{seed}")
        shutil.move(str(events[0]), str(AUDIT / events[0].name))
    details = {
        "reason": "Batch launcher started G1 seed1/2 while identical early-start learners were still running.",
        "canceled_duplicate_pids": [2820478, 2820483],
        "original_learner_pids": [2577349, 2577746],
        "canceled_launcher_rows": canceled,
        "best_checkpoint_steps_rebuilt": best_steps,
        "csv_removed_rows": {path.name: len(rejected) for path, _, _, _, rejected in plans},
        "early_checkpoint_caveat": "Duplicate learners may have overwritten step_0/10016/20000.pt; none is selected as best or final. Raw CSVs and pre-repair best.pt are preserved here.",
    }
    (AUDIT / "repair.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
