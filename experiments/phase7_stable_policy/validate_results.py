"""Check completed budgets, independent tests and preserved baseline inputs."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase7_stable_policy"
ARMS = ("E1", "E2", "E3", "L001", "L005", "L010", "L050", "G1", "E1L", "E2L")
IMMUTABLE = {
    "door_dataset/door_expert_1000.h5": "a17c857eb760517bd0d59e266c4f008c5fcf980faf8b8052d4343b57c7b069a8",
    "door_env/door.py": "b9e710e6195d5cfba46fffc23afd8896871e2b70a6c1ae6d8185625dbe68fb69",
    "door_env/isaac_env.py": "d5efaf1041361547bc6e865ffebca801556eebfcf4306b1bcb3001aad3b80fa6",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    runs = []
    for arm in ARMS:
        for seed in range(5):
            label = f"P7{arm}_seed{seed}"
            rows = read_csv(OUT / f"eval_{label}.csv")
            steps = [int(row["env_steps"]) for row in rows]
            assert len(steps) == 31 and steps[0] == 0 and steps[-1] == 300000, label
            assert all(b > a for a, b in zip(steps, steps[1:])), label
            diagnostic = read_csv(OUT / f"diagnostics_{label}.csv")
            assert int(diagnostic[-1]["env_steps"]) == 300000, label
            checkpoints = ROOT / "checkpoints" / "phase7_stable_policy" / label
            assert (checkpoints / "best.pt").is_file() and (checkpoints / "step_300000.pt").is_file(), label
            runs.append({"run": label, "training_steps": steps[-1], "eval_points": len(rows)})
    statuses = read_csv(OUT / "launch_status.csv") + read_csv(OUT / "low_alpha_launch_status.csv")
    assert len(statuses) == 50 and all(r["exit_code"] in ("0", "already_complete") for r in statuses)
    independent_status = read_csv(OUT / "robustness_status.csv")
    assert len(independent_status) == 275 and all(r["exit_code"] == "0" for r in independent_status)
    independent_files = list((OUT / "robustness").glob("*.csv"))
    assert len(independent_files) == 275
    episodes = 0
    for path in independent_files:
        rows = read_csv(path)
        assert len(rows) == 1, path
        # Field name is recorded by the independent Isaac evaluator.
        row = rows[0]
        episodes += int(row["episodes"])
    assert episodes == 17600
    robustness = json.loads((OUT / "robustness_summary.json").read_text())
    assert len(robustness) == 55
    for row in robustness:
        assert len(row["success"]["per_seed"]) == 5
    hashes = {}
    for name, expected in IMMUTABLE.items():
        digest = hashlib.sha256()
        with (ROOT / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        assert actual == expected, name
        hashes[name] = actual
    report = (ROOT / "docs" / "phase7_stable_policy_report.md").read_text(encoding="utf-8")
    assert "正在进行" not in report and "待全部实验" not in report
    manifest = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid_new_runs": len(runs), "training_steps_per_run": 300000,
        "reused_control_curves": 10,
        "independent_tests": len(independent_files), "independent_episodes": episodes,
        "immutable_sha256": hashes, "runs": runs,
        "audit_note": "G1 seed 1/2 duplicate launch rows and early checkpoints retained separately; formal CSV repaired.",
        "learning_goal_achieved": False,
    }
    (OUT / "completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
