"""Completion gate for budgets, paired data, heldout tests and baseline integrity."""
import hashlib
import json
from pathlib import Path
from experiments.phase8_progress_rl.analyze import read
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"


def main():
    expected = json.loads((OUT/"preflight_audit.json").read_text())["immutable_sha256"]
    for name, digest in expected.items():
        h = hashlib.sha256()
        with (ROOT/name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                h.update(block)
        assert h.hexdigest() == digest, name
    for arm in ("A", "B", "C", "D"):
        for seed in range(5):
            run = f"P8{arm}_seed{seed}"
            rows = read(OUT/f"eval_{run}.csv")
            steps = [int(r["env_steps"]) for r in rows]
            assert len(rows)==31 and steps[0]==0 and steps[-1]==300000
            assert all(b>a for a,b in zip(steps,steps[1:]))
            assert len(read(OUT/f"noise_{run}.csv")) == 31
            assert len(read(OUT/f"diagnostics_{run}.csv")) == 30
            assert (ROOT/"checkpoints"/"phase8_progress_rl"/run/"completed.json").exists()
    launches = read(OUT/"launch_status.csv")
    assert len(launches)==20 and all(r["exit_code"]=="0" for r in launches)
    heldout = read(OUT/"heldout_status.csv")
    assert len(heldout)==80 and all(r["exit_code"]=="0" for r in heldout)
    records = json.loads((OUT/"heldout_summary.json").read_text())
    assert all(len(r["success"]["per_seed"])==5 for r in records)
    manifest = dict(primary_runs=20, training_steps_per_run=300000, independent_tests=80,
                    independent_episodes=5120, input_hashes_preserved=expected,
                    stage_decision=json.loads((OUT/"stage_decision.json").read_text()))
    (OUT/"completion_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
