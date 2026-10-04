"""Wait for complete primary runs, summarize, and apply recorded stage gates."""
import json
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
PYTHON = str(ROOT/".venv"/"bin"/"python")


def main():
    markers = [ROOT/"checkpoints"/"phase8_progress_rl"/f"P8{arm}_seed{seed}"/"completed.json"
               for arm in ("A", "B", "C", "D") for seed in range(5)]
    while not all(path.exists() for path in markers):
        time.sleep(30)
    subprocess.run([PYTHON, "experiments/phase8_progress_rl/analyze.py"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "experiments/phase8_progress_rl/plot.py"], cwd=ROOT, check=True)
    from progress_rl.curriculum import stable_d_gate
    summary = json.loads((OUT/"summary.json").read_text())
    gate = stable_d_gate(summary)
    decision = dict(D_stability_gate_passed=gate, curriculum_status="eligible" if gate else "not_run_gate_failed",
                    initial_randomization_status="eligible" if gate else "not_run_gate_failed",
                    drawer_status="conditional_on_door_and_randomization",
                    gate_definition=json.loads((ROOT/"experiments"/"phase8_progress_rl"/"protocol.json").read_text())["curriculum_gate_D"])
    (OUT/"stage_decision.json").write_text(json.dumps(decision, indent=2))
    subprocess.run([PYTHON, "experiments/phase8_progress_rl/run_heldout.py"], cwd=ROOT, check=True)
    subprocess.run([PYTHON, "experiments/phase8_progress_rl/analyze_heldout.py"], cwd=ROOT, check=True)
    print("Primary training and independent checkpoint tests complete; inspect recorded gates.", flush=True)


if __name__ == "__main__":
    main()
