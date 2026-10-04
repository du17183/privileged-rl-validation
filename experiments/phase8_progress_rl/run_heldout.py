"""Bounded concurrent independent checkpoint tests, with unique output paths."""
import csv
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
LOG = ROOT/"logs"/"phase8_progress_rl"


def run(arms=("A", "B", "C", "D"), perturb=False):
    from experiments.phase8_progress_rl.launch_fast import processes, TRAIN
    conditions = [(0.0, 0.0, 0.0, 1.0), (0.01, 0.0, 0.0, 1.0)]
    if perturb:
        conditions = [(0.0, a, dy, friction) for a, dy, friction in
                      ((2.5, 0.0, 1.0), (5.0, 0.0, 1.0), (0.0, -0.01, 1.0),
                       (0.0, 0.01, 1.0), (0.0, 0.0, 0.8), (0.0, 0.0, 1.2))]
    pending = [(arm, seed, mode, *condition) for arm in arms for seed in range(5)
               for mode in ("best", "final") for condition in conditions]
    stem = "perturb" if perturb else ("heldout_E" if arms == ("E",) else "heldout")
    status = OUT/f"{stem}_status.csv"
    completion = OUT/f"{stem}_completed.json"
    lock = (OUT/f"{stem}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if status.exists():
        if completion.exists():
            return
        raise RuntimeError("Incomplete heldout status already exists; inspect before resuming")
    running = {}
    with status.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "seed", "mode", "noise", "angle", "dy", "friction", "gpu", "exit_code"))
        stream.flush()
        while pending or running:
            for pid, (proc, job, gpu, logfile) in list(running.items()):
                code = proc.poll()
                if code is not None:
                    writer.writerow((*job, gpu, code))
                    stream.flush()
                    logfile.close()
                    del running[pid]
                    if code:
                        raise RuntimeError(f"Independent test failed: {job}")
            readings = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
            training_counts = {gpu: 0 for gpu in range(8)}
            for pid, args in processes(TRAIN):
                if "--smoke" in args:
                    continue
                try:
                    environment = (Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
                    gpu = int(next(value.split(b"=", 1)[1] for value in environment if value.startswith(b"CUDA_VISIBLE_DEVICES=")))
                    training_counts[gpu] += 1
                except (FileNotFoundError, PermissionError, StopIteration):
                    pass
            for line in readings.splitlines():
                gpu, memory = [int(v.strip()) for v in line.split(",")]
                while pending and memory < 100000 and sum(r[2] == gpu for r in running.values())+training_counts[gpu] < 2:
                    index = next((i for i, j in enumerate(pending) if
                        (ROOT/"checkpoints"/"phase8_progress_rl"/f"P8{j[0]}_seed{j[1]}"/"completed.json").exists()), None)
                    if index is None:
                        break
                    job = pending.pop(index)
                    arm, seed, mode, noise, angle, dy, friction = job
                    tag = f"{arm}_s{seed}_{mode}_n{noise:g}_a{angle:g}_y{dy:g}_f{friction:g}"
                    logfile = (LOG/f"heldout_{tag}.log").open("w")
                    proc = subprocess.Popen([str(ROOT/".venv"/"bin"/"python"), "experiments/phase8_progress_rl/evaluate_frozen.py",
                        "--variant", arm, "--seed", str(seed), "--mode", mode, "--noise", str(noise),
                        "--angle-deg", str(angle), "--cabinet-dy", str(dy), "--friction-scale", str(friction), "--device", "cuda:0"],
                        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=logfile, stderr=subprocess.STDOUT)
                    running[proc.pid] = (proc, job, gpu, logfile)
            time.sleep(3)
    completion.write_text(json.dumps(dict(tests=80 if not perturb and len(arms)==4 else len(arms)*5*2*len(conditions),
                                         all_exits_zero=True)))


if __name__ == "__main__":
    run()
