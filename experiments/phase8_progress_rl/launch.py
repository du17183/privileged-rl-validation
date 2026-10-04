"""Unique run folders and a nonblocking GPU queue; other jobs are respected."""
import csv
import fcntl
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
LOG = ROOT/"logs"/"phase8_progress_rl"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (OUT/"launch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [(arm, seed) for seed in range(5) for arm in ("A", "B", "C", "D")]
    status = OUT/"launch_status.csv"
    if status.exists():
        raise RuntimeError("Existing Phase 8 launch status; inspect before resuming")
    streams = []
    running = {}
    with status.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "seed", "gpu", "exit_code", "start_utc", "end_utc"))
        stream.flush()
        while jobs or running:
            for gpu, (process, arm, seed, start, logfile) in list(running.items()):
                code = process.poll()
                if code is not None:
                    writer.writerow((arm, seed, gpu, code, start, datetime.now(timezone.utc).isoformat()))
                    stream.flush()
                    logfile.close()
                    del running[gpu]
                    print(f"finished P8{arm}_seed{seed} gpu={gpu} exit={code}", flush=True)
                    if code:
                        raise RuntimeError("Failed run; stop queue and inspect logs")
            query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            readings = [tuple(int(value.strip()) for value in line.split(",")) for line in query.splitlines()]
            # B300 has 275 GB. Prefer low utilization; one Phase 8 learner/evaluator pair per GPU.
            free = [gpu for gpu, memory, utilization in sorted(readings, key=lambda r: (r[2], r[1]))
                    if memory < 80000 and utilization < 60 and gpu not in running]
            for gpu in free:
                if not jobs:
                    break
                arm, seed = jobs.pop(0)
                run = f"P8{arm}_seed{seed}"
                if (ROOT/"checkpoints"/"phase8_progress_rl"/run).exists():
                    raise RuntimeError(f"Run already exists: {run}")
                logfile = (LOG/f"{run}.log").open("w")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                process = subprocess.Popen([str(ROOT/".venv"/"bin"/"python"), "experiments/phase8_progress_rl/train.py",
                    "--variant", arm, "--seed", str(seed), "--steps", "300000", "--num-envs", "32", "--device", "cuda:0"],
                    cwd=ROOT, env=environment, stdout=logfile, stderr=subprocess.STDOUT)
                running[gpu] = (process, arm, seed, datetime.now(timezone.utc).isoformat(), logfile)
                print(f"started {run} gpu={gpu} pid={process.pid}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
