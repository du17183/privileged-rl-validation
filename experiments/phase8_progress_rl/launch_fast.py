"""Adopt active Phase 8 runs and increase concurrency without restarting them."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
LOG = ROOT/"logs"/"phase8_progress_rl"
TRAIN = "experiments/phase8_progress_rl/train.py"


def processes(script):
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid() or Path(os.readlink(entry/"cwd")).resolve() != ROOT:
                continue
            args = (entry/"cmdline").read_bytes().decode().strip("\0").split("\0")
            if script in args:
                result.append((int(entry.name), args))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return result


def alive(pid):
    try:
        return (Path("/proc")/str(pid)/"stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main():
    stopped = []
    for pid, _ in processes("experiments/phase8_progress_rl/launch.py"):
        os.kill(pid, signal.SIGTERM)
        stopped.append(pid)
    time.sleep(1)
    lock = (OUT/"launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = OUT/"launch_status.csv"
    with status.open(newline="") as stream:
        completed = {(r["variant"], int(r["seed"])) for r in csv.DictReader(stream)}
    running = {}
    for pid, args in processes(TRAIN):
        arm, seed = args[args.index("--variant")+1], int(args[args.index("--seed")+1])
        if "--smoke" in args:
            continue
        environment = (Path("/proc")/str(pid)/"environ").read_bytes().split(b"\0")
        gpu = int(next(value.split(b"=", 1)[1] for value in environment if value.startswith(b"CUDA_VISIBLE_DEVICES=")))
        running[pid] = dict(arm=arm, seed=seed, gpu=gpu, process=None, log=None,
                            start="adopted_from_original_queue")
    active = {(r["arm"], r["seed"]) for r in running.values()}
    pending = [(arm, seed) for seed in range(5) for arm in ("A", "B", "C", "D")
               if (arm, seed) not in completed | active]
    (OUT/"throughput_audit.json").write_text(json.dumps(dict(
        changed_at_utc=datetime.now(timezone.utc).isoformat(), stopped_scheduler_pids=stopped,
        adopted_training_pids=list(running), max_jobs_per_gpu=3,
        training_envs_unchanged=32, update_frequency_unchanged=4, step_budget_unchanged=300000), indent=2))
    with status.open("a", newline="") as stream:
        writer = csv.writer(stream)
        while pending or running:
            for pid, r in list(running.items()):
                code = r["process"].poll() if r["process"] is not None else (None if alive(pid) else 0)
                if code is None:
                    continue
                run = f"P8{r['arm']}_seed{r['seed']}"
                marker = ROOT/"checkpoints"/"phase8_progress_rl"/run/"completed.json"
                if code == 0 and (not marker.exists() or json.loads(marker.read_text())["steps"] != 300000):
                    code = 1
                writer.writerow((r["arm"], r["seed"], r["gpu"], code, r["start"], datetime.now(timezone.utc).isoformat()))
                stream.flush()
                if r["log"] is not None:
                    r["log"].close()
                del running[pid]
                print(f"finished {run} gpu={r['gpu']} exit={code}", flush=True)
                if code:
                    raise RuntimeError(f"Run failed: {run}")
            query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            readings = [tuple(int(v.strip()) for v in line.split(",")) for line in query.splitlines()]
            counts = {gpu: sum(r["gpu"] == gpu for r in running.values()) for gpu, _, _ in readings}
            while pending:
                candidates = [(gpu, memory, utilization) for gpu, memory, utilization in readings
                              if counts[gpu] < 3 and memory < 100000]
                if not candidates:
                    break
                gpu, _, _ = min(candidates, key=lambda r: (counts[r[0]], r[2], r[1]))
                arm, seed = pending.pop(0)
                run = f"P8{arm}_seed{seed}"
                if (ROOT/"checkpoints"/"phase8_progress_rl"/run).exists():
                    raise RuntimeError(f"Unexpected existing run: {run}")
                logfile = (LOG/f"{run}.log").open("w")
                proc = subprocess.Popen([str(ROOT/".venv"/"bin"/"python"), TRAIN,
                    "--variant", arm, "--seed", str(seed), "--steps", "300000", "--num-envs", "32", "--device", "cuda:0"],
                    cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=logfile, stderr=subprocess.STDOUT)
                running[proc.pid] = dict(arm=arm, seed=seed, gpu=gpu, process=proc, log=logfile,
                                        start=datetime.now(timezone.utc).isoformat())
                counts[gpu] += 1
                print(f"started {run} gpu={gpu} pid={proc.pid}", flush=True)
            time.sleep(5)
    print("All 20 primary runs completed.", flush=True)


if __name__ == "__main__":
    main()
