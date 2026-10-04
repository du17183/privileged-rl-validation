"""Only Phase 9 paths are writable; atomically publish each actor candidate."""
import json
import os
import subprocess
import sys
import time
import torch


class EvaluationClient:
    def __init__(self, root, seed, run):
        self.ipc = root/"results"/"phase9_safe_online"/"ipc"/f"{run}_{os.getpid()}"
        self.ipc.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.log = (root/"logs"/"phase9_safe_online"/f"eval_{run}.log").open("w")
        self.process = subprocess.Popen([sys.executable, "experiments/phase9_safe_online/eval_worker.py",
            "--seed", str(seed), "--ipc-dir", str(self.ipc), "--device", "cuda:0"], cwd=root,
            stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self.wait(self.ipc/"ready.json")
        except BaseException:
            self.close()
            raise

    def wait(self, path):
        deadline = time.monotonic()+600
        while not path.exists():
            if (self.ipc/"error.json").exists():
                raise RuntimeError(json.loads((self.ipc/"error.json").read_text())["traceback"])
            if self.process.poll() is not None:
                raise RuntimeError(f"Evaluator failed: {self.log.name}")
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(.025)
        return json.loads(path.read_text())

    def evaluate(self, actor, seed, mode, rounds=1):
        path = self.ipc/f"actor_{self.sequence}.pt"
        torch.save(dict(actor=actor.state_dict(), std_cap=actor.cap), path)
        request = self.ipc/f"request_{self.sequence}.json"
        temp = request.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(actor_path=str(path), seed=seed, mode=mode, rounds=rounds)))
        os.replace(temp, request)
        result = self.wait(self.ipc/f"response_{self.sequence}.json")
        path.unlink()
        self.sequence += 1
        return result

    def pair(self, actor, seed, rounds=1):
        return {mode: self.evaluate(actor, seed, mode, rounds) for mode in ("deterministic", "policy")}

    def close(self):
        (self.ipc/"stop").touch()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

