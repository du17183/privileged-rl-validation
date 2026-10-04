"""Atomic IPC scoped entirely to Phase 8."""
import json
import os
import subprocess
import sys
import time
import torch


class ProgressEvalClient:
    def __init__(self, root, variant, seed, tag, num_envs=32):
        self.ipc = root/"results"/"phase8_progress_rl"/"ipc"/f"{tag}_{os.getpid()}"
        self.ipc.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.log = (root/"logs"/"phase8_progress_rl"/f"eval_{tag}.log").open("w")
        self.process = subprocess.Popen([sys.executable, "experiments/phase8_progress_rl/eval_worker.py",
            "--variant", variant, "--seed", str(seed), "--ipc-dir", str(self.ipc),
            "--num-envs", str(num_envs), "--device", "cuda:0"], cwd=root,
            stdout=self.log, stderr=subprocess.STDOUT)
        try:
            self.wait(self.ipc/"ready.json")
        except BaseException:
            self.close()
            raise

    def wait(self, path):
        deadline = time.monotonic()+300
        while not path.exists():
            if (self.ipc/"error.json").exists():
                raise RuntimeError(json.loads((self.ipc/"error.json").read_text())["traceback"])
            if self.process.poll() is not None:
                raise RuntimeError(f"Evaluator exited {self.process.returncode}; {self.log.name}")
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(0.05)
        return json.loads(path.read_text())

    def evaluate(self, agent, seed, rounds=2, noise=0.0, target=1.0):
        n = self.sequence
        actor_path = self.ipc/f"actor_{n}.pt"
        torch.save(agent.actor.state_dict(), actor_path)
        request = self.ipc/f"request_{n}.json"
        temp = request.with_suffix(".tmp")
        temp.write_text(json.dumps(dict(actor_path=str(actor_path), eval_seed=seed, rounds=rounds, noise=noise, target=target)))
        os.replace(temp, request)
        result = self.wait(self.ipc/f"response_{n}.json")
        actor_path.unlink()
        self.sequence += 1
        return result

    def close(self):
        (self.ipc/"stop").touch()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()
