import json
import os
import subprocess
import sys
import time
import torch
from experiments.phase12_environment_state_feedback.protocol import OUT, LOG


class EvaluationClient:
    def __init__(self, root, seed, arm, run):
        self.ipc = OUT/'ipc'/f'{run}_{os.getpid()}'
        self.ipc.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        self.log = (LOG/f'eval_{run}.log').open('x')
        self.process = subprocess.Popen([sys.executable, '-u', 'experiments/phase12_environment_state_feedback/eval_worker.py',
            '--arm', arm, '--seed', str(seed), '--ipc-dir', str(self.ipc), '--device', 'cuda:0'], cwd=root,
            stdout=self.log, stderr=subprocess.STDOUT)
        try: self.wait(self.ipc/'ready.json')
        except BaseException: self.close();raise

    def wait(self, path):
        deadline = time.monotonic()+1800
        while not path.exists():
            if (self.ipc/'error.json').exists(): raise RuntimeError(json.loads((self.ipc/'error.json').read_text())['traceback'])
            if self.process.poll() is not None: raise RuntimeError(f'Evaluator failed: {self.log.name}')
            if time.monotonic() > deadline: raise TimeoutError(str(path))
            time.sleep(.025)
        return json.loads(path.read_text())

    def tests(self, actor, tests):
        actor_path = self.ipc/f'actor_{self.sequence}.pt'
        torch.save(dict(actor=actor.state_dict(), std_cap=actor.cap), actor_path)
        request = self.ipc/f'request_{self.sequence}.json'
        temp = request.with_suffix('.tmp');temp.write_text(json.dumps(dict(actor_path=str(actor_path), tests=tests)))
        os.replace(temp, request)
        result = self.wait(self.ipc/f'response_{self.sequence}.json')
        actor_path.unlink();self.sequence += 1
        return result

    def pairs(self, actor, conditions, seed, rounds=1):
        tests = [dict(condition=c, mode=m, seed=seed, rounds=rounds) for c in conditions for m in ('deterministic', 'policy')]
        raw = self.tests(actor, tests)
        return {c: {m: raw[f'{c}:{m}:{seed}']['metrics'] for m in ('deterministic', 'policy')} for c in conditions}, raw

    def close(self):
        (self.ipc/'stop').touch()
        try: self.process.wait(timeout=30)
        except subprocess.TimeoutExpired: self.process.terminate();self.process.wait(timeout=10)
        self.log.close()
