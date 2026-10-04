"""Learner-side client for the persistent, independent Isaac evaluator."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


class EvalClient:
    def __init__(self,root,variant,seed,num_envs,device,timeout_s=180):
        self.root=Path(root)
        self.ipc=self.root/'results'/'stable_privileged_rl'/'eval_ipc'/f'{variant}_seed{seed}_{os.getpid()}'
        self.ipc.mkdir(parents=True,exist_ok=False)
        self.sequence=0
        self.timeout_s=timeout_s
        log=self.root/'logs'/'stable_privileged_rl'/f'eval_worker_{variant}_seed{seed}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        self.log=log.open('w',encoding='utf-8')
        cmd=[sys.executable,'experiments/stable_privileged_rl/eval_worker.py',
             '--variant',variant,'--seed',str(seed),'--num-envs',str(num_envs),
             '--ipc-dir',str(self.ipc),'--device',device]
        self.process=subprocess.Popen(cmd,cwd=self.root,stdout=self.log,
                                      stderr=subprocess.STDOUT)
        try:
            self._wait(self.ipc/'ready.json')
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=10)
            self.log.close()
            raise

    def _wait(self,path):
        deadline=time.monotonic()+self.timeout_s
        while not path.exists():
            error=self.ipc/'error.json'
            if error.exists():
                raise RuntimeError(json.loads(error.read_text(encoding='utf-8'))['traceback'])
            code=self.process.poll()
            if code is not None:
                raise RuntimeError(f'Evaluation worker exited {code}; see {self.log.name}')
            if time.monotonic()>deadline:
                raise TimeoutError(f'Evaluation worker timed out: {path}')
            time.sleep(.05)
        return json.loads(path.read_text(encoding='utf-8'))

    def evaluate(self,agent,seed,rounds):
        index=self.sequence
        actor_path=self.ipc/f'actor_{index}.pt'
        torch.save(agent.actor.state_dict(),actor_path)
        request=self.ipc/f'request_{index}.json'
        temporary=request.with_suffix('.tmp')
        temporary.write_text(json.dumps({'actor_path':str(actor_path),'eval_seed':seed,
                                         'rounds':rounds}),encoding='utf-8')
        os.replace(temporary,request)
        result=self._wait(self.ipc/f'response_{index}.json')
        actor_path.unlink()
        self.sequence+=1
        return (result['success_rate'],result['mean_return'],result['contact_rate'],
                result['episodes'],result['eval_env_steps'])

    def close(self):
        if not hasattr(self,'process'):
            return
        (self.ipc/'stop').touch()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()
