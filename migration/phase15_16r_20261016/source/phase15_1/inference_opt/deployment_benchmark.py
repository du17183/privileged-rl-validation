"""Exclusive GPU, batch-one RPC measurements after throughput jobs finish."""
import argparse
import json
import time
from multiprocessing.connection import Client
import h5py
import numpy as np
from phase15.recovery_policy.state import expand
from phase15_1.common import AUTH, ROOT, RESULTS, frozen, save_json


def main(a):
    deadline=time.monotonic()+900
    while True:
        try:conn=Client(a.socket,family='AF_UNIX',authkey=AUTH);break
        except (FileNotFoundError,ConnectionRefusedError):
            if time.monotonic()>deadline:raise
            time.sleep(1)
    def rpc(request):
        conn.send(request);answer=conn.recv()
        if 'error' in answer:raise RuntimeError(answer['error'])
        return answer
    results={}
    try:
        rpc(dict(op='load',checkpoints=frozen(a.seed)))
        keys=json.loads((ROOT/'datasets/random_door_expert/split_v1/split.json').read_text())['splits']['train']
        with h5py.File(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5') as h:
            g=h[keys[0]];raw=np.concatenate((g['robot_state'][:],g['environment_state'][:]),axis=1)
        for module,ticks in [('normal',a.normal),('recovery',a.recovery)]:
            samples=[];horizon=max(10,ticks)
            for j in range(23):
                value=raw[(j*11)%len(raw)][None]
                if module=='recovery':value=expand(value)
                started=time.perf_counter()
                response=rpc(dict(op='infer',module=module,raw=value,noise_seeds=[151099+j],horizon=horizon))
                elapsed=time.perf_counter()-started
                if j>=3:samples.append(dict(model_s=response['latency_s'],service_s=response['service_s'],rpc_s=elapsed))
            results[module]=dict(batch=1,horizon=horizon,execute_ticks=ticks,samples=samples,
                mean_model_s=float(np.mean([s['model_s'] for s in samples])),
                mean_rpc_s=float(np.mean([s['rpc_s'] for s in samples])),
                p95_rpc_s=float(np.quantile([s['rpc_s'] for s in samples],.95)),
                replan_budget_s=ticks/60,budget_met=all(s['rpc_s']<=ticks/60 for s in samples),
                engineering=response['engineering'],concurrent_robot_workers_on_gpu=1,
                simulation_running_on_gpu=False,real_robot_hardware_test=False,rl_updates=0)
        save_json(RESULTS/f'deployment_latency/seed{a.seed}.json',results)
    finally:rpc(dict(op='stop'));conn.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--normal',type=int,required=True);p.add_argument('--recovery',type=int,required=True)
    p.add_argument('--device');main(p.parse_args())
