"""Batch-one measurements on changed real states; include cold costs separately."""
import argparse
import json
import time
import traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from pi05.model import build,restore
from pi05.state_adapter import StateAdapter
from phase15.recovery_policy.model import build_recovery
from phase15.recovery_policy.state import RecoveryAdapter,expand
from phase15_1.common import ROOT,RESULTS,frozen,save_json,sha256
from .engine import Engine


def main(a):
    torch.set_num_threads(4)
    keys=json.loads((ROOT/'datasets/random_door_expert/split_v1/split.json').read_text())['splits']['train']
    with h5py.File(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5') as h:
        g=h[keys[0]];r=g['robot_state'][:];e=g['environment_state'][:]
        raw=np.concatenate((r,e),axis=1)
    results=[]
    for module in ['normal','recovery']:
        net=build() if module=='normal' else build_recovery()
        ck=frozen(0)[module];meta=restore(net,ck);net.eval().requires_grad_(False)
        for backend in [a.backend]:
            engine=Engine(net,(RecoveryAdapter if module=='recovery' else StateAdapter)(meta['mean'],meta['std']),backend,verify_count=23)
            for chunk in [5,10,15,20,30]:
                row=dict(module=module,backend=backend,chunk=chunk,batch=1,checkpoint=ck,sha256=sha256(ck),
                    trained_horizon=10,horizon_extrapolation=chunk>10,samples=[],success_preservation_test_pending=True)
                try:
                    for j in range(23):
                        value=raw[(j*11)%len(raw)][None]
                        if module=='recovery':value=expand(value)
                        wall=time.perf_counter();out,latency=engine.run(value,[151001+j],chunk)
                        torch.cuda.synchronize();service=time.perf_counter()-wall
                        if j==0:row['cold_total_s']=service
                        if j>=3:row['samples'].append(dict(model_s=latency,service_s=service))
                    row.update(engine.status(),model_mean_s=float(np.mean([s['model_s'] for s in row['samples']])),
                        model_p95_s=float(np.quantile([s['model_s'] for s in row['samples']],.95)),
                        budget_s=chunk/60.,deadline_misses=sum(s['service_s']>chunk/60 for s in row['samples']),
                        theoretical_call_hz=60/chunk,rl_updates=0)
                except BaseException:
                    row['failed']=traceback.format_exc();print(row['failed'],flush=True)
                    save_json(RESULTS/f'latency/{a.revision}/{backend}_{module}_chunk{chunk}.json',row)
                    if backend in ['graph','compile','compile_approx']:break
                save_json(RESULTS/f'latency/{a.revision}/{backend}_{module}_chunk{chunk}.json',row)
                results.append(row);print(json.dumps(dict(module=module,backend=backend,chunk=chunk,mean=row.get('model_mean_s'),failed='failed' in row)),flush=True)
        del engine,net;torch.cuda.empty_cache()
    save_json(RESULTS/f'latency/{a.revision}/{a.backend}.complete.json',dict(results=len(results),rl_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cached','unroll','graph','compile','compile_approx'],required=True)
    p.add_argument('--revision',default='verified_v2')
    main(p.parse_args())
