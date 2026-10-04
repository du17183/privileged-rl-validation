"""Source provenance and offline completed-expert quality spread."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.phase10_quality_lwd.data import expert_records
from experiments.phase10_quality_lwd.protocol import PROTOCOL
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase10_quality_lwd'


def main():
    records=expert_records(ROOT/'door_dataset/door_expert_1000.h5')
    scores=np.array([r['quality_score'] for r in records])
    sources=[]
    for seed in range(5):
        p9=ROOT/'checkpoints/phase9_safe_online'/f'P9C_seed{seed}'/'step_300000.pt'
        p8=ROOT/'checkpoints/phase8_progress_rl'/f'P8B_seed{seed}'/'best.pt'
        state=torch.load(p9,map_location='cpu',weights_only=False)
        assert state['std_cap']==.01 and state['env_steps']==300000
        assert state['anchor_sha256']==hashlib.sha256(p8.read_bytes()).hexdigest()
        sources.append(dict(seed=seed,phase9_path=str(p9),phase9_sha256=hashlib.sha256(p9.read_bytes()).hexdigest(),
                            phase8_anchor_sha256=state['anchor_sha256'],std_cap=state['std_cap']))
    manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
              ('door_env/door.py','door_env/isaac_env.py','assets/panda_door_cabinet.usd',
               'door_dataset/door_expert_1000.h5','progress_rl/progress_reward.py')}
    result=dict(protocol=PROTOCOL,sources=sources,immutable_inputs=manifest,
                expert_quality=dict(trajectories=len(records),transitions=sum(r['length'] for r in records),
                    success_fraction=float(np.mean([r['success'] for r in records])),
                    mean=float(scores.mean()),sd=float(scores.std()),quantiles=np.quantile(scores,[0,.1,.5,.9,1]).tolist()),
                offline_caveat='Completed-outcome scorer already contains success and final progress. Its success ranking AUC is tautological and cannot establish learned prediction reliability; replay gains must establish usefulness.')
    with (OUT/'preflight.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['expert_quality']))

if __name__=='__main__':main()
