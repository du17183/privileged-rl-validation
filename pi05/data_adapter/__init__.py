"""Same S/R splits; future action chunks never cross trajectory boundaries."""
import json
from pathlib import Path
import numpy as np
from pi05.state_adapter import StateAdapter
from pi05.action_adapter import chunk_indices
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'datasets/phase14_3/pi05_chunks'

def main():
    original=ROOT/'datasets/phase14_2/prepared_v1'
    manifest=json.loads((original/'manifest.json').read_text());DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists():return
    adapter=StateAdapter(manifest['mean'],manifest['std']);counts={}
    for source in ['base','recovery']:
        for split in ['train','validation','test']:
            output=DATA/f'{source}_{split}.npz'
            if output.exists():continue
            with np.load(original/f'{source}_{split}.npz') as h:
                raw=np.concatenate((h['robot'],h['environment']),-1);tokens,mask=adapter.encode(raw)
                idx,valid=chunk_indices(h['trajectory_index'])
                assert np.all(h['trajectory_index'][idx]==h['trajectory_index'][:,None])
                np.savez(output,raw=raw,tokens=tokens.astype(np.int32),token_mask=mask,
                         actions=h['action'][idx],action_mask=valid,index=h['trajectory_index'])
                counts[source+'_'+split]=dict(anchors=len(raw),trajectories=len(np.unique(h['trajectory_index'])),
                    max_tokens=int(mask.sum(-1).max()),valid_chunk_actions=int(valid.sum()))
                print(json.dumps({source+'_'+split:counts[source+'_'+split]}),flush=True)
    result=dict(original_manifest=manifest['dataset_sha256'],counts=counts,mean=manifest['mean'],std=manifest['std'],
                gt_features=39,target_angle=1,trajectory_boundaries_verified=True,images_used=False,chunk_horizon=10)
    (DATA/'manifest.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
