"""Conserve completed per-env trajectories and audit old artifacts/learning code."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, ARMS


def main():
    results=[]
    for arm in ARMS:
        for seed in range(5):
            run=f'P11{arm}_seed{seed}';marker=json.loads((CKPT/run/'completed.json').read_text())
            assert marker['steps']==300000 and marker['optimizer_steps']==37500
            with h5py.File(ROOT/'datasets'/OUT.name/f'{run}.h5','r') as h5:
                data=h5['transitions'];records=json.loads(h5['episodes_json'][()]);done=data['done'][:,0];reward=data['reward'][:,0]
                next_gt=data['next_privileged'][:];covered=np.zeros(300000,dtype=bool)
                per_env={i:[] for i in range(32)}
                for record in records:
                    indices=record['start']+np.arange(record['length'])*32
                    assert not covered[indices].any();covered[indices]=True
                    assert done[indices[-1]]==1 and done[indices[:-1]].sum()==0
                    assert bool(next_gt[indices[-1],0]>1.)==bool(record['success'])
                    assert abs(float(reward[indices].sum())-record['return_value'])<.02
                    assert abs(float(next_gt[indices[-1],0])-record['final_angle'])<1e-6
                    assert indices[0]%32==record['env_index']
                    per_env[record['env_index']].append((indices[0],indices[-1]))
                for i,spans in per_env.items():
                    spans.sort()
                    if spans: assert spans[0][0]==i
                    for previous,current in zip(spans,spans[1:]): assert current[0]==previous[1]+32
                assert len(records)==marker['online_episodes'] and int(sum(r['success'] for r in records))==marker['online_successes']
                assert data['robot'].shape[1]==marker['observation_dim']
                # Added angle channel must agree with physical GT including terminal transitions.
                if arm!='A':
                    assert np.allclose(data['robot'][:,26],data['privileged'][:,0],atol=1e-7)
                    assert np.allclose(data['next_robot'][:,26],next_gt[:,0],atol=1e-7)
                results.append(dict(arm=arm,seed=seed,episodes=len(records),successes=marker['online_successes'],covered_completed_transitions=int(covered.sum()),
                    pending_transitions=int((~covered).sum()),terminal_state_verified=True,observation_dim=marker['observation_dim']))
    changes=[];inventory=json.loads((OUT/'baseline_inventory.json').read_text())
    for name,old in inventory.items():
        path=ROOT/name
        if not path.exists(): changes.append(name);continue
        stat=path.stat();current=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if 'sha256' in old: current['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if current!=old: changes.append(name)
    manifest=json.loads((OUT/'training_source_manifest.json').read_text())
    # Postprocessing may be added after training; all snapshotted code is immutable.
    source_changes=[name for name,digest in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    result=dict(passed=not changes and not source_changes,datasets=20,results=results,prior_artifacts=len(inventory),prior_changed=changes,
        training_source_changes=source_changes,total_completed_episodes=sum(r['episodes'] for r in results),total_successes=sum(r['successes'] for r in results))
    (OUT/'formal_data_audit.json').write_text(json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('Audit failed')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
if __name__=='__main__': main()
