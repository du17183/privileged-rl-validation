"""Audit successful trajectories and create a fixed trajectory-level split."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def prepare(dataset, validation, output):
    verification=json.loads(Path(validation).read_text())
    if verification['mode']!='validate' or verification['episodes']<256 or verification['success_rate']<=.9:
        raise RuntimeError('Independent random expert gate not passed')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'split.json').exists(): raise FileExistsError('Split is frozen')
    all_rows={};lengths={};parameters={};issues=[]
    with h5py.File(dataset,'r') as h:
        ids=sorted(h.keys())
        if len(ids)<300: raise RuntimeError('Need 300 successes for prespecified 200/50/50 split')
        for key in ids:
            g=h[key]
            arrays={name:g[name][:] for name in ['observation','environment_state','state','action','reward',
                    'next_observation','next_environment_state','next_state','done','terminated','truncated','reset_parameters']}
            length=len(arrays['action']); lengths[key]=length
            for name,dim in [('observation',26),('environment_state',13),('state',11),('action',7),('reward',1),
                             ('next_observation',26),('next_environment_state',13),('next_state',11),('done',1),
                             ('terminated',1),('truncated',1),('reset_parameters',5)]:
                if arrays[name].shape!=(length,dim) or not np.isfinite(arrays[name]).all(): issues.append([key,name,'shape/finite'])
            if not arrays['done'][-1,0] or arrays['done'][:-1].any():issues.append([key,'done','boundary'])
            if not np.array_equal(arrays['done'],arrays['terminated']|arrays['truncated']):issues.append([key,'flags','inconsistent'])
            if not arrays['next_state'][-1,0]>1 or not bool(g.attrs['success']):issues.append([key,'success','invalid'])
            for name in ['observation','state','environment_state']:
                if not np.allclose(arrays['next_'+name][:-1],arrays[name][1:],atol=1e-6):issues.append([key,name,'transition discontinuity'])
            context=arrays['environment_state'];gt=arrays['state']
            if not np.array_equal(context[:,0],gt[:,0]) or not np.array_equal(context[:,4:6],gt[:,9:11]) or not np.array_equal(context[:,6:13],gt[:,2:9]):
                issues.append([key,'environment_state','measurement alignment'])
            if not np.allclose(arrays['reset_parameters'],arrays['reset_parameters'][:1],atol=0):issues.append([key,'reset_parameters','episode drift'])
            par=arrays['reset_parameters'][0];parameters[key]=par.tolist()
            if not (0<=par[0]<=np.deg2rad(5)+1e-6 and np.max(np.abs(par[1:4]))<=.010001 and par[4]==1):issues.append([key,'distribution','out of range'])
            expected=np.clip((context[:,0]-par[0])/(1-par[0]),0,1)
            if not np.allclose(context[:,2],expected,atol=1e-5):issues.append([key,'progress','inconsistent start'])
            all_rows[key]=dict(robot=arrays['observation'],environment=arrays['environment_state'],action=arrays['action'])
    audit=dict(trajectories=len(ids),transitions=sum(lengths.values()),issues=issues,
               expert_gate=verification,length_range=[min(lengths.values()),max(lengths.values())],parameters=parameters)
    (output/'data_audit.json').write_text(json.dumps(audit,indent=2))
    if issues:raise RuntimeError(f'Dataset audit failed: {issues[:3]}')
    rng=np.random.default_rng(13301);rng.shuffle(ids)
    splits=dict(train=ids[:200],validation=ids[200:250],test=ids[250:300])
    saved={}
    for name,keys in splits.items():
        value={field:np.concatenate([all_rows[key][field] for key in keys]) for field in ['robot','environment','action']}
        value['trajectory_index']=np.concatenate([np.full(lengths[key],i,dtype=np.int32) for i,key in enumerate(keys)])
        np.savez(output/(name+'.npz'),**value);saved[name]=value
    full=np.concatenate((saved['train']['robot'],saved['train']['environment']),axis=-1)
    mean=full.mean(axis=0,dtype=np.float64).astype(np.float32)
    std=full.std(axis=0,dtype=np.float64).astype(np.float32)
    # Constant target/clock orientation channels have scale 1, not tiny noise.
    std=np.where(std<1e-4,1.,std).astype(np.float32)
    manifest=dict(seed=13301,splits=splits,lengths=lengths,mean=mean.tolist(),std=std.tolist(),
                  dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                  robot_dim=26,environment_dim=13,input_dim=39)
    with (output/'split.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(dict(trajectories=len(ids),transitions=audit['transitions'],split={k:len(v) for k,v in splits.items()},sha256=manifest['dataset_sha256'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.dataset,a.validation,a.output)
