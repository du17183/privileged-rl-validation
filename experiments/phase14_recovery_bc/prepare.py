"""Frozen source splits and equal-label ordinary-success control."""
import json,hashlib,argparse
from pathlib import Path
import h5py,numpy as np
from experiments.phase14_recovery_bc.gate import require

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase14_recovery_bc'
OUT=ROOT/'datasets/recovery_expert/split_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path,seed):
    values={};issues=[]
    with h5py.File(path,'r') as h:
        keys=sorted(h.keys())
        if len(keys)!=200:raise RuntimeError('Need exactly200 successful added trajectories')
        for key in keys:
            g=h[key];robot=g['observation'][:];environment=g['environment_state'][:];action=g['action'][:]
            for name,dim in [('observation',26),('environment_state',13),('action',7),('state',11)]:
                array=g[name][:]
                if array.shape!=(len(action),dim) or not np.isfinite(array).all():issues.append([key,name,'shape/finite'])
            if not bool(g.attrs['success']) or not bool(g['done'][-1,0]) or bool(g['done'][:-1].any()):issues.append([key,'terminal'])
            for name in ['observation','environment_state','state']:
                if not np.allclose(g['next_'+name][:-1],g[name][1:],atol=1e-6):issues.append([key,name,'continuity'])
            if 'episode_tick' in g:
                age=g['episode_tick'][:].flatten()
                recovery_source='ordinary_control' not in str(path)
                if (recovery_source and age[0]<=0) or not np.array_equal(np.diff(age),np.ones(len(age)-1)):issues.append([key,'reset at takeover'])
            if not np.array_equal(environment[:,0],g['state'][:,0]) or not np.array_equal(environment[:,4:6],g['state'][:,9:11]):issues.append([key,'GT alignment'])
            values[key]=dict(robot=robot,environment=environment,action=action)
    if issues:raise RuntimeError(issues[:5])
    np.random.default_rng(seed).shuffle(keys)
    split=dict(train=keys[:140],validation=keys[140:170],test=keys[170:])
    arrays={}
    for name,ids in split.items():
        data={field:np.concatenate([values[k][field] for k in ids]) for field in ['robot','environment','action']}
        data['trajectory_index']=np.concatenate([np.full(len(values[k]['action']),i,dtype=np.int32) for i,k in enumerate(ids)])
        arrays[name]=data
    return arrays,dict(splits=split,sha256=digest(path),trajectories=len(keys),issues=issues)


def main(run_id=None):
    global R,OUT
    if run_id:
        R=R/run_id;OUT=ROOT/'datasets/recovery_expert'/run_id/'split_v1'
    verification=json.loads((R/'recovery_validation/summary.json').read_text())
    require(verification)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError(OUT)
    sources={};manifest={}
    for name,path,seed in [('recovery',OUT.parent/'collection_v1/trajectories.h5',14301),
                           ('ordinary',OUT.parent/'ordinary_control_v1/trajectories.h5',14302)]:
        sources[name],manifest[name]=read_source(path,seed)
    # Same number of newly labeled training transitions, as well as the same
    # number of added trajectories and source sampling probability.
    size=min(len(sources[s]['train']['action']) for s in sources)
    for name in sources:
        before=len(sources[name]['train']['action'])
        if before>size:
            ids=np.sort(np.random.default_rng(14303).choice(before,size,replace=False))
            sources[name]['train']={k:v[ids] for k,v in sources[name]['train'].items()}
        manifest[name]['training_labels_before_matching']=before
        manifest[name]['training_labels_used']=size
        for split,arrays in sources[name].items():np.savez(OUT/f'{name}_{split}.npz',**arrays)
    base=ROOT/'datasets/random_door_expert/split_v1'
    base_manifest=json.loads((base/'split.json').read_text())
    for split in ['train','validation','test']:
        with np.load(base/(split+'.npz')) as h:np.savez(OUT/f'base_{split}.npz',**dict(h))
    manifest.update(base=base_manifest,mean=base_manifest['mean'],std=base_manifest['std'],
        recovery_validation=verification,label_matching=size,checkpoint_rule='common source-balanced validation MSE')
    manifest['dataset_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(matched_added_labels=size,counts={s:{k:len(v['action']) for k,v in sources[s].items()} for s in sources})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id');main(p.parse_args().run_id)
