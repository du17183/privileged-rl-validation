"""Whole-trajectory splits, frozen S normalization, no evaluation leakage."""
import hashlib,json
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'datasets/phase14_2/prepared_v1'
CASES=['ee_offset','contact_loss','door_regression','stagnation']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def read(path,tag):
    rows={}
    with h5py.File(path,'r') as h:
        for key in sorted(h):
            g=h[key]
            if not g.attrs['success']:raise RuntimeError((path,key,'unsuccessful'))
            robot=g['observation'][:];env=g['environment_state'][:];action=g['action'][:];state=g['state'][:]
            if env.shape!=(len(action),13) or robot.shape!=(len(action),26) or state.shape!=(len(action),11):raise RuntimeError('schema')
            if not np.allclose(env[:,1],1):raise RuntimeError('variable target not allowed')
            # No nested policy/injection prefix groups are visited.
            env[:,1]=state[:,1]
            if not all(np.isfinite(x).all() for x in [robot,env,action,state]):raise RuntimeError('nonfinite')
            if not bool(g['done'][-1]) or g['done'][:-1].any():raise RuntimeError('terminal')
            rows[f'{tag}/{key}']=dict(robot=robot,environment=env,action=action)
    return rows

def array(rows,keys):
    d={k:np.concatenate([rows[i][k] for i in keys]) for k in ['robot','environment','action']}
    d['trajectory_index']=np.concatenate([np.full(len(rows[i]['action']),j,dtype=np.int32) for j,i in enumerate(keys)])
    return d

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError(OUT)
    base=ROOT/'datasets/random_door_expert/split_v1'
    split=json.loads((base/'split.json').read_text());print('BASE KEYS',list(split),flush=True)
    paths={'base':[(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5','S')],
           'recovery':[(ROOT/'datasets/recovery_expert/phase14_1/formal_v1'/case/'trajectories.h5',case) for case in CASES],
           'ordinary':[(ROOT/'datasets/phase14_2/success_size_matched/collection_v1/trajectories.h5','N')]}
    raw={s:{} for s in paths};splits={};hashes={}
    for s,items in paths.items():
        for p,tag in items:raw[s].update(read(p,tag));hashes[str(p.relative_to(ROOT))]=sha(p)
    if [len(raw[s]) for s in raw]!=[300,484,484]:raise RuntimeError('trajectory quotas')
    # Original S split IDs from its immutable manifest.
    base_ids=split.get('splits',split.get('split'))
    if base_ids is None:raise RuntimeError(list(split))
    splits['base']={k:['S/'+i for i in base_ids[k]] for k in ['train','validation','test']}
    for s in ['recovery','ordinary']:
        result={k:[] for k in ['train','validation','test']}
        for tag in sorted(set(i.split('/')[0] for i in raw[s])):
            keys=sorted(i for i in raw[s] if i.split('/')[0]==tag)
            np.random.default_rng(142301+(0 if s=='recovery' else 1)).shuffle(keys)
            train=int(len(keys)*.7);valid=int(len(keys)*.15)
            for k,ids in zip(result,[keys[:train],keys[train:train+valid],keys[train+valid:]]):result[k].extend(ids)
        splits[s]=result
    mean=np.array(split['mean'],dtype=np.float32);std=np.array(split['std'],dtype=np.float32)
    velocity=np.concatenate([raw['base'][i]['environment'][:,1] for i in splits['base']['train']])
    mean[27]=velocity.mean();std[27]=max(float(velocity.std()),1e-3)
    counts={}
    for s in raw:
        counts[s]={}
        for k,ids in splits[s].items():
            d=array(raw[s],ids);np.savez(OUT/f'{s}_{k}.npz',**d)
            counts[s][k]=dict(trajectories=len(ids),transitions=len(d['action']))
        target=ROOT/'datasets/phase14_2'/dict(base='success_original',recovery='recovery',ordinary='success_size_matched')[s]
        target.mkdir(parents=True,exist_ok=True)
        (target/'manifest.json').write_text(json.dumps(dict(source_files=[str(p.relative_to(ROOT)) for p,_ in paths[s]],splits=splits[s],counts=counts[s],no_prefix_labels=True),indent=2))
    m=dict(mean=mean.tolist(),std=std.tolist(),target_angle=1.0,splits=splits,counts=counts,raw_sha256=hashes,
           excluded=['phase14_1/natural_crosscheck_v1','Phase14 natural validation','formal failed attempts','all policy/injection prefixes'],
           added_sampling='256 S + 256 R or N, exactly 5120000 draws per source, all arms20000updates',
           normalization='original train-S frozen except constant-target slot now measured velocity fitted train-S only')
    m['dataset_sha256']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
