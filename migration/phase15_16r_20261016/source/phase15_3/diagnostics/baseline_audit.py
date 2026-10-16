"""Audit historical inputs without writing to a historical namespace."""
import hashlib, json, os, subprocess, time
from pathlib import Path
import h5py

R=Path.cwd().resolve(); O=R/'phase15_3/results'; P=R/'phase15_2/results'
def read(p): return json.loads(Path(p).read_text())
def save(p,d): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
cache={};issues=[];counts={}
def verify(p,want,kind):
    p=Path(p);p=p if p.is_absolute() else R/p
    if p not in cache:
        s=p.stat();h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        t=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
        cache[p]=h.hexdigest()
    counts.setdefault(kind,set()).add(str(p))
    if cache[p]!=want:issues.append([kind,str(p),want,cache[p]])
    return cache[p]
def main():
    manifest=P/'code_versions/final_delivery_v1/delivery_snapshot_final002_20261008T081707Z.json'
    verify(manifest,'d4ce648b01eb50ed549941a0705e882b088d1b6680e907593de86fbcbde6d98e','delivery_manifest')
    m=read(manifest);assert m['full_delivery_complete'] and m['file_count']==2435
    for p,v in m['files'].items():verify(p,v['sha256'],'delivery_files')
    frozen=read(P/'frozen_inputs.json')['files'];effective=dict(frozen);effective.update(read(P/'code_manifest.json')['files'])
    for p,v in effective.items():
        if Path(p).suffix=='.py':verify(p,v,'source')
    for p,v in frozen.items():
        if p.endswith('.pt'):verify(p,v,'checkpoint')
        elif not p.endswith('.py'):verify(p,v,'frozen_other')
    assert len(counts['checkpoint'])==10
    raw=read(P/'code_versions/analysis_v2_audit_markers/archive_manifest.json')['raw_formal_sha256']
    for p,v in raw.items():verify(p,v,'formal_raw')
    source_paths=set();identities={};distances={}
    for split,quota in [('validation',32),('test',100)]:
        identity=[];dist=[]
        for seed in range(5):
            base=P/f'cohorts/{split}/seed{seed}';cohorts=read(base/'manifest.json')
            for case,v in cohorts.items():
                verify(base/(case+'.h5'),v['sha256'],'cohort')
                verify(v['source'],v['source_sha256'],'cohort_source');source_paths.add(v['source'])
            with h5py.File(base/'natural_severe.h5') as h:
                assert len(h)==quota and h.attrs['evaluation_only'] and not h.attrs['training_labels']
                for key in sorted(h):
                    a=h[key].attrs
                    assert a['natural_source_episode_failed'] and not a['training_labels']
                    identity.append((int(a['reference_model_seed']),int(a['reset_seed']),int(a['clone']),int(a['source_episode'])))
                    dist.append(float(a['distance']))
        assert len(set(identity))==len(identity)
        identities[split]=identity;distances[split]=[min(dist),max(dist)]
    assert len(identities['validation'])==160 and len(identities['test'])==500
    assert not set(identities['validation'])&set(identities['test'])
    cases=0;episodes=0;models=set();weights=set()
    for jobfile in sorted((P/'jobs/formal_v1').glob('[0-9][0-9][0-9].json')):
        for j in read(jobfile):
            models.add((j['method'],j['seed']));weights.update(j['checkpoints'].values())
            for role,p in j['checkpoints'].items():assert p in counts['checkpoint']
            for case in j['tests']:
                x=read(R/j['output']/(case+'.json'));cases+=1;episodes+=x['episodes']
                assert x['episode_budget_ticks']==600 and x['rl_updates']==0
                for role,p in j['checkpoints'].items():assert x['checkpoint_sha256'][role]==cache[Path(p)]
    assert (cases,episodes,len(models),len(weights))==(300,30500,25,10)
    for n in ['final_audit.json','funnel_audit.json']:
        assert read(P/n)['passed']
    mirror=R/'results/phase15_2/final_audit.json'
    assert mirror.read_bytes()==(P/'final_audit.json').read_bytes()
    assert read(P/'code_versions/final_delivery_v1/delivery_complete.json')['full_delivery_complete']
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    report=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),passed=not issues,issues=issues,
        checked={k:len(v) for k,v in counts.items()},formal_tests=cases,formal_episodes=episodes,
        model_seed_pairs=sorted(models),checkpoint_count=len(weights),natural_validation=160,natural_test=500,
        natural_overlap=0,natural_identity_definition='reference_model_seed/reset_seed/clone/source_episode',
        actual_distance_m=distances,sha256={str(p.relative_to(R)):v for p,v in cache.items()},
        gpu_csv=gpu,compute_apps_csv=apps,logical_cpus=os.cpu_count(),loadavg=os.getloadavg(),
        history_modified=False,training=False,rl_updates=0)
    save(O/'phase15_2_baseline_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','model_seed_pairs']},indent=2))
    assert not issues
if __name__=='__main__':main()
