"""Package completed Phase14.2 artifacts; release only verified orphan workers."""
import hashlib,importlib.metadata,json,os,platform,shutil,signal,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase14_2_recovery_bc'
NEW=['experiments/phase14_2_recovery_bc','evaluation/recovery_policy_eval.py',
     'evaluation/natural_deviation_eval.py','datasets/phase14_2',
     'checkpoints/phase14_2_recovery_bc','logs/phase14_2_recovery_bc',
     'results/phase14_2_recovery_bc','docs/phase14_2_recovery_bc_report.md']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def guard():
    p=json.loads((R/'progress.json').read_text())
    assert p['all_evaluation_complete'] and not p['missing'] and p['rl_updates']==0
    assert p['main_tests_complete']==245 and p['final_sensitivity_complete']==30
    assert not json.loads((R/'verification.json').read_text())['issues']
    assert not json.loads((R/'preservation_after.json').read_text())['changed']
    assert json.loads((R/'quantity_complete.json').read_text())['tests']==30
    assert json.loads((R/'repair_complete.json').read_text())
    count=0
    for sub in ['evaluation','evaluation_final','quantity_control']:
        for p in (R/sub).glob('*/*.json'):
            d=json.loads(p.read_text())
            if 'episodes' in d:count+=d['episodes']
    assert count==24960,count
    return count

def cleanup():
    # Fixed, inspected PIDs from this task's obsolete workers. Recheck exact
    # command, project and cwd before either signal; never kill another project.
    candidates=[123492,123496,123499,4062216,4062218,4062220,4062221,
                4062223,4062225,4062227,4062228]
    rows=[]
    def owned(pid):
        try:
            command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            args=[v.decode() for v in command if v]
            return '-m' in args and args[args.index('-m')+1]=='experiments.phase14_2_recovery_bc.evaluate' and \
                '--jobs' in args and Path(args[args.index('--jobs')+1]).resolve()==R/'evaluation_jobs.json' and \
                (Path('/proc')/str(pid)/'cwd').resolve()==ROOT
        except (FileNotFoundError,ProcessLookupError,IndexError):return False
    for pid in candidates:
        if owned(pid):
            os.kill(pid,signal.SIGTERM);rows.append({'pid':pid,'action':'SIGTERM','verified_project':str(ROOT)})
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and any(owned(v['pid']) for v in rows):time.sleep(.2)
    for row in rows:
        if owned(row['pid']):
            os.kill(row['pid'],signal.SIGKILL);row['action']='SIGTERM then SIGKILL'
    time.sleep(.2)
    lingering=[v['pid'] for v in rows if owned(v['pid'])]
    d={'utc':datetime.now(timezone.utc).isoformat(),'completion_checked':True,
       'reason':'obsolete evaluation workers remained in Isaac shutdown after completed results',
       'workers':rows,'remaining_own_workers':lingering,'other_projects_signalled':0}
    (R/'worker_cleanup.json').write_text(json.dumps(d,indent=2))
    assert not lingering,lingering

def main():
    episodes=guard();cleanup()
    delivery=R/'delivery';delivery.mkdir(exist_ok=True)
    reused=['configs/runtime_env.sh','envs','door_env','randomized_env','environment_state',
            'progress_rl','recovery_expert','experiments/phase13_random_expert_bc','bc']
    sources={}
    for relative in reused:
        p=ROOT/relative
        for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix not in ['.py','.json','.yaml','.yml','.sh']:continue
            rel=f.relative_to(ROOT);dest=delivery/'reused_source_snapshot'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
            sources[str(rel)]=sha(f)
    versions=sorted({(d.metadata.get('Name','unknown'),d.version) for d in importlib.metadata.distributions()})
    runtime={'utc':datetime.now(timezone.utc).isoformat(),'host':platform.node(),
       'python':sys.version,'platform':platform.platform(),'packages':[{'name':n,'version':v} for n,v in versions],
       'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True).splitlines(),
       'rl_updates':0,'models':45,'valid_evaluation_episodes':episodes}
    (delivery/'runtime.json').write_text(json.dumps(runtime,indent=2))
    (delivery/'reused_source_sha256.json').write_text(json.dumps(sources,indent=2))
    readme='''# Phase14.2 delivery

Complete new source, data splits, two ordinary expert collections, all model
weights, logs, pressure cohorts, episode results, statistics, plots and report.
No RL was trained. No qualified new Anchor was created.

Read docs/phase14_2_recovery_bc_report.md first. Validation-MSE-selected and
final20k checkpoint results are different estimands and both are retained.
evaluation_initialization_v1 is historical debug evidence, excluded from all
final statistics. Existing Phase13/14.1 raw HDF files remain at original paths;
their hashes and references are in dataset manifests. Prepared arrays reproduce
this phase without duplicating those historical raw HDF files.

Source snapshots of unchanged dependencies are under delivery/reused_source_snapshot
and are not extracted over the old project sources. runtime.json records actual
installed packages and GPU inventory. artifact_sha256.json covers all new files
except itself. The external archive SHA256 is saved beside the archive.

Resource logs include obsolete shutdown-stalled workers; worker_cleanup.json
records verified cleanup of only this task's processes after all gates completed.
'''
    (delivery/'README.md').write_text(readme,encoding='utf-8')
    files=[]
    for relative in NEW:
        p=ROOT/relative
        for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc':files.append(f)
    manifest=delivery/'artifact_sha256.json'
    digests={str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files if f!=manifest}
    manifest.write_text(json.dumps(digests,indent=2));files.append(manifest)
    archive=ROOT/'results/phase14_2_recovery_bc_review_bundle.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for f in sorted(set(files)):t.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    digest=sha(archive);Path(str(archive)+'.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,
                      'sha256':digest,'files':len(set(files)),'valid_episodes':episodes}),flush=True)

if __name__=='__main__':main()
