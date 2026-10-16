"""Retire only a verified CPU manager; adopt uninterrupted native workers."""
import sys,os,json,time,tarfile,io,signal,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc,protocol
from experiments.phase16_r_residual_sac.orchestrate_v4 import proc

def exact(old):
    now=proc(old['pid'])
    return now.get('alive') and all(now.get(k)==old.get(k) for k in ['argv','ppid','start_ticks'])

def main():
    protocol();folder=RES/'training_handoff_v4';folder.mkdir(exist_ok=False)
    version=EXP/'source_versions/supervision012_shared_gpu';version.mkdir(parents=True,exist_ok=False)
    prior=json.loads((RES/'supervisor_launch.json').read_text())['process']
    assert prior['pid']==1342135 and prior['start_ticks']==1025213049 and prior['ppid']==1342107
    assert prior['argv']=='.venv/bin/python -u -m experiments.phase16_r_residual_sac.orchestrate_v3 --adopt-training'
    assert exact(prior),'Original CPU manager identity changed'
    life=json.loads((RES/'lifecycle.json').read_text());assert exact(life['supervisor'])
    assert len(life['active'])==2 and all(row['job']['mode']=='train' for row in life['active']),life['active']
    for row in life['active']:
        for k in ['physics','inference']:assert exact(row[k]),('Native identity changed',row[k])
    manifest=json.loads((RES/'source_manifest.json').read_text())
    for rel,wanted in manifest['files'].items():assert sha(ROOT/rel)==wanted,('Frozen execution source changed',rel)
    preceding=json.loads((RES/'supplementary_source_v3.json').read_text())
    for rel,wanted in preceding['files'].items():assert sha(ROOT/rel)==wanted,('Previous CPU source changed',rel)
    names=['orchestrate_v4.py','handoff_train_v4.py']
    added={str((EXP/n).relative_to(ROOT)):sha(EXP/n) for n in names}
    save(RES/'supplementary_source_v4.json',dict(utc=utc(),files=added,frozen_source_manifest_sha256=sha(RES/'source_manifest.json'),protocol_sha256=sha(EXP/'protocol.json'),reason='CPU scheduling follows frozen minimum65GB/free and maximum3pairs; shared physical eval allowed, exclusive deployment latency separate; native training math, model, reward, budget, update ratio and physics unchanged',added_after_training_started=True))
    before=dict(utc=utc(),prior_supervisor=prior,helper=proc(1342107),active=life['active'],prior_launched=life['launched'],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'],text=True),apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),protocol_sha256=sha(EXP/'protocol.json'),frozen_source_manifest_sha256=sha(RES/'source_manifest.json'),baseline_sha256=sha(RES/'Frozen_pi05_baseline.json'),checkpoint_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'checkpoints/phase16_r').rglob('*.pt')})
    save(version/'before_identity.json',before)
    paths=[ROOT/rel for rel in manifest['files']]+[ROOT/rel for rel in preceding['files']]+[EXP/n for n in names]
    paths+=[p for p in RES.rglob('*.json') if 'monitor' not in p.parts and 'source_versions' not in p.parts and 'attempt' not in str(p)]
    paths+=list((ROOT/'logs/phase16_r').glob('*.log'))+[version/'before_identity.json']
    unique={str(p.relative_to(ROOT)):p for p in paths if p.is_file() and 'training_handoff_v4' not in p.parts}
    hashes={};archive=version/'before_cpu_retirement.tar.gz'
    import hashlib
    with tarfile.open(archive,'w:gz') as stream:
        for rel,path in unique.items():
            data=path.read_bytes();hashes[rel]=hashlib.sha256(data).hexdigest();info=tarfile.TarInfo(rel);info.size=len(data);stream.addfile(info,io.BytesIO(data))
    save(version/'manifest.json',dict(utc=utc(),snapshot_files=hashes,archive_sha256=sha(archive),native_workers_interrupted=False))
    # Full identity checked again immediately before signalling the owned CPU
    # manager. Popen start_new_session keeps all native workers outside its group.
    assert exact(prior)
    for row in before['active']:
        for k in ['physics','inference']:assert exact(row[k])
    os.kill(prior['pid'],signal.SIGINT)
    save(version/'signal.json',dict(utc=utc(),pid=prior['pid'],identity=prior,signal='SIGINT',scope='one owned CPU supervisor only',workers_signalled=0))
    deadline=time.monotonic()+30
    while exact(prior) or exact(before['helper']):
        if time.monotonic()>deadline:raise RuntimeError('CPU manager/helper retirement has not completed; no duplicate manager launched')
        time.sleep(.25)
    adopted=[]
    for row in before['active']:
        current=dict(row)
        for k in ['physics','inference']:
            now=proc(row[k]['pid']);assert now.get('alive') and all(now.get(f)==row[k].get(f) for f in ['argv','start_ticks']),('Uninterrupted native identity lost',row[k],now)
            assert now['ppid']==1,('Native process did not reparent naturally',now)
            current[k]=now
        adopted.append(current)
    save(version/'after_identity.json',dict(utc=utc(),prior_supervisor=proc(prior['pid']),prior_helper=proc(before['helper']['pid']),active=adopted,workers_signalled=0))
    for name in ['supervisor_launch.json','pipeline_failed.json','lifecycle.json']:
        p=RES/name
        if p.exists():p.rename(version/name)
    for p in (RES/'training_handoff_v3').glob('*.json'):
        (version/'old_helper').mkdir(exist_ok=True)
        (version/'old_helper'/p.name).write_bytes(p.read_bytes())
    adoption=dict(utc=utc(),only_CPU_supervisor_retired=True,workers_signalled=0,prior_supervisor=prior,prior_launched=before['prior_launched'],active=adopted,protocol_sha256=before['protocol_sha256'],frozen_source_manifest_sha256=before['frozen_source_manifest_sha256'],baseline_sha256=before['baseline_sha256'],archive_sha256=sha(archive),source_supplement_sha256=sha(RES/'supplementary_source_v4.json'),note='Original parent exit codes for adopted workers are unavailable; actual process exit and saved completion evidence are reported separately')
    save(RES/'supervision_adoption_v4.json',adoption)
    command=['bash','-c','source configs/runtime_env.sh && exec .venv/bin/python -u -m experiments.phase16_r_residual_sac.orchestrate_v4 --adopt-training']
    with (ROOT/'logs/phase16_r/supervisor012.log').open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.1);save(folder/'launch.json',dict(utc=utc(),helper=proc(os.getpid()),supervisor=proc(child.pid),adoption_sha256=sha(RES/'supervision_adoption_v4.json'),source_sha256=sha(Path(__file__)),workers_signalled=0))
        print('UNINTERRUPTED_WORKERS_ADOPTED '+str(child.pid),flush=True);rc=child.wait()
    save(folder/'supervisor_exit.json',dict(utc=utc(),exit_code=rc,after=proc(child.pid)))
    if rc:raise RuntimeError('New CPU manager failed; inspect preserved native trials')
    commands=['analyze','ledger','critic_quality','latency','final_report','audit']
    for name in commands:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run(['.venv/bin/python','-m','experiments.phase16_r_residual_sac.'+name],cwd=ROOT,env=env)
        save(folder/(name+'_exit.json'),dict(utc=utc(),exit_code=result.returncode,selected_gpu=0 if name=='latency' else None))
        if result.returncode:raise RuntimeError('Postprocessing failed '+name)
    save(folder/'complete.json',dict(utc=utc(),passed=True,server_report_and_audit_complete=True,local_delivery_pending=True,visual_review_pending=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        save(RES/'training_handoff_v4/failed.json',dict(utc=utc(),error=traceback.format_exc(),workers_signalled=0));print(traceback.format_exc(),flush=True);sys.exit(1)
