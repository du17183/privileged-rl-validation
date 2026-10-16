"""Continue only after the existing baseline supervisor and every child exited."""
import sys,os,json,time,tarfile,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,sha,save,utc,protocol
from experiments.phase16_r_residual_sac.monitor import proc

def same_live(old):
    current=proc(old['pid'])
    return current.get('alive') and all(current.get(k)==old.get(k) for k in ['argv','ppid','start_ticks'])
def main():
    protocol();folder=RES/'training_handoff_v2';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'launch.json').exists():raise RuntimeError('Training handoff already exists; inspect it')
    prior=json.loads((RES/'supervisor_launch.json').read_text());original=prior['process']
    save(folder/'launch.json',dict(utc=utc(),process=proc(os.getpid()),wait_for=original,source_sha256=sha(Path(__file__)),no_workers_started=True))
    deadline=time.monotonic()+7200
    while same_live(original):
        if (RES/'pipeline_failed.json').exists():raise RuntimeError('Baseline technical failure; training remains forbidden')
        save(folder/'progress.json',dict(utc=utc(),phase='waiting_for_current_baseline',supervisor=proc(original['pid']),main_RL_started=False))
        if time.monotonic()>deadline:raise RuntimeError('Baseline handoff wait expired; no training launched')
        time.sleep(15)
    life=json.loads((RES/'lifecycle.json').read_text());expected=[life['supervisor'],*life.get('launched',[])]
    for path in RES.rglob('physics_children.json'):expected.extend(r['process'] for r in json.loads(path.read_text())['children'])
    assert not any(same_live(p) for p in expected),'Owned baseline child remains alive'
    assert not (RES/'pipeline_failed.json').exists(),'Baseline pipeline failed'
    baseline=json.loads((RES/'Frozen_pi05_baseline.json').read_text());assert baseline['passed'] and len(baseline['rows'])==5 and baseline['before_any_main_RL']
    for seed in range(5):assert json.loads((RES/f'baseline/seed{seed}/complete.json').read_text())['passed']
    for name in ['physical_gate','sac_real_gate']:assert json.loads((RES/f'engineering/{name}.json').read_text())['passed']
    oldmanifest=json.loads((RES/'source_manifest.json').read_text())
    for rel,wanted in oldmanifest['files'].items():assert sha(ROOT/rel)==wanted,'Execution source SHA drift '+rel
    historical=json.loads((RES/'historical_inputs.json').read_text())
    for rel,wanted in historical['protected'].items():assert sha(ROOT/rel)==wanted,'Historical source/data/result SHA drift '+rel
    for row in historical['normal'].values():assert sha(ROOT/row['checkpoint'])==row['sha256']
    version=EXP/'source_versions/pretraining010';version.mkdir(parents=True,exist_ok=False)
    originals=[ROOT/rel for rel in oldmanifest['files']]+[RES/'source_manifest.json',RES/'supervisor_launch.json',RES/'lifecycle.json',RES/'Frozen_pi05_baseline.json']
    originals+=list((ROOT/'logs/phase16_r').glob('00[6-9]*.log'))+list((ROOT/'logs/phase16_r').glob('01[0-1]*.log'))
    originals+=list((RES/'engineering').glob('*.json'))+list((RES/'baseline').glob('seed*/*.json'))+list((RES/'baseline').glob('seed*/*.csv'))
    unique={str(p.relative_to(ROOT)):p for p in originals if p.exists() and p.is_file()};archive=version/'baseline_source_logs_outputs.tar.gz'
    hashes={rel:sha(path) for rel,path in unique.items()}
    with tarfile.open(archive,'w:gz') as t:
        for rel,path in unique.items():t.add(path,arcname=rel)
    save(version/'manifest.json',dict(utc=utc(),original_files=hashes,archive_sha256=sha(archive),prior_protocol_sha256=sha(EXP/'protocol.json'),original_identity_exit=True,signals_sent=[]))
    (RES/'supervisor_launch.json').rename(version/'supervisor_launch.json')
    files={}
    for root in [EXP,ROOT/'algorithms/residual_sac',ROOT/'evaluation/phase16_r']:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ['.py','.json','.txt'] and 'source_versions' not in path.parts:files[str(path.relative_to(ROOT))]=sha(path)
    save(RES/'source_manifest.json',dict(utc=utc(),files=files,protocol_sha256=sha(EXP/'protocol.json'),before_main_training=True,baseline_manifest_sha256=sha(version/'source_manifest.json') if (version/'source_manifest.json').exists() else hashes['results/phase16_r/source_manifest.json'],baseline_source_archive_sha256=sha(archive)))
    command=['bash','-c','source configs/runtime_env.sh && exec .venv/bin/python -u -m experiments.phase16_r_residual_sac.orchestrate_v2 --train-only']
    logfile=ROOT/'logs/phase16_r/supervisor010.log'
    with logfile.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.1);save(folder/'training_launch.json',dict(utc=utc(),process=proc(child.pid),source_manifest_sha256=sha(RES/'source_manifest.json'),Frozen_pi05_baseline_sha256=sha(RES/'Frozen_pi05_baseline.json'),prior_processes_actual_exited=True,prior_pid_reuse_is_not_live_experiment=True))
        print('MAIN_RL_LAUNCHED '+str(child.pid),flush=True);rc=child.wait()
    save(folder/'training_exit.json',dict(utc=utc(),exit_code=rc,after=proc(child.pid),signals_sent=[]))
    if rc:raise RuntimeError('Training supervisor failed; inspect affected trial before further work')
    # Postprocessing is independent of learned control, and does not change the
    # protocol or claim success for scientifically negative/early-stop results.
    commands=[
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.analyze'],
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.ledger'],
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.critic_quality'],
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.latency'],
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.final_report'],
        ['.venv/bin/python','-m','experiments.phase16_r_residual_sac.audit']]
    for command in commands:
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run(command,cwd=ROOT,env=env,check=False)
        if result.returncode:raise RuntimeError('Affected postprocessing failed '+str(command))
    save(folder/'complete.json',dict(utc=utc(),passed=True,server_report_and_audit_complete=True,local_delivery_pending=True,visual_review_pending=True))
    print('SERVER_REPORT_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        save(RES/'training_handoff_v2/failed.json',dict(utc=utc(),error=traceback.format_exc(),signals_sent=[]));print(traceback.format_exc(),flush=True);sys.exit(1)
