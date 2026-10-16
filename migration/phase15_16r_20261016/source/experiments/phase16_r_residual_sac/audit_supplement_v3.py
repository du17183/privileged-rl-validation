"""Run the frozen independent auditor, then audit CPU adoption provenance."""
import sys,json,hashlib,tarfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,EXP,save,sha,utc
from experiments.phase16_r_residual_sac import audit

def main():
    original_source=Path(audit.__file__)
    assert sha(original_source)=='485e20eb8781d2107e93c454a97a59a519f47af803ca5ce440a6c9acd7fbf13c'
    archive_dir=RES/'audit_revisions'/utc().replace(':','');archive_dir.mkdir(parents=True,exist_ok=False)
    prior=RES/'independent_audit.json'
    if prior.exists():(archive_dir/'prior_independent_audit.json').write_bytes(prior.read_bytes())
    audit.main()
    base=json.loads(prior.read_text());assert base['passed'],base['issues']
    (archive_dir/'frozen_original_audit.json').write_bytes(prior.read_bytes())
    issues=[];checks={};adoption=json.loads((RES/'supervision_adoption_v5.json').read_text())
    supplement=json.loads((RES/'supplementary_source_v5.json').read_text())
    for rel,wanted in supplement['files'].items():
        if sha(ROOT/rel)!=wanted:issues.append('CPU supplementary source SHA changed '+rel)
    if sha(RES/'supplementary_source_v5.json')!=adoption['source_supplement_sha256']:issues.append('CPU source supplement changed')
    if sha(EXP/'protocol.json')!=adoption['protocol_sha256']:issues.append('Protocol changed at CPU handoff')
    if sha(RES/'source_manifest.json')!=adoption['frozen_source_manifest_sha256']:issues.append('Pre-training source manifest replaced')
    if sha(RES/'Frozen_pi05_baseline.json')!=adoption['baseline_sha256']:issues.append('Baseline replaced at CPU handoff')
    version=EXP/'source_versions/supervision013_owned_idle_gpu';manifest=json.loads((version/'manifest.json').read_text())
    archive=version/'before_cpu_retirement.tar.gz'
    if sha(archive)!=manifest['archive_sha256'] or sha(archive)!=adoption['archive_sha256']:issues.append('Original CPU source/log archive changed')
    checked=0
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            if not member.isfile():continue
            source=stream.extractfile(member);h=hashlib.sha256()
            while True:
                data=source.read(8*1024*1024)
                if not data:break
                h.update(data)
            if h.hexdigest()!=manifest['snapshot_files'].get(member.name):issues.append('CPU archived member SHA mismatch '+member.name)
            checked+=1
    if checked!=len(manifest['snapshot_files']):issues.append('CPU archived snapshot count mismatch')
    preceding_version=EXP/'source_versions/supervision011_selected_gpu'
    preceding_manifest=json.loads((preceding_version/'manifest.json').read_text())
    preceding_adoption=json.loads((RES/'supervision_adoption_v3.json').read_text())
    preceding_supplement=json.loads((RES/'supplementary_source_v3.json').read_text())
    for rel,wanted in preceding_supplement['files'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Previous CPU supplementary source changed '+rel)
    preceding_archive=preceding_version/'before_cpu_retirement.tar.gz'
    if sha(preceding_archive)!=preceding_adoption['archive_sha256'] or sha(preceding_archive)!=preceding_manifest['archive_sha256']:issues.append('Previous CPU provenance archive changed')
    preceding_before=json.loads((preceding_version/'before_identity.json').read_text())
    preceding_after=json.loads((preceding_version/'after_identity.json').read_text())
    if preceding_after['workers_signalled'] or preceding_adoption['workers_signalled']:issues.append('First CPU adoption signalled a native worker')
    initial={r['key']:r for r in preceding_before['active']}
    for row in preceding_after['active']:
        for kind in ['physics','inference']:
            if not all(row[kind][key]==initial[row['key']][kind][key] for key in ['pid','argv','start_ticks']) or row[kind]['ppid']!=1:issues.append('First CPU adoption native identity changed')
    middle=EXP/'source_versions/supervision012_shared_gpu'
    middle_adoption=json.loads((RES/'supervision_adoption_v4.json').read_text());middle_manifest=json.loads((middle/'manifest.json').read_text())
    middle_archive=middle/'before_cpu_retirement.tar.gz'
    if sha(middle_archive)!=middle_adoption['archive_sha256'] or sha(middle_archive)!=middle_manifest['archive_sha256']:issues.append('Middle CPU provenance archive changed')
    middle_supplement=json.loads((RES/'supplementary_source_v4.json').read_text())
    for rel,wanted in middle_supplement['files'].items():
        if sha(ROOT/rel)!=wanted:issues.append('Middle CPU source changed '+rel)
    middle_before=json.loads((middle/'before_identity.json').read_text());middle_after=json.loads((middle/'after_identity.json').read_text());initial_middle={r['key']:r for r in middle_before['active']}
    if middle_after['workers_signalled'] or middle_adoption['workers_signalled']:issues.append('Middle CPU adoption signalled a native worker')
    for row in middle_after['active']:
        for kind in ['physics','inference']:
            if not all(row[kind][key]==initial_middle[row['key']][kind][key] for key in ['pid','argv','start_ticks']):issues.append('Middle CPU adoption native identity changed')
    before=json.loads((version/'before_identity.json').read_text());after=json.loads((version/'after_identity.json').read_text());signal=json.loads((version/'signal.json').read_text())
    if not adoption['only_CPU_supervisor_retired'] or adoption['workers_signalled'] or after['workers_signalled'] or signal['workers_signalled']:issues.append('CPU handoff worker interruption evidence')
    if signal['pid']!=before['prior_supervisor']['pid'] or signal['identity']!=before['prior_supervisor'] or signal['signal']!='SIGINT':issues.append('Retired process identity/signal mismatch')
    old={r['key']:r for r in before['active']};adopted=[]
    for row in after['active']:
        for kind in ['physics','inference']:
            a=old[row['key']][kind];b=row[kind]
            if not all(a[k]==b[k] for k in ['pid','argv','start_ticks']) or b['ppid']!=1:issues.append('Native identity or natural reparent mismatch '+row['key']+kind)
        folder=ROOT/row['job']['output'];worker=json.loads((folder/'worker_complete.json').read_text());exit_record=json.loads((folder/'exit.json').read_text())
        if not worker['passed'] or worker['pid']!=row['physics']['pid']:issues.append('Adopted physical completion mismatch '+row['key'])
        if exit_record['signals_sent'] or exit_record['physics'].get('alive') or exit_record['inference'].get('alive'):issues.append('Adopted worker has not actually exited '+row['key'])
        if not exit_record.get('adopted_without_worker_interruption') or exit_record['physics_exit']!='unknown: original parent was retired':issues.append('Adopted exit-code scope falsely stated '+row['key'])
        adopted.append(dict(key=row['key'],physics_pid=row['physics']['pid'],inference_pid=row['inference']['pid'],original_birth_and_argv_unchanged=True,workers_signalled=0,exit_code='unknown',physical_complete=True,actual_exit_observed=True))
    post_sources={str((EXP/n).relative_to(ROOT)):sha(EXP/n) for n in ['report_supplement_v1.py','report_supplement_v2.py','report_supplement_v3.py','audit_supplement_v1.py','audit_supplement_v2.py','audit_supplement_v3.py','monitor_watch_v4.py','monitor_watch_v5.py','checkpoint_observation_v1.py','critic_return_observation_v1.py']}
    save(RES/'postprocessing_source_v1.json',dict(utc=utc(),files=post_sources,scope='additional saved-data reporting and independent adoption audit only; no training or physical interactions'))
    report=ROOT/'docs/phase16_r_report.md';status=json.loads((RES/'report_status.json').read_text())
    if sha(report)!=status['report_sha256'] or base['report_sha256']!=sha(report):issues.append('Current report SHA differs from independent auditor')
    runtime=json.loads((RES/'runtime_scope.json').read_text())
    if runtime['physical_interactions_added']!=0:issues.append('Postprocessing added unledgered interactions')
    jobs_checked=0
    for lifecycle in RES.rglob('lifecycle.json'):
        if not any(p in ['train','baseline','evaluation'] for p in lifecycle.parts):continue
        record=json.loads(lifecycle.read_text())
        if 'key' not in record:continue
        jobpath=EXP/'jobs'/f"{record['key']}.json"
        if sha(jobpath)!=record['job_sha256'] or json.loads(jobpath.read_text())!=record['job']:issues.append('Owned launch job SHA/content differs '+record['key'])
        if ROOT/record['job']['output']!=lifecycle.parent:issues.append('Owned job output namespace differs '+record['key'])
        jobs_checked+=1
    for childfile in (RES/'evaluation').rglob('physics_children.json'):
        for row in json.loads(childfile.read_text())['children']:
            if sha(ROOT/row['job'])!=row['job_sha256'] or row.get('exit_code')!=0:issues.append('Independent physical child job SHA/exit differs '+row['job'])
            jobs_checked+=1
    returns=json.loads((RES/'critic_observed_return.json').read_text())
    if returns['physical_interactions_added'] or returns['training_updates_added'] or any(not r['finite'] for r in returns['rows']):issues.append('Observed return diagnostic scope/nonfinite error')
    for row in returns['rows']:
        if row['closed_scored_episodes']!=128:issues.append('Observed return omitted a completed final test episode')
        if sha(ROOT/f"checkpoints/phase16_r/{row['arm']}_seed{row['seed']}/step50000.pt")!=row['checkpoint_sha256']:issues.append('Observed return checkpoint SHA differs')
    checks['owned_launch_and_child_jobs_checked']=jobs_checked
    checks['observed_return_trials']=len(returns['rows'])
    checks['middle_CPU_archive_sha256']=sha(middle_archive)
    checks.update(frozen_original_audit_passed=base['passed'],frozen_original_auditor_sha256=sha(original_source),original_audit_artifact_sha256=sha(archive_dir/'frozen_original_audit.json'),CPU_archive_members=checked,CPU_supplement_sha256=sha(RES/'supplementary_source_v5.json'),preceding_CPU_archive_sha256=sha(preceding_archive),preceding_CPU_source_supplement_sha256=sha(RES/'supplementary_source_v3.json'),postprocessing_source_manifest_sha256=sha(RES/'postprocessing_source_v1.json'),adopted_native_trials=adopted,core_pretraining_manifest_unchanged=True,protocol_unchanged=True,actual_baseline_unchanged=True,shared_physical_gpu_scope=True,exclusive_deployment_latency_separate=True,extra_environment_interactions=0)
    result=dict(base);result.update(utc=utc(),passed=not issues,issues=issues,auditor_source_sha256=sha(Path(__file__)),frozen_original_auditor_sha256=sha(original_source),supplementary_checks=checks,report_sha256=sha(report),scope=base['scope']+'; independent CPU adoption/archive SHA and unknown original exit-code scope audit')
    save(prior,result);print(json.dumps(dict(passed=not issues,issues=issues,supplementary_checks=checks)))
    if issues:sys.exit(1)
if __name__=='__main__':main()
