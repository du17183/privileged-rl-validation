"""A single authorized supervisor. No historical runs or workers are restarted."""
import os,sys,json,time,subprocess,traceback,shlex
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,EXP,RES,save,sha,utc,protocol
P=protocol();active={};launched=[];serial=max([int(p.name.split('_')[0]) for p in (EXP/'jobs').glob('*.json')]+[0])
LOG=ROOT/'logs/phase16_r';LOG.mkdir(parents=True,exist_ok=True)

def proc(pid):
    p=Path(f'/proc/{pid}')
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,alive=stat[0] not in ['Z','X'],state=stat[0],ppid=int(stat[1]),start_ticks=int(stat[19]),argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,alive=False)

def resources(gpu):
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free={int(x.split(',')[0]):int(x.split(',')[2]) for x in r.stdout.strip().splitlines()};uuids={int(x.split(',')[0]):x.split(',')[1].strip() for x in r.stdout.strip().splitlines()}
    r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    known={v['pid'] for v in launched}
    for v in active.values():
        childfile=v['out']/'physics_children.json'
        if childfile.exists():
            for row in json.loads(childfile.read_text())['children']:
                recorded=row['process'];current=proc(recorded['pid'])
                if current.get('alive') and all(current.get(k)==recorded.get(k) for k in ['argv','ppid','start_ticks']):known.add(current['pid'])
    external=[dict(gpu_uuid=x.split(',')[0].strip(),pid=int(x.split(',')[1])) for x in r.stdout.strip().splitlines() if x.strip() and int(x.split(',')[1]) not in known]
    others=[x['pid'] for x in external if x['gpu_uuid']==uuids[gpu]]
    return free[gpu]>=P['resources']['minimum_gpu_free_MiB'],dict(utc=utc(),free_MiB=free,unowned_compute_pids=others,external_compute_apps=external,selected_gpu=gpu,loadavg=os.getloadavg(),scope='shared physical evaluation/training permitted when selected GPU free>=frozen minimum65GB; maximum3pairs unchanged; external projects remain fully visible and untouched; deployment latency separately exclusive')

def launch(job,gpu):
    global serial
    okay,snapshot=resources(gpu)
    save(RES/'resource_snapshot.json',snapshot)
    if not okay:return None
    serial+=1;key=f'{serial:03d}_{job["mode"]}_{job.get("arm","A")}_s{job["seed"]}_{job.get("step",0)}'
    jobpath=EXP/'jobs'/f'{key}.json';save(jobpath,job)
    out=ROOT/job['output'];out.mkdir(parents=True,exist_ok=True)
    sock=RES/f's{serial}.sock'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT),PHASE15_1_GRAPH_MEMORY_GIB='48')
    infcmd=[str(ROOT/'.venv_pi05/bin/python'),'-u','-m','evaluation.phase16_r.inference_worker','--socket',str(sock),'--backend','graph']
    driver=job['mode'] in ['engineering','baseline','evalnode']
    if driver:infcmd.append('--persistent')
    module='evaluation.phase16_r.driver' if driver else 'evaluation.phase16_r.runner_v2'
    physcmd=['bash','-c','source configs/runtime_env.sh && exec '+shlex.join([str(ROOT/'.venv/bin/python'),'-u','-m',module,'--job',str(jobpath),'--socket',str(sock),'--device','cuda:0'])]
    logs=[(LOG/(key+'_inference.log')).open('x'),(LOG/(key+'_physics.log')).open('x')]
    inf=subprocess.Popen(infcmd,cwd=ROOT,env=env,stdout=logs[0],stderr=subprocess.STDOUT,start_new_session=True)
    physics=subprocess.Popen(physcmd,cwd=ROOT,env=env,stdout=logs[1],stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.1)
    data=dict(key=key,job=job,gpu=gpu,inf=inf,physics=physics,logs=logs,out=out)
    record=dict(utc=utc(),key=key,gpu=gpu,job=job,job_sha256=sha(jobpath),inference=proc(inf.pid),physics=proc(physics.pid),expected_inference_argv=infcmd,expected_physics_argv=physcmd)
    launched.extend([record['inference'],record['physics']]);save(out/'lifecycle.json',record);active[key]=data
    print('LAUNCHED '+json.dumps(dict(key=key,gpu=gpu,inference_pid=inf.pid,physics_pid=physics.pid)),flush=True)
    return key

def finish_poll(allow_failures=False):
    completed=[]
    for key,v in list(active.items()):
        if v.get('adopted'):
            physics=proc(v['physics_pid']);inference=proc(v['inference_pid'])
            # Never signal or replace the adopted native workers. Their original
            # physics state, model, chunk, Replay and optimizer remain live.
            if physics.get('alive') or inference.get('alive'):continue
            failed=(v['out']/'failed.json').exists();finished=(v['out']/'worker_complete.json').exists()
            v['failed']=bool(failed or not finished)
            save(v['out']/'exit.json',dict(utc=utc(),physics_exit='unknown: original parent was retired',inference_exit='unknown: original parent was retired',physics=physics,inference=inference,signals_sent=[],completion_evidence=finished,technical_failure=failed,adopted_without_worker_interruption=True))
            completed.append(v);del active[key]
            if v['failed'] and not allow_failures:raise RuntimeError('Adopted new worker failed '+key)
            continue
        rc=v['physics'].poll();ir=v['inf'].poll()
        if ir is not None and rc is None and ir!=0:
            if not allow_failures:raise RuntimeError('Inference failed; physics error propagation pending '+key)
            save(v['out']/'stop.json',dict(utc=utc(),reason='owned inference failed; physical worker must preserve its affected trial'))
        if rc is None:continue
        if ir is None:continue
        save(v['out']/'exit.json',dict(utc=utc(),physics_exit=rc,inference_exit=ir,physics=proc(v['physics'].pid),inference=proc(v['inf'].pid),signals_sent=[]))
        for f in v['logs']:f.close()
        v['failed']=bool(rc or ir)
        completed.append(v);del active[key]
        if v['failed'] and not allow_failures:raise RuntimeError('Affected new worker failed; historical runs intact: '+key)
    save(RES/'lifecycle.json',dict(utc=utc(),supervisor=proc(os.getpid()),active=[dict(key=k,gpu=v['gpu'],job=v['job'],physics=proc(v['physics_pid'] if v.get('adopted') else v['physics'].pid),inference=proc(v['inference_pid'] if v.get('adopted') else v['inf'].pid)) for k,v in active.items()],launched=launched))
    return completed

def run_jobs(jobs):
    jobs=list(jobs)
    while jobs or active:
        finish_poll()
        occupied={v['gpu'] for v in active.values()}
        for gpu in [0,1,2]:
            if jobs and gpu not in occupied:
                if launch(jobs[0],gpu):jobs.pop(0)
        time.sleep(3)

def baseline_ready(write=True):
    rows=[]
    for seed in range(5):
        out=RES/f'baseline/seed{seed}'
        assert json.loads((out/'complete.json').read_text())['passed']
        x=json.loads((out/'A_random.json').read_text());assert x['episodes']==128
        rows.append(dict(seed=seed,success=x['success'],episodes=x['episodes'],checkpoint_sha256=x['checkpoint_sha256']))
    gate=RES/'engineering'
    assert json.loads((gate/'physical_gate.json').read_text())['passed']
    assert json.loads((gate/'sac_real_gate.json').read_text())['passed']
    if write:save(RES/'Frozen_pi05_baseline.json',dict(utc=utc(),passed=True,model_seeds=5,rows=rows,mean_success=sum(r['success'] for r in rows)/5,before_any_main_RL=True,physics_gate_sha256=sha(gate/'physical_gate.json'),sac_gate_sha256=sha(gate/'sac_real_gate.json'),protocol_sha256=sha(EXP/'protocol.json')))
    else:
        original=json.loads((RES/'Frozen_pi05_baseline.json').read_text())
        assert original['passed'] and original['before_any_main_RL'] and original['rows']==rows

def training():
    started={(v['job']['arm'],v['job']['seed']) for v in active.values() if v['job']['mode']=='train'}
    pending=[dict(mode='train',seed=s,arm=a,output=f'results/phase16_r/train/{a}_seed{s}') for s in range(3) for a in ['C','D'] if (a,s) not in started]
    processed={str(RES/f'train/{v["job"]["arm"]}_seed{v["job"]["seed"]}/node{v["job"]["step"]}.json') for v in active.values() if v['job']['mode']=='evalnode'}
    evalkey=next((k for k,v in active.items() if v['job']['mode']=='evalnode'),None);streak={};best={};train_finished=[]
    if (RES/'validation_selection.json').exists():best=json.loads((RES/'validation_selection.json').read_text())['best']
    for arm,seed in started:
        key=f'{arm}_seed{seed}';baseline=json.loads((RES/f'baseline/seed{seed}/A_validation.json').read_text())['success'];value=0
        for step in P['evaluation']['nodes']:
            status=RES/f'train/{key}/node{step}.validation_done.json'
            if status.exists():
                processed.add(str(status.with_name(f'node{step}.json')));row=json.loads(status.read_text());value=value+1 if row.get('sampled_success',baseline)<baseline-.20 else 0
        streak[key]=value
    while pending or active:
        for done in finish_poll(allow_failures=True):
            job=done['job']
            if job['mode']=='train':train_finished.append(job)
            else:
                arm=job['arm'];seed=job['seed'];step=job['step'];k=f'{arm}_seed{seed}'
                out=done['out']
                if done['failed']:
                    trainout=RES/f'train/{k}'
                    save(trainout/'stop.json',dict(utc=utc(),reason='independent evaluation technical failure',evaluation=job['output']))
                    save(trainout/f'node{step}.validation_done.json',dict(utc=utc(),passed=False,evaluation_failed=True));evalkey=None;continue
                rate=json.loads((out/'validation_sampled.json').read_text())['success']
                original=json.loads((RES/f'baseline/seed{seed}/A_validation.json').read_text())['success']
                streak[k]=streak.get(k,0)+1 if rate<original-.20 else 0
                if k not in best or rate>best[k]['success']:best[k]=dict(success=rate,step=step,checkpoint=job['checkpoint'],checkpoint_sha256=job['checkpoint_sha256'])
                save(RES/'validation_selection.json',dict(utc=utc(),validation_only=True,best=best))
                trainout=RES/f'train/{k}'
                if streak[k]>=2:save(trainout/'stop.json',dict(utc=utc(),reason='two successive >20pp sampled validation degradations',rate=rate,baseline=original))
                save(trainout/f'node{step}.validation_done.json',dict(utc=utc(),passed=streak[k]<2,sampled_success=rate,baseline_validation=original,independent_evaluation_output=str(out.relative_to(ROOT))))
                evalkey=None
        # Two training pairs retain their live states while a third pair evaluates.
        used={v['gpu'] for v in active.values()}
        for gpu in [0,1]:
            if pending and gpu not in used:
                if launch(pending[0],gpu):pending.pop(0)
        if evalkey is None and 2 not in used:
            candidates=[]
            for v in active.values():
                if v['job']['mode']!='train':continue
                for p in v['out'].glob('node*.json'):
                    if 'validation_done' in p.name or str(p) in processed:continue
                    candidates.append(p)
            if candidates:
                path=sorted(candidates,key=lambda p:p.stat().st_mtime)[0];node=json.loads(path.read_text())
                job=dict(mode='evalnode',**{k:node[k] for k in ['arm','seed','step','checkpoint','checkpoint_sha256']},output=f'results/phase16_r/evaluation/{node["arm"]}_seed{node["seed"]}/step{node["step"]}')
                evalkey=launch(job,2)
                if evalkey:processed.add(str(path))
        time.sleep(3)
    assert len(train_finished)==6
    # Test the preselected best if it differs from final, with no test-driven reselection.
    jobs=[]
    for key,row in best.items():
        if row['step']==50000:continue
        arm,seed=key.split('_seed')
        jobs.append(dict(mode='evalnode',arm=arm,seed=int(seed),step=row['step'],checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],final_test=True,output=f'results/phase16_r/evaluation/{key}/best_test'))
    run_jobs(jobs)
    statuses={}
    for job in train_finished:
        folder=ROOT/job['output'];name=job['arm']+'_seed'+str(job['seed'])
        statuses[name]=json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else json.loads((folder/'stopped.json').read_text()) if (folder/'stopped.json').exists() else dict(passed=False,technical_failure=True,**(json.loads((folder/'technical_fault_preservation.json').read_text()) if (folder/'technical_fault_preservation.json').exists() else {}))
    complete=all(v.get('passed') for v in statuses.values())
    save(RES/('first_round_complete.json' if complete else 'first_round_execution_finished.json'),dict(utc=utc(),passed=complete,trials=6,training_seeds=3,training_interactions=sum(v.get('training_interactions',0) for v in statuses.values()),trial_status=statuses,automatic_extension=False,anchor_qualified=False,LWD_DIVL=False))

def main():
    if (RES/'supervisor_launch.json').exists():raise RuntimeError('An experiment launch already exists; inspect it, never duplicate')
    save(RES/'supervisor_launch.json',dict(utc=utc(),process=proc(os.getpid()),protocol_sha256=sha(EXP/'protocol.json')))
    if '--adopt-training' in sys.argv:
        adoption=json.loads((RES/'supervision_adoption_v4.json').read_text())
        assert adoption['only_CPU_supervisor_retired'] and adoption['workers_signalled']==0
        assert not proc(adoption['prior_supervisor']['pid']).get('alive')
        assert sha(EXP/'protocol.json')==adoption['protocol_sha256']
        assert sha(RES/'source_manifest.json')==adoption['frozen_source_manifest_sha256']
        assert sha(RES/'Frozen_pi05_baseline.json')==adoption['baseline_sha256']
        native_ids={r[k]['pid'] for r in adoption['active'] for k in ['physics','inference']}
        launched.extend(r for r in adoption['prior_launched'] if r['pid'] not in native_ids)
        for row in adoption['active']:
            physics=proc(row['physics']['pid']);inference=proc(row['inference']['pid'])
            for old,current in [(row['physics'],physics),(row['inference'],inference)]:
                if current.get('alive'):assert all(current.get(k)==old.get(k) for k in ['argv','start_ticks']),('Adoption identity mismatch',old,current)
            data=dict(key=row['key'],job=row['job'],gpu=row['gpu'],physics_pid=row['physics']['pid'],inference_pid=row['inference']['pid'],out=ROOT/row['job']['output'],logs=[],adopted=True)
            active[row['key']]=data;launched.extend([physics,inference])
        baseline_ready(write=False);training();return
    if '--probe-only' in sys.argv:
        run_jobs([dict(mode='reset_probe',seed=0,output='results/phase16_r/reset_probe003')]);return
    if '--train-only' in sys.argv:
        baseline_ready();training();return
    run_jobs([dict(mode='engineering',seed=0,output='results/phase16_r/engineering')])
    run_jobs([dict(mode='baseline',seed=s,output=f'results/phase16_r/baseline/seed{s}') for s in range(5)])
    baseline_ready()
    if '--preflight-only' in sys.argv:return
    training()
    print('FIRST_ROUND_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        save(RES/'pipeline_failed.json',dict(utc=utc(),error=traceback.format_exc(),active=[dict(key=k,gpu=v['gpu'],job=v['job'],physics=proc(v['physics_pid'] if v.get('adopted') else v['physics'].pid),inference=proc(v['inference_pid'] if v.get('adopted') else v['inf'].pid)) for k,v in active.items()],signals_sent=[]))
        print(traceback.format_exc(),flush=True);sys.exit(1)
