"""Authorized staged run: pilot closed-loop gate, then paired five-seed fits."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
from runtime_paths import PI05_PYTHON
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc';L=ROOT/'logs/phase14_3_pi05_bc';C=ROOT/'checkpoints/phase14_3_pi05_bc'
ISAAC=str(ROOT/'.venv/bin/python');PI=PI05_PYTHON
TESTS=['fixed','random','ee_offset','contact_loss','door_regression','stagnation','natural_severe']
PROCS={};JOBS=json.loads((R/'lifecycle.json').read_text()) if (R/'lifecycle.json').exists() else {}

def record():
    temp=R/'lifecycle.tmp';temp.write_text(json.dumps(JOBS,indent=2));os.replace(temp,R/'lifecycle.json')
def launch(name,args,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    f=(L/(name+'.log')).open('a');p=subprocess.Popen(['nice','-n','10',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);f.close()
    PROCS[name]=p;JOBS[name]={'status':'running','pid':p.pid,'gpu':gpu,'started':time.time()};record();return p
def wait(names):
    while True:
        finished=True
        for name in names:
            p=PROCS[name];code=p.poll()
            if code is None:finished=False;continue
            if JOBS[name]['status']=='running':JOBS[name].update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time());record()
            if code:raise RuntimeError((name,code))
        if finished:return
        time.sleep(3)
def job(arm,seed,step,split):
    validation=split=='validation'
    return dict(checkpoint=f'checkpoints/phase14_3_pi05_bc/{arm}_seed{seed}_step{step}.pt',
       output=f'results/phase14_3_pi05_bc/{split}/{arm}_seed{seed}'+(f'/step{step}' if validation else ''),
       tests=TESTS,seed=143501 if validation else 143502,fixed_n=32 if validation else 64,
       random_n=64 if validation else 128,cohorts=f'results/phase14_3_pi05_bc/cohorts/{split}',split=split)
def evaluate(jobs,name,slots=8):
    file=R/(name+'_jobs.json');file.write_text(json.dumps(jobs,indent=2));names=[]
    for slot in range(min(slots,len(jobs))):
        sock=R/f'{name}_rpc{slot}.sock'
        if any(j['checkpoint'].split('/')[-1].startswith(('C_','D_')) for j in jobs[slot::slots]):
            launch(f'{name}_server{slot}',[PI,'-u','-m','pi05.inference.server','--socket',str(sock)],slot)
        worker=f'{name}_eval{slot}';launch(worker,[ISAAC,'-u','-m','evaluation.pi05_closed_loop_eval','--jobs',str(file),'--slot',str(slot),'--slots',str(slots),'--socket',str(sock),'--device','cuda:0'],slot);names.append(worker)
    wait(names)
    for j in jobs:
        for test in j['tests']:
            path=ROOT/j['output']/(test+'.json')
            if not path.exists():raise RuntimeError(('missing evaluation',str(path)))
    servers=[n for n in PROCS if n.startswith(name+'_server')]
    if servers:wait(servers)

def select():
    chosen={}
    for arm in ['A','B','C','D']:
        for seed in range(5):
            steps=[1000,5000,10000,15000,20000] if arm in ['A','B'] else [250,1250,2500,3750,5000];scores=[]
            for step in steps:
                directory=R/'validation'/f'{arm}_seed{seed}'/f'step{step}'
                d={t:json.loads((directory/(t+'.json')).read_text()) for t in TESTS}
                priority=(d['random']['success'],sum(d[t]['success'] for t in TESTS[2:6])/4,d['natural_severe']['success'],d['fixed']['success'],-step)
                scores.append((priority,step))
            priority,step=max(scores);chosen[f'{arm}_seed{seed}']={'step':step,'priority':priority,'selection':'independent closed-loop validation only'}
    (R/'selection.json').write_text(json.dumps(chosen,indent=2));return chosen

def main():
    R.mkdir(exist_ok=True);L.mkdir(exist_ok=True)
    # Relocate the finished preflight fit into a pilot-only namespace.
    if (C/'D_seed0_complete.json').exists():
        meta=json.loads((C/'D_seed0_complete.json').read_text())
        if meta['updates']==50:
            dest=C/'pilot';dest.mkdir(exist_ok=True)
            for name in ['D_seed0_complete.json','D_seed0_training.csv','D_seed0_step50.pt','D_seed0_step50.ready.json']:
                old=C/name;new=dest/name
                assert old.resolve().is_relative_to(C.resolve()) and new.resolve().is_relative_to(C.resolve()) and not new.exists()
                old.rename(new)
    setup=[]
    for split,n,seed,gpu in [('validation',32,143601,6),('test',64,143701,7)]:
        cohort=R/'cohorts'/split
        if all((cohort/(t+'.json')).exists() for t in TESTS[2:]):continue
        name='cohorts_'+split;launch(name,[ISAAC,'-u','-m','experiments.phase14_3_pi05_bc.cohorts','--output',str(cohort),
          '--checkpoint',str(ROOT/'checkpoints/phase13_random_expert_bc/randomized_gt_candidate_anchor.pt'),
          '--seed-base',str(seed),'--episodes',str(n),'--device','cuda:0'],gpu);setup.append(name)
    for seed in range(5):
        if (C/f'mse_seed{seed}_complete.json').exists():continue
        name=f'mse_seed{seed}';launch(name,[ISAAC,'-u','-m','experiments.phase14_3_pi05_bc.train_mse','--seed',str(seed)],seed);setup.append(name)
    for seed in range(5):
        if (C/f'mse_seed{seed}_early_complete.json').exists():continue
        name=f'mse_early_seed{seed}';launch(name,[ISAAC,'-u','-m','experiments.phase14_3_pi05_bc.train_mse','--seed',str(seed),'--early-only'],seed);setup.append(name)
    wait(setup)
    if not (R/'pilot_gate.json').exists():
        if not (C/'pilot/D_seed0_complete.json').exists():
            launch('pilot_train_v2',[PI,'-u','-m','pi05.train','--arm','D','--seed','0','--updates','50'],0)
            wait(['pilot_train_v2'])
        j=job('D',0,50,'validation');j['checkpoint']='checkpoints/phase14_3_pi05_bc/pilot/D_seed0_step50.pt';j['output']='results/phase14_3_pi05_bc/pilot_evaluation'
        # Resume the already-running pilot without touching its simulator or
        # inference worker. This supervisor restart only adds early candidates.
        previous=JOBS.get('pilot_eval0',{})
        pid=previous.get('pid',-1);proc=Path('/proc')/str(pid)/'cmdline'
        running=proc.exists() and b'evaluation.pi05_closed_loop_eval' in proc.read_bytes()
        if running:
            deadline=time.monotonic()+1800
            while proc.exists() and proc.read_bytes():
                if time.monotonic()>deadline:raise TimeoutError('Existing pilot did not finish')
                time.sleep(3)
            for name in ['pilot_eval0','pilot_server0']:
                JOBS[name].update(status='complete',resumed_completion=True,finished=time.time())
            record()
        if not all((ROOT/j['output']/(t+'.json')).exists() for t in TESTS):evaluate([j],'pilot',slots=1)
        results=[json.loads((ROOT/j['output']/(t+'.json')).read_text()) for t in TESTS]
        assert all(d['inference']['handoff_replans']==d['episodes'] for d in results[2:]),'Missing pressure handoff replan'
        gate={'pretrained_forward_backward':True,'checkpoint_restore':True,'gt_forward_sensitivity':True,
          'closed_loop_all7_tests':len(results)==7,'chunk_cache_reset_handoff':True,'rl_updates':0,
          'performance_interpretation':'50-update fit is only interface validation; not included in formal comparison'}
        (R/'pilot_gate.json').write_text(json.dumps(gate,indent=2))
    # The five-seed model runs start only AFTER the pilot closed-loop gate.
    pending=[(arm,seed) for seed in range(5) for arm in ['C','D'] if not (C/f'{arm}_seed{seed}_complete.json').exists()]
    active={}
    while pending or active:
        for gpu in range(8):
            if gpu in active:continue
            if not pending:break
            arm,seed=pending.pop(0);name=f'pi05_{arm}_seed{seed}'
            launch(name,[PI,'-u','-m','pi05.train','--arm',arm,'--seed',str(seed)],gpu);active[gpu]=name
        for gpu,name in list(active.items()):
            code=PROCS[name].poll()
            if code is None:continue
            JOBS[name].update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time());record()
            if code:raise RuntimeError((name,code))
            del active[gpu]
        time.sleep(3)
    val=[job(arm,seed,step,'validation') for seed in range(5) for arm in ['A','B','C','D'] for step in ([1000,5000,10000,15000,20000] if arm in ['A','B'] else [250,1250,2500,3750,5000])]
    evaluate(val,'validation');chosen=select()
    tests=[job(arm,seed,chosen[f'{arm}_seed{seed}']['step'],'test') for seed in range(5) for arm in ['A','B','C','D']]
    evaluate(tests,'test')
    subprocess.run([ISAAC,'-m','experiments.phase14_3_pi05_bc.audit','--check'],cwd=ROOT,check=True)
    (R/'experiment_complete.json').write_text(json.dumps({'models':20,'validation_checkpoints':100,'test_models':20,'rl_updates':0}));print('ALL EXPERIMENTS COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (R/'pipeline_failed.txt').write_text(traceback.format_exc());raise
