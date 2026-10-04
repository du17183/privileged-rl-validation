"""Same-S-expert quantity control; do not replace the declared six arms.

The first N is matched to R's controller, but differs from S's planner style.
E_S/F_S additionally isolate ordinary data volume with S's exact planner.
Same frozen common validation criterion and final20k sensitivity as A--F.
"""
import argparse,hashlib,json,shutil,time,subprocess,sys,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .prepare import read,array,sha
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc';L=ROOT/'logs/phase14_2_recovery_bc'
P=ROOT/'datasets/phase14_2/quantity_control_v1';C=ROOT/'checkpoints/phase14_2_recovery_bc/quantity_control'

def prepare():
    original=ROOT/'datasets/phase14_2/prepared_v1';source=ROOT/'datasets/phase14_2/success_size_matched/same_S_expert_v1/trajectories.h5'
    rows=read(source,'N_sameS');keys=sorted(rows)
    if len(keys)!=484:raise RuntimeError('quantity control quota')
    np.random.default_rng(142303).shuffle(keys);splits=dict(train=keys[:338],validation=keys[338:410],test=keys[410:])
    P.mkdir(parents=True,exist_ok=True)
    for s in ['base','recovery']:
        for k in ['train','validation','test']:shutil.copy2(original/f'{s}_{k}.npz',P/f'{s}_{k}.npz')
    for k,ids in splits.items():np.savez(P/f'ordinary_{k}.npz',**array(rows,ids))
    # Common validation target is immutable, identical to all declared arms;
    # retain own-N validation separately for reporting, not model selection.
    shutil.copy2(P/'ordinary_validation.npz',P/'own_N_validation.npz')
    shutil.copy2(original/'ordinary_validation.npz',P/'ordinary_validation.npz')
    m=json.loads((original/'manifest.json').read_text());m['quantity_control']=dict(source=str(source.relative_to(ROOT)),sha256=sha(source),
        splits=splits,planner='Phase13 staged_free unchanged',selection='same frozen common S/R/N_v5 validation as declared A-F',
        paired_updates=20000,batch=512,additional_source_examples=5120000,source_style_amendment='Declared before same-S-expert collection outcomes; initial N matches R controller, additional N_sameS isolates quantity')
    m['dataset_sha256']=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()
    (P/'manifest.json').write_text(json.dumps(m,indent=2))

def fit(seed):
    from . import train
    train.ARMS={'E_S':('robot','ordinary'),'F_S':('full','ordinary')}
    train.main(argparse.Namespace(data=str(P),output=str(C),seed=seed,updates=20000,device='cuda:0'))

def run(module,args,log,gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (L/log).open('a') as f:
        child=subprocess.Popen(['nice','-n','10',sys.executable,'-u','-m',module,*args],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        code=child.wait()
    if code:raise RuntimeError(log)

def pipeline():
    source=ROOT/'datasets/phase14_2/success_size_matched/same_S_expert_v1/summary.json'
    started=time.monotonic()
    while not source.exists():
        if time.monotonic()-started>3600:raise TimeoutError(source)
        time.sleep(5)
    if not (P/'manifest.json').exists():prepare()
    for seed in range(5):
        if not (C/f'bc_seed{seed}_summary.json').exists():run('experiments.phase14_2_recovery_bc.quantity_control',['--mode','fit','--seed',str(seed)],f'quantity_train_seed{seed}.log',2)
    jobs=[]
    for seed in range(5):
        for arm in ['E_S','F_S']:
            for kind in ['best','final']:
                jobs.append(dict(checkpoint=f'checkpoints/phase14_2_recovery_bc/quantity_control/{arm}_seed{seed}_{kind}.pt',
                    output=f'results/phase14_2_recovery_bc/quantity_control/{arm}_seed{seed}_{kind}',tests=['fixed','random'] if kind=='best' else ['random']))
    file=R/'quantity_jobs.json';file.write_text(json.dumps(jobs,indent=2))
    with ThreadPoolExecutor(max_workers=4) as pool:
        work=[pool.submit(run,'experiments.phase14_2_recovery_bc.evaluate',
                         ['--jobs',str(file),'--slot',str(s),'--slots','4','--device','cuda:0'],f'quantity_eval_slot{s}.log',s) for s in range(4)]
        for f in work:f.result()
    missing=[str(ROOT/j['output']/(t+'.json')) for j in jobs for t in j['tests'] if not (ROOT/j['output']/(t+'.json')).exists()]
    if missing:raise RuntimeError(missing)
    (R/'quantity_complete.json').write_text(json.dumps(dict(models=10,tests=30,rl_updates=0)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pipeline','fit'],default='pipeline');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.mode=='fit':fit(a.seed)
    else:pipeline()
