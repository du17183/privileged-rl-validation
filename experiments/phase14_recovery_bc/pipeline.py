"""Conditional recovery gate, paired BC, independent testing; immutable logs."""
import argparse,json,os,subprocess,sys,threading,hashlib,shutil,re
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase14_recovery_bc';LOG=ROOT/'logs/phase14_recovery_bc'
DATA=ROOT/'datasets/recovery_expert';CHECK=ROOT/'checkpoints/phase14_recovery_bc'
LOCK=threading.Lock();JOBS={}
RUN_ID=None;RECIPE={}


def source_recipe():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ['recovery_expert','bc','experiments/phase14_recovery_bc']
        for p in (ROOT/directory).glob('*.py')}


def check_recipe():
    if RECIPE and source_recipe()!=RECIPE:raise RuntimeError('Expert/BC source changed during run; use a new run-id instead of mixing recipes')


def state(name,status,**extra):
    with LOCK:
        JOBS[name]=dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**extra)
        value=dict(owner_pid=os.getpid(),jobs=JOBS)
        temp=R/'pipeline_state.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(R/'pipeline_state.json')
    print(json.dumps(dict(job=name,**JOBS[name])),flush=True)


def run(name,module,arguments,gpu=1,timeout=7200):
    check_recipe()
    path=LOG/(name+'.log')
    if path.exists():raise FileExistsError(path)
    state(name,'running',gpu=gpu)
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(gpu)
    environment['OMP_NUM_THREADS']='4';environment['MKL_NUM_THREADS']='4'
    with path.open('x') as f:
        result=subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:state(name,'failed',exit_code=result.returncode);raise RuntimeError(name)
    check_recipe()
    state(name,'complete',gpu=gpu)


def expert_stage():
    candidate=ROOT/'checkpoints/phase13_random_expert_bc/randomized_gt_candidate_anchor.pt'
    run('recovery_validation','recovery_expert.collect_recovery_data',
        ['--mode','validate','--episodes',160,'--seed',14101,'--save-trajectories','--output',R/'recovery_validation',
         '--checkpoint',candidate,'--device','cuda:0'])
    validation=json.loads((R/'recovery_validation/summary.json').read_text())
    if validation['recovery_success_rate']<=.9:
        state('expert_gate','failed',success=validation['recovery_success_rate'],next='Repair expert; BC/RL not launched')
        return False
    state('expert_gate','passed',success=validation['recovery_success_rate'])
    def controlled(kind,gpu):
        run(f'expert_{kind}','experiments.phase14_recovery_bc.evaluate_recovery',
            ['--expert','--case',kind,'--episodes',64,'--seed',14120,'--output',R/f'expert_controlled/{kind}.json','--device','cuda:0'],gpu)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(controlled,case,gpu) for case,gpu in zip(['contact_loss','ee_offset','door_regression'],[2,3,4])]
        for f in fs:f.result()
    # Formal unselected live-policy gate precedes demonstration selection.
    def recovery():run('recovery_collection','recovery_expert.collect_recovery_data',
        ['--mode','collect','--episodes',200,'--seed',14201,'--output',DATA/'collection_v1','--checkpoint',candidate,'--device','cuda:0'],1)
    def ordinary():run('ordinary_control_collection','recovery_expert.collect_ordinary_data',
        ['--mode','collect','--episodes',200,'--seed',14202,'--output',DATA/'ordinary_control_v1','--device','cuda:0'],2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(recovery),pool.submit(ordinary)]
        for f in fs:f.result()
    recovery_summary=json.loads((DATA/'collection_v1/summary.json').read_text())
    if recovery_summary['recovery_success_rate']<=.9:
        state('collection_expert_gate','failed',success=recovery_summary['recovery_success_rate'],next='Collection distribution gate failed; no BC')
        return False
    run('prepare','experiments.phase14_recovery_bc.prepare',['--run-id',RUN_ID] if RUN_ID else [],gpu=1)
    return True


def bc_stage():
    with ThreadPoolExecutor(max_workers=5) as pool:
        fs=[pool.submit(run,f'bc_seed{seed}','bc.recovery_bc_train',
            ['--data',DATA/'split_v1','--output',CHECK,'--seed',seed,'--device','cuda:0'],seed) for seed in range(5)]
        for f in fs:f.result()
    state('bc_training_all','complete')


def evaluate_seed(seed):
    for arm in 'ABCDEF':
        checkpoint=CHECK/f'{arm}_seed{seed}_best.pt'
        for kind,level,episodes in [('random',2,128),('fixed',0,64)]:
            run(f'{kind}_{arm}_seed{seed}','experiments.phase13_random_expert_bc.evaluate',
                ['--checkpoint',checkpoint,'--output',R/f'{kind}/{arm}_seed{seed}.json','--episodes',episodes,
                 '--level',level,'--num-envs',32,'--seed',14501+seed,'--device','cuda:0'],seed)
        for case in ['contact_loss','ee_offset','door_regression']:
            run(f'{case}_{arm}_seed{seed}','experiments.phase14_recovery_bc.evaluate_recovery',
                ['--checkpoint',checkpoint,'--output',R/f'recovery/{case}_{arm}_seed{seed}.json','--episodes',64,
                 '--case',case,'--seed',14601+seed,'--device','cuda:0'],seed)
    state(f'evaluations_seed{seed}','complete')


def main(a):
    global R,LOG,DATA,CHECK,RUN_ID,RECIPE
    if a.run_id:
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',a.run_id):raise ValueError('Use a safe unique run-id')
        RUN_ID=a.run_id;R=R/a.run_id;LOG=LOG/a.run_id;DATA=DATA/a.run_id;CHECK=CHECK/a.run_id
    for p in [R,LOG,DATA,CHECK]:p.mkdir(parents=True,exist_ok=True)
    RECIPE=source_recipe();snapshot=R/'source_recipe.json'
    if snapshot.exists() and json.loads(snapshot.read_text())!=RECIPE:raise RuntimeError('Run source changed; choose a new run-id')
    if not snapshot.exists():
        snapshot.write_text(json.dumps(RECIPE,indent=2))
        for name in RECIPE:
            target=R/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    protocol=json.loads((ROOT/'experiments/phase14_recovery_bc/protocol.json').read_text())
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2))
    if a.stage in ['expert','all']:
        if not expert_stage():return
    if a.stage in ['bc','all']:bc_stage()
    if a.stage in ['evaluation','all']:
        with ThreadPoolExecutor(max_workers=5) as pool:
            fs=[pool.submit(evaluate_seed,seed) for seed in range(5)]
            for f in fs:f.result()
        state('bc_evaluation_all','complete',next='paired inference and conditional anchor qualification; no RL without stable anchor')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['expert','bc','evaluation','all'],default='all');p.add_argument('--run-id')
    try:main(p.parse_args())
    except Exception as e:state('pipeline','failed',error=repr(e));raise
