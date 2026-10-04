"""Complete diagnostics after the formal validation/test pipeline finishes."""
import json,os,subprocess,time
from pathlib import Path
from runtime_paths import PI05_PYTHON
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_3_pi05_bc';L=ROOT/'logs/phase14_3_pi05_bc'
ISAAC=str(ROOT/'.venv/bin/python');PI=PI05_PYTHON

def command(name,args,gpu=None):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (L/(name+'.log')).open('a') as f:
        subprocess.run(['nice','-n','10',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    while not (R/'experiment_complete.json').exists():
        if (R/'pipeline_failed.txt').exists():raise RuntimeError((R/'pipeline_failed.txt').read_text())
        time.sleep(30)
    from .run import evaluate,job
    command('analyze',[ISAAC,'-u','-m','experiments.phase14_3_pi05_bc.analyze'])
    command('mse_checkpoint_diagnosis',[ISAAC,'-u','-m','evaluation.mse_checkpoint_diagnosis'])
    chosen=json.loads((R/'selection.json').read_text());jobs=[]
    for seed in range(5):
        for ablation in ['all_gt','handle_pose']:
            j=job('D',seed,chosen[f'D_seed{seed}']['step'],'test');j['tests']=['random'];j['ablation']=ablation
            j['output']=f'results/phase14_3_pi05_bc/gt_ablation/D_seed{seed}_{ablation}';jobs.append(j)
    evaluate(jobs,'gt_ablation')
    command('action_diagnosis',[PI,'-u','-m','evaluation.pi05_action_diagnosis'],0)
    command('isaac_provenance_final',[ISAAC,'-m','experiments.phase14_3_pi05_bc.provenance','--name','isaac_final'])
    command('pi05_provenance_final',[PI,'-m','experiments.phase14_3_pi05_bc.provenance','--name','pi05_final','--full'])
    command('report',[ISAAC,'-u','-m','experiments.phase14_3_pi05_bc.report'])
    command('audit_final',[ISAAC,'-m','experiments.phase14_3_pi05_bc.audit','--check'])
    (R/'delivery_complete.json').write_text(json.dumps(dict(complete=True,rl_updates=0,finished=time.time())))
    print('PHASE14.3 REPORT AND DIAGNOSTICS COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        (R/'finalize_failed.txt').write_text(traceback.format_exc());raise
