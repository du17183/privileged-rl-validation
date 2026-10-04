"""Mechanism diagnostics after the full five-seed primary comparison."""
import json
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.phase13_random_expert_bc.pipeline import ROOT,RESULT,CHECK,run,state
from experiments.phase13_random_expert_bc.analyze import primary


def main():
    primary(RESULT)
    run('action_sensitivity','experiments.phase13_random_expert_bc.analyze',['--sensitivity'])
    run('coverage_diagnosis','experiments.phase13_random_expert_bc.coverage',[],1200)
    run('settling_probe','experiments.phase13_random_expert_bc.settling_probe',['--device','cuda:0'],600)
    def mask_job(mask):
        return ('mask_'+mask,'experiments.phase13_random_expert_bc.evaluate',
                ['--checkpoint',CHECK/'B_seed0_best.pt','--output',RESULT/f'masking/{mask}.json',
                 '--episodes',64,'--num-envs',32,'--seed',13601,'--mask',mask,'--device','cuda:0'],1800)
    # The two required null controls also serve as a bounded parallel preflight.
    # Independent 32-clone processes preserve the established physical layout.
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(run,*mask_job(mask)) for mask in ['none','none_repeat']]
        for task in tasks:task.result()
    first=json.loads((RESULT/'masking/none.json').read_text())
    repeat=json.loads((RESULT/'masking/none_repeat.json').read_text())
    left=list(csv.DictReader((RESULT/'masking/none.csv').open()))
    right=list(csv.DictReader((RESULT/'masking/none_repeat.csv').open()))
    cols=['clone','episode','angle0_rad','offset_x','offset_y','offset_z','friction_scale']
    a=sorted(tuple(r[c] for c in cols) for r in left);b=sorted(tuple(r[c] for c in cols) for r in right)
    delta=abs(first['success']-repeat['success'])
    passed=a==b and delta<=2/64
    (RESULT/'diagnostic_concurrency_preflight.json').write_text(json.dumps(dict(
        workers=2,physical_gpu=1,clones_per_process=32,total_episodes=128,parameters_exact_match=a==b,
        success_difference=delta,passed=passed,external_processes_signaled=0),indent=2))
    jobs=[mask_job(mask) for mask in ['angle_only','coherent_angle_progress_remaining','handle_pose','contact','all_environment']]
    for seed in range(5):
        jobs.append((f'stochastic_B_seed{seed}','experiments.phase13_random_expert_bc.evaluate',
            ['--checkpoint',CHECK/f'B_seed{seed}_best.pt','--output',RESULT/f'stochastic/B_seed{seed}.json',
             '--episodes',64,'--num-envs',32,'--seed',13801+seed,'--stochastic','--device','cuda:0'],1800))
    # If null reproducibility fails, retain evidence and complete remaining tests
    # serially; no result is overwritten or silently selected.
    with ThreadPoolExecutor(max_workers=2 if passed else 1) as pool:
        tasks=[pool.submit(run,*job) for job in jobs]
        for task in tasks:task.result()
    run('bc_report','experiments.phase13_random_expert_bc.report',[])
    state('bc_diagnostics_complete',status='complete',next='review anchor gate; RL remains conditional')


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('post_bc_failed',status='failed',error=repr(error));raise
