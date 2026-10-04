"""Separately registered operational anchor selection and untouched retest.

The primary five-seed gate is not changed or reclassified. This secondary
selection is reported explicitly, with training-seed variance still unresolved.
No candidate is retrained and no RL starts before independent retesting.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from experiments.phase13_random_expert_bc.pipeline import ROOT,RESULT,CHECK,run,state
from experiments.phase13_random_expert_bc.analyze import estimate,wilson


def main():
    primary=json.loads((RESULT/'bc_primary_summary.json').read_text())
    diff=primary['paired']['success']
    if diff['ci95'][0]<=0 or diff['positive_seeds']<4:
        state('anchor_selection_skipped',status='complete',reason='No clear paired GT benefit');return
    protocol=dict(primary_gate_unchanged=primary['anchor_gate_passed'],
        rationale='All five paired GT improvements positive and paired t CI above zero, but mean-success engineering gate failed; user permits new BC+GT anchor if clearly better. Separate operational selection addresses the chosen deployed model, not five-seed training stability.',
        selection=dict(candidates=list(range(5)),episodes=64,common_reset_seed=14101,criterion='highest success; then lower offline validation MSE; then seed id'),
        independent_retest=dict(cohort_seeds=[14201,14202,14203],deterministic_episodes_each=128,stochastic_episodes_each=64),
        gate='Each deterministic cohort >=60%, pooled deterministic Wilson lower bound >60%, stochastic mean >=60% and every stochastic cohort >=50%. No re-selection if this retest fails.',
        clarification='Retest cohorts are evaluation replicates for one selected model, not five independent training seeds. Does not establish superiority over historical Phase12 recipe.')
    with (RESULT/'anchor_selection_protocol.json').open('x') as f:json.dump(protocol,f,indent=2)
    def evaluate(name,checkpoint,output,episodes,seed,stochastic=False):
        arguments=['--checkpoint',checkpoint,'--output',output,'--episodes',episodes,'--num-envs',32,'--seed',seed,'--device','cuda:0']
        if stochastic:arguments.append('--stochastic')
        return run(name,'experiments.phase13_random_expert_bc.evaluate',arguments,1800)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(evaluate,f'anchor_select_seed{seed}',CHECK/f'B_seed{seed}_best.pt',
                         RESULT/f'anchor_selection/seed{seed}.json',64,14101) for seed in range(5)]
        for job in jobs:job.result()
    candidates=[]
    for seed in range(5):
        v=json.loads((RESULT/f'anchor_selection/seed{seed}.json').read_text())
        c=torch.load(CHECK/f'B_seed{seed}_best.pt',map_location='cpu',weights_only=False)
        candidates.append(dict(seed=seed,success=v['success'],validation_mse=c['validation_mse']))
    chosen=sorted(candidates,key=lambda v:(-v['success'],v['validation_mse'],v['seed']))[0]
    source=CHECK/f"B_seed{chosen['seed']}_best.pt"
    (RESULT/'anchor_selection_choice.json').write_text(json.dumps(dict(chosen=chosen,candidates=candidates,sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[]
        for index,seed in enumerate(protocol['independent_retest']['cohort_seeds']):
            for stochastic,episodes in [(False,128),(True,64)]:
                mode='stochastic' if stochastic else 'deterministic'
                jobs.append(pool.submit(evaluate,f'anchor_retest_{mode}_{index}',source,
                    RESULT/f'anchor_retest/{mode}_{index}.json',episodes,seed,stochastic))
        for job in jobs:job.result()
    det=[json.loads((RESULT/f'anchor_retest/deterministic_{i}.json').read_text()) for i in range(3)]
    stochastic=[json.loads((RESULT/f'anchor_retest/stochastic_{i}.json').read_text()) for i in range(3)]
    n=sum(r['episodes'] for r in det);k=sum(round(r['success']*r['episodes']) for r in det)
    interval=wilson(k,n)
    passed=all(r['success']>=.6 for r in det) and interval[0]>.6 and sum(r['success'] for r in stochastic)/3>=.6 and all(r['success']>=.5 for r in stochastic)
    result=dict(chosen=chosen,candidates=candidates,deterministic=det,stochastic=stochastic,
                deterministic_summary=estimate([r['success'] for r in det]),
                stochastic_summary=estimate([r['success'] for r in stochastic]),pooled_deterministic_wilson=interval,
                operational_anchor_gate_passed=passed,primary_five_seed_gate_passed=primary['anchor_gate_passed'],
                source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    if passed:
        destination=CHECK/'randomized_gt_anchor.pt'
        with destination.open('xb') as f:f.write(source.read_bytes())
        result['anchor']=str(destination.relative_to(ROOT))
    (RESULT/'anchor_validation.json').write_text(json.dumps(result,indent=2))
    state('operational_anchor_validation',status='passed' if passed else 'failed',chosen_seed=chosen['seed'],mean_success=result['deterministic_summary']['mean'],next='small controlled RL permitted' if passed else 'no RL; no further candidate search')


if __name__=='__main__':
    try:main()
    except Exception as error:state('anchor_validation_failed',status='failed',error=repr(error));raise
