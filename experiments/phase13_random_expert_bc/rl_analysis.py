"""Independent final/best tests and paired three-seed feasibility inference."""
import json
from pathlib import Path
import numpy as np
from experiments.phase13_random_expert_bc.analyze import estimate,paired

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


def main():
    rows=[]
    for seed in range(3):
        for arm in ['A','B','C']:
            train=json.loads((R/f'rl/{arm}_seed{seed}/summary.json').read_text())
            final=json.loads((R/f'rl_heldout/{arm}_seed{seed}_final_deterministic.json').read_text())
            best=(final if arm=='A' else json.loads((R/f'rl_heldout/{arm}_seed{seed}_best_deterministic.json').read_text()))
            curve=train['curves'];x=np.array([c['environment_steps'] for c in curve]);y=np.array([c['success'] for c in curve])
            row=dict(arm=arm,seed=seed,interactions=train['interactions'],updates=train['updates'],
                     validation_auc=float(np.trapz(y,x)/x[-1]),validation_best=train['best_validation'],
                     validation_final=train['final_validation'],best_step=train['best_step'],
                     best_success=best['success'],final_success=final['success'],
                     best_final_gap=best['success']-final['success'],contact_success=final['contact_success'],
                     final_angle=final['mean_final_angle_rad'],online_successes=train['online_successes'],
                     online_episodes=train['online_episodes'],online_success_fraction=train['online_successes']/max(1,train['online_episodes']),
                     heldout_interactions=final['interactions'],rollbacks=0,wall_seconds=train['elapsed_s'])
            if arm=='C':
                stochastic=json.loads((R/f'rl_heldout/C_seed{seed}_final_stochastic.json').read_text())
                row['final_stochastic_success']=stochastic['success']
            rows.append(row)
    metrics=['validation_auc','best_success','final_success','best_final_gap','contact_success','final_angle',
             'online_successes','online_success_fraction']
    grouped={arm:[r for r in rows if r['arm']==arm] for arm in ['A','B','C']}
    summaries={arm:{metric:estimate([r[metric] for r in grouped[arm]]) for metric in metrics} for arm in grouped}
    differences={f'{b}_minus_{a}':{metric:paired([r[metric] for r in grouped[b]],[r[metric] for r in grouped[a]])
                                  for metric in ['final_success','best_final_gap','validation_auc']}
                 for a,b in [('A','B'),('A','C'),('B','C')]}
    summaries['C']['final_stochastic_success']=estimate([r['final_stochastic_success'] for r in grouped['C']])
    result=dict(per_seed=rows,summary=summaries,paired=differences,total_train_interactions=sum(r['interactions'] for r in rows),
       source='One independently selected BC model; three RL training/collector seeds. Not three independent BC initializations.',
       caveats=['n=3 exact two-sided sign-flip minimum p=.25; paired t assumptions and wide CI must be stated.',
                'Best is selected on validation only and tested independently; heldout best-final can be negative from sampling noise.',
                'Frozen arm has identical weights throughout; AUC interpolates initial and final fixed-cohort evaluations, no invented intermediate measurements.',
                'Online success proportions exclude unfinished episodes and refer to changing sampled policies, not stationary heldout final success.',
                'Original five-BC-seed and operational engineering gates remain failed; small RL entry follows user relative-benefit prerequisite.'])
    (R/'rl_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=summaries,paired=differences),indent=2))


if __name__=='__main__':main()
