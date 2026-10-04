"""Five-seed factorial inference, original reset buckets, recovery outcomes."""
import csv,json,argparse
from pathlib import Path
import numpy as np
from experiments.phase13_random_expert_bc.analyze import estimate,paired,wilson

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_recovery_bc';CHECK=ROOT/'checkpoints/phase14_recovery_bc'


def main(run_id=None):
    global R,CHECK
    if run_id:R=R/run_id;CHECK=CHECK/run_id
    rows=[];buckets=[];offline=[]
    for seed in range(5):
        offline+=json.loads((CHECK/f'bc_seed{seed}_summary.json').read_text())['test']
        for arm in 'ABCDEF':
            random=json.loads((R/f'random/{arm}_seed{seed}.json').read_text())
            fixed=json.loads((R/f'fixed/{arm}_seed{seed}.json').read_text())
            row=dict(arm=arm,seed=seed,random_success=random['success'],fixed_success=fixed['success'],
                random_max_angle=random['mean_max_angle_rad'],random_final_angle=random['mean_final_angle_rad'],
                random_contact=random['contact_success'])
            values=[]
            for case in ['contact_loss','ee_offset','door_regression']:
                value=json.loads((R/f'recovery/{case}_{arm}_seed{seed}.json').read_text())
                row[case+'_success']=value['success'];row[case+'_valid_fraction']=value['perturbation_valid_fraction'];values.append(value['success'])
            row['recovery_success']=float(np.mean(values));rows.append(row)
            episodes=list(csv.DictReader((R/f'random/{arm}_seed{seed}.csv').open()))
            for kind,bounds in [('initial_angle_deg',[(0,1),(1,2),(2,3),(3,4),(4,5.001)]),('handle_linf_mm',[(0,5),(5,7.5),(7.5,10.001)])]:
                for lo,hi in bounds:
                    selected=[e for e in episodes if lo<=(np.rad2deg(float(e['angle0_rad'])) if kind=='initial_angle_deg' else max(abs(float(e[k])) for k in ['offset_x','offset_y','offset_z'])*1000)<hi]
                    if selected:buckets.append(dict(arm=arm,seed=seed,kind=kind,lower=lo,upper=hi,episodes=len(selected),successes=sum(int(e['success']) for e in selected),success=float(np.mean([int(e['success']) for e in selected]))))
    metrics=['random_success','fixed_success','recovery_success','contact_loss_success','ee_offset_success','door_regression_success','random_contact']
    summaries={arm:{metric:estimate([r[metric] for r in rows if r['arm']==arm]) for metric in metrics} for arm in 'ABCDEF'}
    differences={}
    for first,second in [('A','B'),('A','C'),('B','D'),('C','D'),('A','E'),('B','F'),('E','C'),('F','D')]:
        differences[f'{second}_minus_{first}']={metric:paired([r[metric] for r in rows if r['arm']==second],[r[metric] for r in rows if r['arm']==first]) for metric in metrics}
    interactions={}
    for metric in metrics:
        lookup={(r['arm'],r['seed']):r[metric] for r in rows}
        gain=[(lookup['D',s]-lookup['C',s])-(lookup['B',s]-lookup['A',s]) for s in range(5)]
        interactions[metric]=paired(gain,[0.]*5)
    with (R/'metrics_by_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with (R/'randomization_buckets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=buckets[0].keys());w.writeheader();w.writerows(buckets)
    result=dict(per_seed=rows,summary=summaries,paired=differences,gt_x_recovery_interaction=interactions,offline=offline,buckets=buckets,
                gate=dict(D_random_mean_ge80=summaries['D']['random_success']['mean']>=.80,
                          D_recovery_mean_ge80=summaries['D']['recovery_success']['mean']>=.80,
                          D_minus_B_random_ci_positive=differences['D_minus_B']['random_success']['ci95'][0]>0),
                caveats=['n5 exact two-sided sign-flip minimum p0.0625','Initial angle quickly settles; buckets describe sampled reset angle, not persistent-angle generalization',
                         'New heldout seeds; Phase13 percentages are historical unmatched references','Controlled perturbation invalid attempts are retained; test success conditioned on physically valid failures'])
    (R/'bc_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summary=summaries,gate=result['gate']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id');main(p.parse_args().run_id)
