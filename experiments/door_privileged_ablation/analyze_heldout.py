"""Aggregate independent 64-episode best/final checkpoint replays."""

import csv
from pathlib import Path
import numpy as np

from experiments.door_privileged_ablation.analyze import OUT, NEW, bootstrap, signflip, write_csv


def main():
    rows=[]
    for variant in NEW:
        for seed in range(5):
            for mode in ('best','final'):
                path=OUT/f'heldout_{variant}_seed{seed}_{mode}.csv'
                with path.open(newline='',encoding='utf-8') as stream:
                    row=next(csv.DictReader(stream))
                if int(row['episodes'])!=64: raise ValueError(path)
                rows.append(row)
    write_csv(OUT/'heldout_all.csv',rows)
    aggregates=[]
    for variant in NEW:
        for mode in ('best','final'):
            subset=[r for r in rows if r['variant']==variant and r['mode']==mode]
            success=np.array([float(r['heldout_success_rate']) for r in subset])
            ci=bootstrap(success)
            aggregates.append({'variant':variant,'mode':mode,'seeds':5,
                               'heldout_success_mean':float(success.mean()),
                               'ci95_low':ci[0],'ci95_high':ci[1],
                               'reward_mean':float(np.mean([float(r['heldout_mean_return']) for r in subset])),
                               'final_angle_mean_rad':float(np.mean([float(r['heldout_final_angle_rad']) for r in subset])),
                               'contact_rate_mean':float(np.mean([float(r['heldout_contact_rate']) for r in subset]))})
    write_csv(OUT/'heldout_aggregate.csv',aggregates)
    comparisons=[]
    for baseline,contender in (('diag_A','diag_B'),('diag_A','B1'),('diag_A','B2'),
                               ('diag_A','B5'),('diag_A','B6'),('E0','E1'),
                               ('E0','E2'),('E1','E2'),('E0','E3'),('E1','E3'),('E2','E3')):
        for mode in ('best','final'):
            diff=np.array([float(next(r for r in rows if r['variant']==contender and r['seed']==str(seed) and r['mode']==mode)['heldout_success_rate'])-
                           float(next(r for r in rows if r['variant']==baseline and r['seed']==str(seed) and r['mode']==mode)['heldout_success_rate'])
                           for seed in range(5)])
            ci=bootstrap(diff)
            comparisons.append({'comparison':f'{contender}_minus_{baseline}',
                                'mode':mode,'mean_difference':float(diff.mean()),
                                'ci95_low':ci[0],'ci95_high':ci[1],
                                'exact_signflip_p':signflip(diff)})
    write_csv(OUT/'heldout_paired_effects.csv',comparisons)
    print(f'Aggregated {len(rows)} independent heldout replays')


if __name__=='__main__':
    main()
