"""Audit stored trajectory values against realized discounted returns."""

import csv
from pathlib import Path

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'stable_privileged_rl'


def rank_corr(a,b):
    if len(a)<3: return float('nan')
    x=np.argsort(np.argsort(a)).astype(float)
    y=np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(x,y)[0,1])


def main():
    rows=[]
    for variant in ('Q0','QGT','QV'):
        for seed in range(5):
            path=OUT/f'trajectories_{variant}_seed{seed}.h5'
            with h5py.File(path,'r') as h5:
                episode_id=h5['episode_id'][:]
                complete=episode_id>=0
                ids,first=np.unique(episode_id[complete],return_index=True)
                positions=np.flatnonzero(complete)[first]
                # h5py fancy indices must be increasing, while episode IDs
                # follow completion order rather than transition order.
                value=h5['trajectory_value'][:][positions]
                realized=h5['return'][:][positions]
                success=h5['success'][:][positions]
                weights=h5['sampling_weight'][:]
                recent=slice(max(0,len(ids)-256),len(ids))
                ess=float(weights.sum()**2/np.square(weights).sum()/len(weights))
                rows.append({'variant':variant,'seed':seed,'complete_episodes':len(ids),
                             'rank_rho_all':rank_corr(value,realized),
                             'rank_rho_last256':rank_corr(value[recent],realized[recent]),
                             'last256_success_rate':float(np.mean(success[recent])),
                             'sampling_ess_fraction':ess,
                             'min_sampling_weight':float(weights.min()),
                             'max_sampling_weight':float(weights.max())})
    out=OUT/'trajectory_value_calibration.csv'
    with out.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'Wrote {len(rows)} value-calibration rows to {out}')


if __name__=='__main__': main()
