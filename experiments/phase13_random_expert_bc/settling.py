"""Measure what happens to randomized door angle before expert approach."""
import json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    rows=[]
    with h5py.File(ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5','r') as h:
        for name in h:
            g=h[name];phase=g['planner_phase'][:,0];angle=g['state'][:,0]
            rest=np.flatnonzero(phase==0)
            rows.append(dict(trajectory=name,initial_deg=float(np.rad2deg(angle[0])),
                             rest_end_deg=float(np.rad2deg(angle[rest[-1]])),rest_steps=len(rest)))
    delta=np.array([r['rest_end_deg']-r['initial_deg'] for r in rows])
    end=np.array([r['rest_end_deg'] for r in rows])
    result=dict(trajectories=len(rows),mean_initial_deg=float(np.mean([r['initial_deg'] for r in rows])),
                mean_rest_end_deg=float(end.mean()),mean_abs_drift_deg=float(np.abs(delta).mean()),
                fraction_drift_over_1deg=float(np.mean(np.abs(delta)>1)),
                fraction_rest_end_under_half_degree=float(np.mean(abs(end)<.5)),
                rows=rows,source_fact=dict(inherited_cabinet_doors_actuator='ImplicitActuatorCfg',
                    stiffness=10.,damping=2.5,effort_limit_sim=87.,
                    path='third_party/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/cabinet_env_cfg.py'),
                interpretation='The initial physical reset angle is randomized correctly, but usually relaxes near zero during 16 REST ticks. Inherited restoring actuator is a plausible cause; contact may contribute. Exact causal split requires intervention and is not established by this audit. Task remains unchanged.')
    path=ROOT/'results/phase13_random_expert_bc/initial_angle_settling.json'
    path.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
