"""Read-only support of the nominal expert data in measured feature space."""
import json
import h5py
import numpy as np
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT
initial=[];all_states=[];steps=[]
with h5py.File(ROOT/'door_dataset/door_expert_1000.h5','r') as h5:
    for name in sorted(h5):
        if not name.startswith('traj_'):continue
        state=h5[name]['state'][:]
        initial.append(state[0]);all_states.append(state);steps.append(len(state))
initial=np.stack(initial);all_states=np.concatenate(all_states)
result=dict(trajectories=len(initial),transitions=len(all_states),
    initial_angle_rad_min_max=[float(initial[:,0].min()),float(initial[:,0].max())],
    trajectory_angle_rad_quantiles=np.quantile(all_states[:,0],[0,.25,.5,.75,1]).tolist(),
    initial_handle_xyz_min=initial[:,2:5].min(0).tolist(),initial_handle_xyz_max=initial[:,2:5].max(0).tolist(),
    all_handle_xyz_min=all_states[:,2:5].min(0).tolist(),all_handle_xyz_max=all_states[:,2:5].max(0).tolist(),
    initial_contact_fraction=initial[:,9:11].mean(0).tolist(),all_contact_fraction=all_states[:,9:11].mean(0).tolist(),
    episode_length_quantiles=np.quantile(steps,[0,.25,.5,.75,1]).tolist(),
    limitation='Marginal angle/pose range along successful opening does not imply support for robot-home observations combined with new initial angle/fixture position; no randomized expert data were added.')
(OUT/'expert_coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
