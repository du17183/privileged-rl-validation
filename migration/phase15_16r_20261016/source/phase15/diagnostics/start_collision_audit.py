"""Audit recorded pilot starts without generating data or learning a policy."""
import argparse
import os
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(headless=True).app

import json
import types
import h5py
import numpy as np
import torch
from randomized_env.door_randomization import create
from door_env.door import door_index
from phase15.common import ROOT, RESULTS, save_json
from phase15.reacquire.pregrasp_planner import PregraspPlanner


def main():
    output = ROOT / a.output
    if not output.resolve().is_relative_to(RESULTS.resolve()) or output.exists():
        raise RuntimeError('Audit requires fresh Phase15 output')
    env = create(32, a.device or 'cuda:0', 143501, level=2)
    cohort = ROOT / 'results/phase14_3_pi05_bc/cohorts/validation/natural_severe.h5'
    with h5py.File(cohort, 'r') as h:
        keys = sorted(h)
        parameters = np.stack([h[k]['reset_parameters'][:] for k in keys])
        robot = np.stack([h[k]['handoff_robot'][:] for k in keys])
        gt = np.stack([h[k]['handoff_state'][:] for k in keys])
    fixed = torch.tensor(parameters, device=env.device)
    env._sample = types.MethodType(lambda self, ids: fixed[ids].clone(), env)
    env.reset(seed=142801)
    ids = torch.arange(32, device=env.device)
    rob, cab = env.scene['robot'], env.scene['cabinet']
    rob.write_joint_state_to_sim(torch.tensor(robot[:, :9], device=env.device),
                                 torch.tensor(robot[:, 9:18], device=env.device), env_ids=ids)
    q, v = cab.data.joint_pos.clone(), cab.data.joint_vel.clone()
    j = door_index(env)
    q[:, j], v[:, j] = torch.tensor(gt[:, 0], device=env.device), torch.tensor(gt[:, 1], device=env.device)
    cab.write_joint_state_to_sim(q, v, env_ids=ids)
    env.sim.forward()
    env.scene['ee_frame'].reset(ids)
    env.scene['ee_frame'].update(0., force_recompute=True)
    planner = PregraspPlanner(env)
    details = []
    for i, row in enumerate(robot[:, :9]):
        detail = planner.collision_details(row, planner.world(i))
        detail.update(cohort_id=keys[i], original_rejected=bool(planner.collision(row, planner.world(i))),
                      q=row.tolist())
        details.append(detail)
    shapes = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items() if k != 'geometry'}
              for s in planner.shapes]
    save_json(output, dict(samples=details, soft_bounds=planner.bounds.tolist(),
        planning_lower=planner.lower.tolist(), planning_upper=planner.upper.tolist(),
        actual_joint_bounds=rob.data.joint_pos_limits[0].cpu().numpy().tolist(),
        shapes=shapes, snapshot_initialization_only=True, actions_executed=0, rl_updates=0))
    print('START AUDIT COMPLETE ' + str(output), flush=True)
    env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        os._exit(1)
    os._exit(0)
