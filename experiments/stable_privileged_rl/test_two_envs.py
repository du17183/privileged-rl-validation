"""Temporary feasibility check for independent train/eval Isaac environments."""
from runtime_paths import project_path

import argparse
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(headless=True).app

import torch
import traceback
from pathlib import Path
import isaaclab_tasks  # noqa
from door_env.door import make_cfg,robot_observation
from door_env.isaac_env import create_env

one=two=None
marker=Path(project_path('/home/xiaolong/privileged_rl_validation/results/stable_privileged_rl/two_envs_marker.txt'))
try:
    marker.write_text('before_first\n')
    one=create_env(make_cfg(2,device=args.device or 'cuda:0',seed=100))
    marker.write_text(marker.read_text()+'after_first\n')
    before=robot_observation(one).clone()
    marker.write_text(marker.read_text()+'before_second\n')
    two=create_env(make_cfg(2,device=args.device or 'cuda:0',seed=200))
    marker.write_text(marker.read_text()+'after_second\n')
    print('same_sim',one.sim is two.sim,flush=True)
    two.reset(seed=200)
    two.step(torch.zeros(2,7,device=two.device))
    after=robot_observation(one)
    print('train_obs_changed_by_eval_step',not torch.equal(before,after),flush=True)
except BaseException:
    marker.write_text(marker.read_text()+traceback.format_exc())
    raise
finally:
    if two: two.close()
    if one: one.close()
    app.close()
