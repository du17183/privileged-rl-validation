"""Verify isolated material/fixture overrides before independent test rollout."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(headless=True).app
import json
import math
from pathlib import Path
import torch
import isaaclab_tasks
from progress_rl.door_env import config,create
from door_env.door import door_index,door_angle
root=Path(__file__).resolve().parents[2]
try:
    env=create(config('B',2,args.device or 'cuda:0',214777))
    cabinet=env.scene['cabinet'];material=cabinet.root_physx_view.get_material_properties().clone()
    expected=material.clone();expected[:,:,:2] *= .9
    cabinet.root_physx_view.set_material_properties(expected,torch.arange(2,device='cpu'))
    cabinet.data.default_joint_pos[:,door_index(env)] = math.radians(5)
    cabinet.data.default_root_state[:,1] += .01
    env.reset(seed=214777)
    actual=cabinet.root_physx_view.get_material_properties()
    assert torch.allclose(actual,expected,atol=1e-6)
    assert abs(float(door_angle(env).mean())-math.radians(5))<.001
    assert abs(float((cabinet.data.root_pos_w[:,1]-env.scene.env_origins[:,1]).mean())-.01)<.001
    (root/'results/phase10_quality_lwd/physics_smoke.json').write_text(json.dumps(dict(passed=True,
        observed_angle=float(door_angle(env).mean()),static_friction_mean=float(actual[:,:,0].mean()),
        dynamic_friction_mean=float(actual[:,:,1].mean()),observed_fixture_dy=.01,rollout_steps=0),indent=2))
    print('Independent physical override checks passed',flush=True)
    env.close()
finally:app.close()
