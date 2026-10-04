"""Read-only task diagnostic: standard zero-arm/open-gripper control."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);a=p.parse_args()
app=AppLauncher(headless=True).app
import json
from pathlib import Path
import torch
import isaaclab_tasks  # noqa: F401
from randomized_env.door_randomization import create
from door_env.door import door_index,door_angle,door_velocity


def main():
    torch.set_num_threads(4)
    env=create(4,a.device or 'cuda:0',13991,level=2,preset=dict(angle_deg=5.,offset_xyz=[0.,0.,0.],friction_scale=1.))
    rows=[]
    try:
        cabinet=env.scene['cabinet'];index=door_index(env)
        action=torch.zeros((4,7),device=env.device);action[:,-1]=1
        for step in range(17):
            truth=env.get_tool_state()
            rows.append(dict(step=step,angle_rad=door_angle(env).flatten().cpu().tolist(),
                             velocity=door_velocity(env).flatten().cpu().tolist(),
                             commanded_door_position=cabinet.data.joint_pos_target[:,index].cpu().tolist(),
                             contact=truth[:,9:11].cpu().tolist()))
            if step<16:env.step(action)
        result=dict(seed=13991,num_envs=4,interactions=64,reset_angle_deg=5.,offset_xyz=[0.,0.,0.],
                    control='arm Cartesian increments zero; gripper open; no cabinet actions or configuration changes',rows=rows,
                    caution='Observational check of current target and passive evolution, not an isolated causal intervention')
        path=Path('results/phase13_random_expert_bc/settling_probe.json')
        with path.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    try:main()
    finally:app.close()
