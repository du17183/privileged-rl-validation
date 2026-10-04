"""Inspect door knob USD geometry and articulate the right door."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app

import torch
from pxr import Usd, UsdGeom
import isaaclab_tasks  # noqa: F401
from envs.drawer import make_cfg
from envs.isaac_env import create_env

cfg = make_cfg(1, device="cuda:0", seed=123)
env = create_env(cfg)
stage = env.sim.stage
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if path.startswith('/World/envs/env_0/Cabinet') and ('door_right' in path or 'nob' in path):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform):
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
            print('PRIM', path, str(prim.GetTypeName()), tuple(bbox.GetMin()), tuple(bbox.GetMax()), flush=True)
        if 'door_right_nob_link' in path:
            print('KNOB_PRIM',path,prim.GetTypeName(),flush=True)

cab = env.scene['cabinet']
j = cab.joint_names.index('door_right_joint')
for q in [0.0, 0.4, 0.8, 1.2]:
    joint_pos = cab.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[:, j] = q
    cab.write_joint_state_to_sim(joint_pos, joint_vel)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.step_dt)
    print('Q',q,'JOINT',cab.data.joint_pos[0,j].item(),flush=True)
    k = cab.body_names.index('door_right_nob_link')
    print('PHYS_BODY', cab.data.body_pos_w[0,k].tolist(), cab.data.body_quat_w[0,k].tolist(), flush=True)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith('/World/envs/env_0/Cabinet/door_right_nob_link') and (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform)):
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
            print('MOVING',path,tuple(bbox.GetMin()),tuple(bbox.GetMax()),flush=True)
env.close()
app.close()
