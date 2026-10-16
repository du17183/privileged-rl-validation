"""Small, train-only, angle-conditioned pose set; same collision model."""
import json
import os
import numpy as np
import pinocchio as pin
import h5py
from phase15_2.common import ROOT,RESULTS,save_json,sha256,redirect_calibration
redirect_calibration()
from phase15_1.severe_reacquire.pregrasp_planner import PregraspPlanner as Original,rotation


class NormalProfile:
    def __init__(self):
        source=ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5'
        split=ROOT/'datasets/random_door_expert/split_v1/split.json'
        keys=json.loads(split.read_text())['splits']['train'];rows=[]
        with h5py.File(source) as h:
            for name in keys:
                g=h[name];r=g['robot_state'][:];s=g['state'][:]
                valid=np.flatnonzero((s[:,9:11]>.5).all(-1)&(s[:,0]<1.05))
                for k in valid[::8]:
                    hr=rotation(s[k,5:9]);er=rotation(r[k,21:25])
                    rows.append(dict(angle=float(s[k,0]),q=r[k,:9].copy(),pos=hr.T@(r[k,18:21]-s[k,2:5]),rot=hr.T@er))
        if len(rows)<100:raise RuntimeError('Insufficient frozen Normal train support')
        self.rows=rows
        self.angle=np.array([x['angle'] for x in rows]);self.q=np.array([x['q'] for x in rows])
        self.pos=np.array([x['pos'] for x in rows]);self.rot=np.array([x['rot'] for x in rows])
        save_json(RESULTS/f'calibration/normal_entry_profile_{os.getpid()}.json',dict(source=str(source),source_sha256=sha256(source),split_sha256=sha256(split),used_split='train',samples=len(rows),position_limit_m=.035,orientation_limit_rad=.30,joint_nearest_linf_rad=.65,door_angle_support_tolerance_rad=.12,training_performed=False))

    def reference(self,angle,q):
        ids=np.flatnonzero(np.abs(self.angle-angle)<.10)
        if not len(ids):ids=np.argsort(np.abs(self.angle-angle))[:32]
        return int(ids[np.argmin(((self.q[ids,:7]-q[:7])**2).sum(-1))])

    def entry(self,raw,planner,i):
        hr=rotation(raw[35:39]);er=rotation(raw[21:25]);q=raw[:9]
        pos=hr.T@(raw[18:21]-raw[32:35]);relative=hr.T@er
        ids=np.flatnonzero(np.abs(self.angle-raw[26])<.12)
        if not len(ids):return False,dict(reason='door_angle_outside_train_support')
        poserr=np.linalg.norm(self.pos[ids]-pos,axis=1)
        roterr=np.array([np.linalg.norm(pin.log3(r@relative.T)) for r in self.rot[ids]])
        qerr=np.max(np.abs(self.q[ids,:7]-q[:7]),axis=1)
        score=poserr/.035+roterr/.30+qerr/.65;j=int(np.argmin(score))
        margin=float(np.min(np.minimum(q[:7]-planner.lower[:7],planner.upper[:7]-q[:7])))
        width=float(q[7:9].sum());contact=bool(np.all(raw[30:32]>.5))
        valid=contact and .005<width<.079 and poserr[j]<.035 and roterr[j]<.30 and qerr[j]<.65 and margin>=0 and 600-round(raw[25]*600)>=45
        return bool(valid),dict(contact=contact,position_error_m=float(poserr[j]),orientation_error_rad=float(roterr[j]),train_joint_linf_rad=float(qerr[j]),joint_safe_margin_rad=margin,train_reference=int(ids[j]),remaining_ticks=int(600-round(raw[25]*600)))


class PregraspPlanner(Original):
    def __init__(self,env,retreat='R2',geometry=False):
        super().__init__(env,search=False)
        self.retreat_mode=retreat;self.dynamic_geometry=geometry
        self.retreat_origins={};self.retreat_cache={};self.profile_data=NormalProfile()

    def goal(self,i,pregrasp=True):
        if not self.dynamic_geometry:return super().goal(i,pregrasp)
        from door_env.door import handle_pose
        hp,hq=handle_pose(self.env);root=self.env.scene['robot'].data
        joint=root.joint_pos[i].cpu().numpy();angle=float(self.env.get_tool_state()[i,0])
        index=self.profile_data.reference(angle,joint)
        rr=rotation(root.root_quat_w[i].cpu().numpy());rp=root.root_pos_w[i].cpu().numpy();hr=rotation(hq[i].cpu().numpy())
        self.grasp_posture=self.profile_data.q[index].copy()
        offset=self.profile_data.pos[index]+(.065*self.approach_direction if pregrasp else 0)
        return pin.SE3(rr.T@hr@self.profile_data.rot[index],rr.T@(hp[i].cpu().numpy()+hr@offset-rp))

    def door_frame(self,i):
        world=self.world(i)
        candidates=[(s,t) for s,t in world if s['body_name']=='door_right_link']
        if not candidates:
            candidates=[(s,t) for s,t in world if 'door_right' in s['body_name'] and 'nob' not in s['body_name']]
        if not candidates:
            # Existing collision OBB export need not contain the door mesh.
            # Use immutable USD visual dimensions and live GT rigid-body pose
            # for the retreat direction only; collision shapes remain unchanged.
            if not hasattr(self,'door_plane_geometry'):
                from pxr import Usd,UsdGeom
                import omni.usd
                cabinet=self.env.scene['cabinet']
                names=[name for name in cabinet.body_names if 'door_right' in name and 'nob' not in name and 'handle' not in name]
                if len(names)!=1:raise RuntimeError('Ambiguous frozen door rigid body '+str(cabinet.body_names))
                name=names[0];prim=omni.usd.get_context().get_stage().GetPrimAtPath('/World/envs/env_0/Cabinet/'+name)
                if not prim.IsValid():raise RuntimeError('Missing frozen door USD prim')
                cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
                bounds=cache.ComputeRelativeBound(prim,prim).ComputeAlignedRange()
                low,high=np.asarray(bounds.GetMin()),np.asarray(bounds.GetMax());size=high-low
                if not np.isfinite(size).all() or np.min(size)<=0:raise RuntimeError('Invalid frozen door visual dimensions')
                self.door_plane_geometry=(cabinet.body_names.index(name),(low+high)/2,int(np.argmin(size)))
                save_json(RESULTS/f'calibration/door_plane_{os.getpid()}.json',dict(body=name,size=size.tolist(),thin_axis=int(np.argmin(size)),
                    usd_source=str(ROOT/'assets/panda_door_cabinet.usd'),usd_sha256=sha256(ROOT/'assets/panda_door_cabinet.usd'),
                    collision_shape_export_contains_door=False,collision_shapes_unchanged=True,used_for='GT direction only'))
            j,offset,axis=self.door_plane_geometry
            cab=self.env.scene['cabinet'].data;robot=self.env.scene['robot'].data
            rr=rotation(robot.root_quat_w[i].cpu().numpy());rp=robot.root_pos_w[i].cpu().numpy()
            orientation=rr.T@rotation(cab.body_quat_w[i,j].cpu().numpy())
            center=rr.T@(cab.body_pos_w[i,j].cpu().numpy()-rp)+orientation@offset
            normal=orientation[:,axis]
            if np.dot(-center,normal)<0:normal=-normal
            return center,normal
        s,t=max(candidates,key=lambda pair:float(np.prod(pair[0]['size'])))
        normal=np.asarray(t.getRotation())[:,int(np.argmin(s['size']))]
        center=np.asarray(t.getTranslation())
        if np.dot(-center,normal)<0:normal=-normal
        return center,normal

    def plan(self,i,goal,seed):
        path,meta=super().plan(i,goal,seed)
        if path is not None or not self.dynamic_geometry:return path,meta
        # Exactly two small geometry candidates, no sweep and no safety relaxation.
        for offset in [np.array([0.,0.,.010]),np.array([0.,.010,0.])]:
            pose=pin.SE3(goal.rotation,goal.translation+offset)
            candidate,extra=super().plan(i,pose,seed)
            meta['plan_s']+=extra['plan_s'];meta['ik_success']|=extra['ik_success'];meta['collision_rejections']+=extra['collision_rejections']
            if candidate is not None:
                extra['geometry_offset_m']=offset.tolist();extra['plan_s']=meta['plan_s'];return candidate,extra
        return None,meta
