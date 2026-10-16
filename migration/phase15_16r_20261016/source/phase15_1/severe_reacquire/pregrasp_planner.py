"""Train-split geometry calibration and checked six-dimensional planning."""
import json
import os
import time
import h5py
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from phase15.reacquire.pregrasp_planner import PregraspPlanner as Original,rotation
from phase15_1.common import ROOT,RESULTS,save_json,sha256


class PregraspPlanner(Original):
    def __init__(self,env,search=True):
        super().__init__(env);self.search=search
        path=ROOT/'datasets/random_door_expert/collection_v1/trajectories.h5'
        split=ROOT/'datasets/random_door_expert/split_v1/split.json'
        keys=json.loads(split.read_text())['splits']['train']
        poses=[];quats=[];directions=[];joints=[]
        with h5py.File(path,'r') as h:
            for key in keys:
                g=h[key];r=g['robot_state'][:];s=g['state'][:];phase=g['planner_phase'][:].ravel()
                valid=np.flatnonzero((s[:,9:11]>.5).all(-1)&(phase>=4))
                if not len(valid):continue
                k=int(valid[0]);hr=rotation(s[k,5:9])
                poses.append(hr.T@(r[k,18:21]-s[k,2:5]))
                q=pin.Quaternion(hr.T@rotation(r[k,21:25])).coeffs();q=np.r_[q[3],q[:3]]
                if quats and np.dot(q,quats[0])<0:q=-q
                quats.append(q);joints.append(r[k,:9])
                before=np.flatnonzero((phase==1)|(phase==2))
                if len(before):
                    j=int(before[-1]);direction=hr.T@(r[j,18:21]-r[k,18:21])
                    if np.linalg.norm(direction)>.015:directions.append(direction/np.linalg.norm(direction))
        if len(poses)<20 or not directions:raise RuntimeError('Insufficient train-only grasp calibration')
        self.grasp_relative=np.median(poses,axis=0)
        q=np.mean(quats,axis=0);self.grasp_rotation=rotation(q/np.linalg.norm(q))
        direction=np.median(directions,axis=0);self.approach_direction=direction/np.linalg.norm(direction)
        joints=np.array(joints);self.grasp_posture=joints[np.argmin(((joints[:,:7]-np.median(joints[:,:7],axis=0))**2).sum(-1))]
        self.profile_request=fcl.DistanceRequest();self.profile_request.enable_nearest_points=False
        save_json(RESULTS/f'calibration/worker_{os.getpid()}.json',dict(source=str(path),sha256=sha256(path),split_sha256=sha256(split),
            used_split='train',count=len(poses),relative_grasp_position=self.grasp_relative.tolist(),
            relative_grasp_rotation=self.grasp_rotation.tolist(),approach_direction=self.approach_direction.tolist(),
            clear_joint=self.clear.tolist(),no_new_training_data=True,trained_policy_frozen=True))

    def goal(self,i,pregrasp=True):
        from door_env.door import handle_pose
        hp,hq=handle_pose(self.env);root=self.env.scene['robot'].data
        rr=rotation(root.root_quat_w[i].cpu().numpy());rp=root.root_pos_w[i].cpu().numpy()
        hr=rotation(hq[i].cpu().numpy())
        offset=self.grasp_relative+(.08*self.approach_direction if pregrasp else 0)
        return pin.SE3(rr.T@hr@self.grasp_rotation,rr.T@(hp[i].cpu().numpy()+hr@offset-rp))

    def profile(self,row,world,allow_handle=False):
        """Signed penetration of the same frozen collision shapes, not physics force."""
        pin.updateGeometryPlacements(self.model,self.data,self.geom,self.gdata,self.pack(row))
        penetrations={}
        for pair in self.geom.collisionPairs:
            a,b=self.geom.geometryObjects[pair.first],self.geom.geometryObjects[pair.second]
            pa,pb=self.gdata.oMg[pair.first],self.gdata.oMg[pair.second]
            d=fcl.distance(a.geometry,fcl.Transform3f(pa.rotation,pa.translation),b.geometry,
                fcl.Transform3f(pb.rotation,pb.translation),self.profile_request,fcl.DistanceResult())
            if d<0:penetrations['self:'+a.name+'|'+b.name]=float(-d)
        for j,g in enumerate(self.geom.geometryObjects):
            pose=self.gdata.oMg[j];transform=fcl.Transform3f(pose.rotation,pose.translation)
            for shape,other in world:
                if shape['body_name']=='ground' and g.parentJoint==0:continue
                if allow_handle and shape['body_name']=='door_right_nob_link' and any(v in g.name for v in ['hand','finger']):continue
                d=fcl.distance(g.geometry,transform,shape['geometry'],other,self.profile_request,fcl.DistanceResult())
                if d<0:penetrations[g.name+'|'+shape['name']]=float(-d)
        violation=np.maximum(self.lower[:7]-row[:7],0)+np.maximum(row[:7]-self.upper[:7],0)
        return penetrations,violation

    def escape_edge(self,a,b,world):
        before,limits=self.profile(a,world)
        for t in np.linspace(0,1,5)[1:]:
            now,margin=self.profile(a*(1-t)+b*t,world)
            if np.any(margin>limits+2e-6):return False
            if any(depth>before.get(pair,0)+1e-5 for pair,depth in now.items()):return False
            before,limits=now,margin
        return True

    def plan(self,i,goal,seed):
        if self.search:return self.search_plan(i,goal,seed)
        start=self.env.scene['robot'].data.joint_pos[i].cpu().numpy().copy();world=self.world(i)
        tick=time.perf_counter();meta=dict(ik_success=False,planning_success=False,start_collision=False,
            collision_rejections=0,rrt_nodes=0,plan_s=0.,failure=None)
        if self.collision(start,world):
            meta.update(start_collision=True,failure='start_collision_or_joint_limit');return None,meta
        for initial in [start.copy(),self.grasp_posture.copy(),self.clear.copy()]:
            initial[7:]=.04;q=self.ik(goal,initial)
            if q is None:continue
            meta['ik_success']=True
            if not self.collision(q,world) and self.edge(start,q,world):
                meta.update(planning_success=True,plan_s=time.perf_counter()-tick,path_waypoints=2)
                return [start,q],meta
            meta['collision_rejections']+=1
        meta.update(failure='checked_straight_path_not_found' if meta['ik_success'] else 'ik_unreachable',plan_s=time.perf_counter()-tick)
        return None,meta

    def search_plan(self, i, goal, seed):
        start = self.env.scene['robot'].data.joint_pos[i].cpu().numpy().copy()
        world = self.world(i)
        rng = np.random.default_rng(seed)
        tick = time.perf_counter()
        meta = dict(ik_success=False, planning_success=False, start_collision=False,
                    collision_rejections=0, rrt_nodes=0, plan_s=0., failure=None)
        if self.collision(start, world):
            meta.update(start_collision=True, failure='start_collision_or_joint_limit',
                        start_details=self.collision_details(start, world), plan_s=time.perf_counter() - tick)
            return None, meta
        goals = []
        seeds = [self.grasp_posture.copy(), self.clear.copy(), start.copy()]
        seeds += [np.r_[np.clip(start[:7] + rng.normal(0, .3, 7), self.lower[:7], self.upper[:7]), [.04, .04]] for _ in range(6)]
        for initial in seeds:
            initial[7:] = .04
            q = self.ik(goal, initial)
            if q is None:
                continue
            meta['ik_success'] = True
            if self.collision(q, world):
                meta['collision_rejections'] += 1
                continue
            goals.append(q)
        if not goals:
            meta.update(failure='goal_collision' if meta['ik_success'] else 'ik_unreachable', plan_s=time.perf_counter() - tick)
            return None, meta
        goals.sort(key=lambda q: np.linalg.norm(q[:7] - self.grasp_posture[:7]) + .2*np.linalg.norm(q[:7] - start[:7]))
        for q in goals:
            if self.edge(start, q, world):
                meta.update(planning_success=True, plan_s=time.perf_counter() - tick, path_waypoints=2)
                return [start, q], meta
        # Fixed seed, joint-space RRT; small bounded planning budget per event.
        nodes, parents = [start], [-1]
        deadline = time.perf_counter() + 2.
        for iteration in range(800):
            if time.perf_counter() > deadline:
                break
            q = goals[iteration % len(goals)] if rng.random() < .25 else np.r_[rng.uniform(self.lower[:7], self.upper[:7]), [.04, .04]]
            index = int(np.argmin([np.linalg.norm(v[:7] - q[:7]) for v in nodes]))
            a = nodes[index]
            b = a + (q - a) * min(1., .25 / max(np.linalg.norm(q[:7] - a[:7]), 1e-8))
            if not self.edge(a, b, world):
                meta['collision_rejections'] += 1
                continue
            nodes.append(b)
            parents.append(index)
            for goalq in goals:
                if np.linalg.norm(b[:7] - goalq[:7]) > .65 or not self.edge(b, goalq, world):
                    continue
                path = [goalq, b]
                k = index
                while k >= 0:
                    path.append(nodes[k])
                    k = parents[k]
                path.reverse()
                meta.update(planning_success=True, plan_s=time.perf_counter() - tick,
                            rrt_nodes=len(nodes), path_waypoints=len(path))
                return path, meta
        meta.update(failure='collision_safe_path_not_found', rrt_nodes=len(nodes), plan_s=time.perf_counter() - tick)
        return None, meta
