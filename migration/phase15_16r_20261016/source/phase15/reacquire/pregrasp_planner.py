"""Joint-space IK/RRT with Panda collision geometry and live cabinet bounds.

Cabinet collision shapes are conservatively replaced by individual oriented
bounding boxes in their rigid-body frame. Joint edges are checked at <=.03rad.
This is discrete collision checking, not a continuous or tracking guarantee.
Physics/action/reset are unchanged; no state or joint targets are written.
"""
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import h5py
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from pxr import Usd, UsdGeom, UsdPhysics
import omni.usd
from phase15.common import ROOT


def rotation(q):
    q = np.asarray(q, float)
    return pin.Quaternion(q[0], q[1], q[2], q[3]).normalized().matrix()


class PregraspPlanner:
    def __init__(self, env):
        self.env = env
        urdf = Path(sys.prefix) / 'lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame = self.model.getFrameId('panda_hand')
        self.offset = pin.SE3(np.eye(3), np.array([0., 0., .107]))
        self.indices = [self.model.joints[self.model.getJointId(name)].idx_q for name in env.scene['robot'].joint_names]
        self.geom = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.COLLISION,
                                        [str(urdf.parent.parent.parent), str(urdf.parent.parent)])
        # Adjacent rigid bodies and intentional finger/hand pairs are excluded.
        # Read topology from URDF: this also avoids the unregistered unsigned
        # std::vector binding of Model.parents in the bundled Pin2.7 runtime.
        joints = ET.parse(urdf).getroot().findall('joint')
        child_joint = {j.find('child').attrib['link']: j for j in joints}
        def owner(link):
            j = child_joint.get(link)
            if j is None:
                return 0
            if j.attrib['type'] != 'fixed':
                return self.model.getJointId(j.attrib['name'])
            return owner(j.find('parent').attrib['link'])
        parent_ids = {self.model.getJointId(j.attrib['name']): owner(j.find('parent').attrib['link'])
                      for j in joints if j.attrib['type'] != 'fixed'}
        def ancestors(j):
            result = []
            while j:
                result.append(j)
                j = parent_ids[int(j)]
            result.append(0)
            return result
        for i, a in enumerate(self.geom.geometryObjects):
            for j, b in enumerate(self.geom.geometryObjects[:i]):
                aa, bb = ancestors(a.parentJoint), ancestors(b.parentJoint)
                distance = min(aa.index(k) + bb.index(k) for k in set(aa) & set(bb))
                if distance > 2:
                    self.geom.addCollisionPair(pin.CollisionPair(j, i))
        self.gdata = pin.GeometryData(self.geom)
        self.request = fcl.CollisionRequest()
        self.bounds = env.scene['robot'].data.soft_joint_pos_limits[0].cpu().numpy()
        self.lower, self.upper = self.bounds[:, 0] + .006, self.bounds[:, 1] - .006
        self.lower[7:] = 0.
        self.upper[7:] = .04
        base = ROOT / 'datasets/random_door_expert'
        keys = json.loads((base / 'split_v1/split.json').read_text())['splits']['train']
        poses, quats = [], []
        with h5py.File(base / 'collection_v1/trajectories.h5', 'r') as h:
            for key in keys:
                phase = h[key]['planner_phase'][:].flatten()
                clear = np.flatnonzero(phase == 1)
                rest = np.flatnonzero(phase == 0)
                if len(clear):
                    poses.append(h[key]['observation'][clear[-1], :9])
                if len(rest):
                    quats.append(h[key]['observation'][rest[-1], 21:25])
        poses = np.asarray(poses)
        self.clear = poses[np.argmin(((poses[:, :7] - np.median(poses[:, :7], axis=0)) ** 2).sum(-1))]
        quats = np.asarray(quats)
        quats *= np.where((quats * quats[:1]).sum(-1, keepdims=True) < 0, -1, 1)
        self.grasp_quat = quats.mean(0)
        self.grasp_quat /= np.linalg.norm(self.grasp_quat)
        self.shapes = self._cabinet_shapes()
        if not self.shapes:
            raise RuntimeError('No cabinet collision shapes: cannot plan safely')
        # Actual FK must agree before this planner can issue any action.
        term = env.action_manager.get_term('arm_action')
        actualp, actualq = term._compute_frame_pose()
        q = env.scene['robot'].data.joint_pos.cpu().numpy()
        errors = []
        for i, row in enumerate(q):
            pose = self.fk(row)
            errors.append((np.linalg.norm(pose.translation - actualp[i].cpu().numpy()),
                           np.linalg.norm(pin.log3(pose.rotation @ rotation(actualq[i].cpu().numpy()).T))))
        self.fk_error = np.max(errors, axis=0).tolist()
        if self.fk_error[0] > .003 or self.fk_error[1] > .03:
            raise RuntimeError(f'Panda collision/IK model FK mismatch: {self.fk_error}')
        print(json.dumps(dict(planner='joint_ik_rrt_collision_checked', fk_error=self.fk_error,
            robot_collision_shapes=len(self.geom.geometryObjects), cabinet_shapes=len(self.shapes),
            self_collision_pairs=len(self.geom.collisionPairs))), flush=True)

    def _cabinet_shapes(self):
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath('/World/envs/env_0/Cabinet')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
        bodies = self.env.scene['cabinet'].body_names
        result = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(prim)
            if api.GetCollisionEnabledAttr().Get() is False:
                continue
            if not prim.IsA(UsdGeom.Boundable):
                continue
            ancestor = prim
            while ancestor.IsValid() and ancestor.GetName() not in bodies:
                ancestor = ancestor.GetParent()
            if not ancestor.IsValid():
                raise RuntimeError('Unmapped cabinet collision body ' + str(prim.GetPath()))
            bound = cache.ComputeRelativeBound(prim, ancestor).ComputeAlignedRange()
            low, high = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
            size = high - low
            if not np.isfinite(size).all() or np.any(size < 1e-5):
                continue
            result.append(dict(body=bodies.index(ancestor.GetName()), body_name=ancestor.GetName(),
                name=str(prim.GetPath()), center=(low + high) / 2, size=size, geometry=fcl.Box(*size)))
        return result

    def pack(self, row):
        q = np.zeros(self.model.nq)
        q[self.indices] = row
        return q

    def fk(self, row):
        pin.forwardKinematics(self.model, self.data, self.pack(row))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame] * self.offset

    def world(self, i):
        cab = self.env.scene['cabinet'].data
        root = self.env.scene['robot'].data
        rr = rotation(root.root_quat_w[i].cpu().numpy())
        rp = root.root_pos_w[i].cpu().numpy()
        objects = []
        for shape in self.shapes:
            j = shape['body']
            r = rr.T @ rotation(cab.body_quat_w[i, j].cpu().numpy())
            p = rr.T @ (cab.body_pos_w[i, j].cpu().numpy() - rp) + r @ shape['center']
            objects.append((shape, fcl.Transform3f(r, p)))
        floor = dict(body_name='ground', name='/World/GroundPlane', geometry=fcl.Box(100., 100., .10))
        objects.append((floor, fcl.Transform3f(rr.T, rr.T @ (np.array([rp[0], rp[1], -.05]) - rp))))
        return objects

    def collision(self, row, world, allow_handle=False):
        if np.any(row[:7] < self.lower[:7] - 1e-5) or np.any(row[:7] > self.upper[:7] + 1e-5):
            return True
        q = self.pack(row)
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, q)
        if pin.computeCollisions(self.geom, self.gdata, True):
            return True
        for j, g in enumerate(self.geom.geometryObjects):
            pose = self.gdata.oMg[j]
            t = fcl.Transform3f(pose.rotation, pose.translation)
            for shape, other in world:
                if shape['body_name'] == 'ground' and g.parentJoint == 0:
                    continue
                # Only intended final gripper-handle contact is exempted.
                if allow_handle and shape['body_name'] == 'door_right_nob_link' and any(x in g.name for x in ['hand', 'finger']):
                    continue
                if fcl.collide(g.geometry, t, shape['geometry'], other, self.request, fcl.CollisionResult()):
                    return True
        return False

    def collision_details(self, row, world):
        """Audit the unchanged rejection rules; this does not grant escape."""
        violations = np.maximum(self.lower[:7] - row[:7], 0) + np.maximum(row[:7] - self.upper[:7], 0)
        details = dict(joint_margin_violations_rad=violations.tolist(),
            joint_margin_violation=bool(np.any(violations > 1e-5)), self_collisions=[], scene_collisions=[])
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, self.pack(row))
        pin.computeCollisions(self.geom, self.gdata, False)
        for pair, result in zip(self.geom.collisionPairs, self.gdata.collisionResults):
            if result.isCollision():
                details['self_collisions'].append([self.geom.geometryObjects[pair.first].name,
                                                  self.geom.geometryObjects[pair.second].name])
        for j, g in enumerate(self.geom.geometryObjects):
            pose = self.gdata.oMg[j]
            t = fcl.Transform3f(pose.rotation, pose.translation)
            for shape, other in world:
                if shape['body_name'] == 'ground' and g.parentJoint == 0:
                    continue
                if fcl.collide(g.geometry, t, shape['geometry'], other, self.request, fcl.CollisionResult()):
                    details['scene_collisions'].append([g.name, shape['name']])
        return details

    def edge(self, a, b, world, allow_handle=False):
        n = max(2, int(np.ceil(np.max(np.abs(b[:7] - a[:7])) / .03)) + 1)
        return all(not self.collision(a * (1 - t) + b * t, world, allow_handle)
                   for t in np.linspace(0, 1, n))

    def ik(self, goal, initial):
        row = initial.copy()
        for _ in range(180):
            pose = self.fk(row)
            error = np.r_[goal.translation - pose.translation, pin.log3(goal.rotation @ pose.rotation.T)]
            if np.linalg.norm(error[:3]) < .004 and np.linalg.norm(error[3:]) < .08:
                return row
            j = pin.computeFrameJacobian(self.model, self.data, self.pack(row), self.frame, pin.LOCAL_WORLD_ALIGNED)
            j[:3] -= pin.skew(pose.rotation @ self.offset.translation) @ j[3:]
            dq = j[:6, :7].T @ np.linalg.solve(j[:6, :7] @ j[:6, :7].T + .015 ** 2 * np.eye(6), error)
            row[:7] = np.clip(row[:7] + np.clip(.6 * dq, -.10, .10), self.lower[:7], self.upper[:7])
        return None

    def plan(self, i, goal, seed):
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
        seeds = [start.copy(), self.clear.copy()]
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
        goals.sort(key=lambda q: np.linalg.norm(q[:7] - start[:7]))
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
