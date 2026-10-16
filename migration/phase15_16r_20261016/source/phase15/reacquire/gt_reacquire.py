"""Release -> checked pregrasp path -> align -> grasp -> policy handback."""
import time
import numpy as np
import torch
import pinocchio as pin
from isaaclab.utils.math import compute_pose_error
from door_env.door import handle_pose
from recovery_expert.ik_feasibility import project
from .pregrasp_planner import PregraspPlanner, rotation


class GTReacquire:
    def __init__(self, env):
        self.env = env
        self.planner = PregraspPlanner(env)
        self.ready = np.zeros(32, bool)
        self.phase = np.full(32, 'idle', dtype=object)
        self.path = [None] * 32
        self.cursor = np.zeros(32, int)
        self.elapsed = np.zeros(32, int)
        self.grasp_run = np.zeros(32, int)
        self.history = [[] for _ in range(32)]
        self.current = [None] * 32
        self.grasp_target = [None] * 32

    def reset(self, ids):
        for i in ids:
            self.ready[i] = False
            self.phase[i] = 'idle'
            self.path[i] = None
            self.cursor[i] = 0
            self.elapsed[i] = 0
            self.grasp_run[i] = 0
            self.history[i] = []
            self.current[i] = None
            self.grasp_target[i] = None

    def target(self, i, pregrasp):
        env = self.env
        gt = env.get_tool_state()
        handle, _ = handle_pose(env)
        angle = float(gt[i, 0])
        normal = np.array([np.cos(angle), np.sin(angle), 0.])
        pos = handle[i].cpu().numpy() - (.08 if pregrasp else -.03) * normal
        root = env.scene['robot'].data
        rr = rotation(root.root_quat_w[i].cpu().numpy())
        rp = root.root_pos_w[i].cpu().numpy()
        orientation = pin.rpy.rpyToMatrix(0., 0., angle) @ rotation(self.planner.grasp_quat)
        return pin.SE3(rr.T @ orientation, rr.T @ (pos - rp))

    def start(self, ids):
        for i in ids:
            age = int(self.env.episode_length_buf[i])
            self.ready[i] = False
            self.elapsed[i] = 0
            self.grasp_run[i] = 0
            self.grasp_target[i] = None
            path, meta = self.planner.plan(int(i), self.target(int(i), True), 150900 + int(i) * 1009 + age)
            meta.update(trigger_tick=age, reached_pregrasp=False, regrasp=False,
                handback=False, failure_stage=meta['failure'],
                geometry='Panda URDF collision shapes + live per-shape cabinet conservative OBBs',
                discrete_joint_edge_resolution_rad=.03, responsible_for_opening=False)
            self.history[i].append(meta)
            self.current[i] = meta
            self.path[i] = path
            self.cursor[i] = 1
            self.phase[i] = 'release' if path is not None else 'failed'

    @torch.no_grad()
    def action(self, active):
        tick = time.perf_counter()
        env = self.env
        term = env.action_manager.get_term('arm_action')
        p, q = term._compute_frame_pose()
        joint = env.scene['robot'].data.joint_pos.cpu().numpy()
        gt = env.get_tool_state().cpu().numpy()
        handle, _ = handle_pose(env)
        frame = env.scene['ee_frame'].data
        dist = (frame.target_pos_w[:, 0] - handle).norm(dim=-1).cpu().numpy()
        action = torch.zeros((32, 7), device=env.device)
        action[:, 6] = 1
        for i in np.flatnonzero(active):
            self.elapsed[i] += 1
            meta = self.current[i]
            if meta is None:
                raise RuntimeError('Planner called without takeover')
            phase = self.phase[i]
            if phase in ['failed', 'ready']:
                action[i, 6] = -1 if phase == 'ready' else 1
                continue
            if self.elapsed[i] > 210:
                self.phase[i] = 'failed'
                meta['failure_stage'] = 'time_budget'
                continue
            if phase == 'release':
                if joint[i, 7:9].sum() > .07 or self.elapsed[i] >= 12:
                    self.phase[i] = 'path'
                continue
            if phase == 'path':
                path = self.path[i]
                k = self.cursor[i]
                desired = path[k]
                delta = desired[:7] - joint[i, :7]
                waypoint = self.planner.fk(desired)
                actual_rotation = rotation(q[i].cpu().numpy())
                pose_reached = (np.linalg.norm(waypoint.translation - p[i].cpu().numpy()) < .008 and
                    np.linalg.norm(pin.log3(waypoint.rotation @ actual_rotation.T)) < .08)
                if np.max(np.abs(delta)) < .07 or pose_reached:
                    k += 1
                    self.cursor[i] = k
                    if k >= len(path):
                        self.phase[i] = 'align'
                        # Freeze the alignment goal. Repeatedly chasing handle
                        # +normal would unintentionally become an opening servo.
                        self.grasp_target[i] = self.target(int(i), False)
                        meta['reached_pregrasp'] = True
                        meta['pregrasp_tick'] = int(env.episode_length_buf[i])
                        continue
                    desired = path[k]
                    delta = desired[:7] - joint[i, :7]
                # Bound local tracking waypoint; execute ONLY original IK action.
                fraction = min(1., .045 / max(np.max(np.abs(delta)), 1e-8))
                virtual = joint[i].copy()
                virtual[:7] += fraction * delta
                goal = self.planner.fk(virtual)
            else:
                goal = self.grasp_target[i]
            quat = pin.Quaternion(goal.rotation).coeffs()
            tp = torch.tensor(goal.translation, dtype=torch.float32, device=env.device)[None]
            tq = torch.tensor([[quat[3], quat[0], quat[1], quat[2]]], dtype=torch.float32, device=env.device)
            pe, re = compute_pose_error(p[i:i + 1], q[i:i + 1], tp, tq, rot_error_type='axis_angle')
            scale = term._scale[i] if term._scale.ndim == 2 else term._scale
            action[i, :6] = (torch.cat((pe, re), -1) / scale).clamp(-1, 1)[0]
            action[i, 3:6].clamp_(-.35, .35)
            if phase in ['align', 'grasp']:
                if phase == 'align' and dist[i] < .04 and float(re.norm()) < .20:
                    self.phase[i] = 'grasp'
                if self.phase[i] == 'grasp':
                    action[i, 6] = -1
                    grasp = np.all(gt[i, 9:11] > .5) and dist[i] < .06 and .005 < joint[i, 7:9].sum() < .079
                    self.grasp_run[i] = self.grasp_run[i] + 1 if grasp else 0
                    if self.grasp_run[i] >= 3:
                        self.ready[i] = True
                        self.phase[i] = 'ready'
                        meta.update(regrasp=True, handback=True, handback_tick=int(env.episode_length_buf[i]), failure_stage=None)
        # Bound joint movements through the unchanged real DLS map. Check the
        # predicted segment too; reject an unsafe command instead of clipping
        # through an obstacle. Intended hand/finger contact allowed only at grasp.
        mask = torch.tensor(active, device=env.device)
        action = project(env, action, mask)
        jac = term._compute_frame_jacobian().clone()
        damping = term._ik_controller.cfg.ik_params['lambda_val']
        inverse = torch.linalg.solve(jac @ jac.transpose(1, 2) + damping ** 2 * torch.eye(6, device=env.device), jac).transpose(1, 2)
        delta = (inverse @ (action[:, :6] * term._scale)[:, :, None]).squeeze(-1).cpu().numpy()
        for i in np.flatnonzero(active):
            if self.phase[i] in ['failed', 'ready', 'release']:
                continue
            predicted = joint[i].copy()
            predicted[:7] += delta[i]
            allowed = self.phase[i] in ['align', 'grasp']
            if not self.planner.edge(joint[i], predicted, self.planner.world(int(i)), allow_handle=allowed):
                action[i, :6] = 0
                self.current[i]['tracking_collision_rejections'] = self.current[i].get('tracking_collision_rejections', 0) + 1
        return action.cpu().numpy(), time.perf_counter() - tick

    def records(self, i):
        return [dict(v) for v in self.history[i]]
