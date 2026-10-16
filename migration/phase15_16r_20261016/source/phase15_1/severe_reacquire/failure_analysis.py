"""Diagnostic observations never change action, reset, physics or routing."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from phase15.common import save_json
from phase15.policy_client import ModularPolicy
from phase15.recovery_policy.state import relative_features
from phase15_1.common import ROOT, RESULTS


PHASES = ['idle', 'failed', 'release', 'path', 'align', 'grasp', 'ready', 'retreat', 'pregrasp']


class DiagnosticPolicy(ModularPolicy):
    def configure(self, cohort=None):
        super().configure(cohort)
        self.trace = []
        self.frame = 0
        self.before_action = None
        self.measured_latency = []

    def forward(self, x):
        action = super().forward(x)
        self.before_action = self.current_raw.copy()
        return action

    def observe(self, robot, gt, done):
        if self.before_action is not None and np.any(self.active):
            r, g = robot.detach().cpu().numpy(), gt.detach().cpu().numpy()
            q = self.before_action[:, :9]
            bounds = self.env.scene['robot'].data.joint_pos_limits.cpu().numpy()
            margins = np.minimum(q - bounds[..., 0], bounds[..., 1] - q)
            scale = self.env.action_manager.get_term('arm_action')._scale.cpu().numpy()
            command = self.last_action[:, :6] * scale
            for i in np.flatnonzero(self.active):
                model = None
                if self.planner is not None and self.frame % 10 == 0:
                    model = self.planner.planner.collision_details(q[i], self.planner.planner.world(int(i)))
                plan = self.planner.current[i] if self.planner is not None else None
                phase = self.planner.phase[i] if self.planner is not None else 'idle'
                self.trace.append(dict(clone=int(i), batch=int(self.batch), episode=int(self.episode[i]),
                    robot_state=self.before_action[i, :26].tolist(), environment_state=self.before_action[i, 26:].tolist(),
                    eef_handle_relative_pose=relative_features(self.before_action[i:i+1])[0, 1:].tolist(),
                    joint_limit_margin=margins[i].tolist(), ik_reachable=(plan.get('ik_success') if plan else None),
                    collision_model=model, contact_after=g[i, 9:11].tolist(),
                    policy_phase=int(self.router.state[i]), planner_phase=str(phase),
                    remaining_ticks=int(600-round(self.before_action[i, 25]*600)),
                    action=self.last_action[i].tolist(), commanded_displacement=command[i].tolist(),
                    actual_displacement=(r[i, 18:21]-self.before_action[i, 18:21]).tolist(),
                    actual_next_eef_pose=r[i, 18:25].tolist(), next_door_angle=float(g[i, 0]),
                    terminal=bool(done[i]), success=bool(done[i] and g[i, 0] > 1.)))
            self.frame += 1
        super().observe(robot, gt, done)


def classify(row, trace):
    if row['success']:
        return 'success', 'Original final door angle exceeds 1rad'
    plans = row['planner']
    if not plans:
        return 'A', 'No planner takeover; no safe retreat performed'
    plan = plans[-1]
    if plan.get('handback'):
        return 'H', 'Valid grasp handed back; frozen Normal did not finish opening'
    if plan.get('failure') == 'start_collision_or_joint_limit':
        return 'A', 'Start rejected; old module has no safe retreat from this state'
    if not plan.get('ik_success'):
        return 'C', 'No IK solution found by the tested solver and seeds'
    if not plan.get('planning_success'):
        return 'B', 'No collision-checked path to pregrasp; not proof of global unreachability'
    if plan.get('failure_stage') == 'time_budget' or row['original_episode_ticks'] >= 600 and not plan.get('reached_pregrasp'):
        return 'I', 'Planner or original episode budget exhausted'
    if not plan.get('reached_pregrasp'):
        return 'E', 'Pregrasp tracking/pose did not reach the tested tolerance'
    recent = [t for t in trace if t['planner_phase'] in ['align', 'grasp']]
    if not any(all(c > .5 for c in t['contact_after']) for t in recent):
        return 'F', 'Pregrasp reached but no measured bilateral finger contact'
    return 'G', 'Contact was measured but the valid-grasp hold did not pass'


def write_analysis(directory):
    directory = Path(directory)
    rows = json.loads((directory / 'natural_severe.telemetry.json').read_text())
    trace = json.loads((directory / 'natural_severe.trace.json').read_text())
    counts = Counter()
    classified = []
    for row in rows:
        group = int(row['cohort_id'].split('_')[-1]) // 32
        selected = [t for t in trace if t['clone'] == row['clone'] and t['batch'] == group]
        kind, why = classify(row, selected)
        counts[kind] += 1
        classified.append(dict(cohort_id=row['cohort_id'], success=row['success'], category=kind,
            reason=why, trace_steps=len(selected), final_angle_rad=row['final_angle_rad']))
    save_json(directory / 'failure_classification.json', dict(counts=dict(counts), episodes=classified,
        classification_limits='Model collision blockage is separate from measured finger contact. D(actual collision) requires instrumentation; no fabricated actual collision label. IK solver failure is not a mathematical unreachability proof.', rl_updates=0))
    with (directory / 'failure_classification.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(classified[0]));w.writeheader();w.writerows(classified)
    result = json.loads((directory / 'natural_severe.json').read_text())
    old = json.loads((ROOT / 'phase15/results/test/D_seed0/natural_severe.json').read_text())
    save_json(RESULTS / 'diagnosis/gate.json', dict(recorded_episodes=len(rows), trace_rows=len(trace),
        required_state_and_command_fields_present=all(all(k in t for k in ['robot_state','environment_state','eef_handle_relative_pose','joint_limit_margin','ik_reachable','collision_model','policy_phase','planner_phase','remaining_ticks','commanded_displacement','actual_displacement']) for t in trace),
        current_success=result['success'], phase15_same_seed_success=old['success'],
        reproduction_difference=result['success']-old['success'], policies_frozen=True, rl_updates=0))
    lines = ['# Phase15.1 严重偏离失败专项诊断', '',
        '先复现Phase15旧GT Re-acquire；本次只有观测与日志新增，动作、Router、权重、任务、reset及600tick预算保持不变。', '',
        f'单seed0、原Phase15 test自然严重偏离64 episodes。本次成功率{result["success"]*100:.2f}%，原同seed为{old["success"]*100:.2f}%。这是诊断复现，不能替代Phase15.1五seed独立结果。', '',
        '| 类别 | episode数量 |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in sorted(counts.items())]
    lines += ['', 'A=起点拒绝/无安全撤离；B=未找到pregrasp安全路径；C=测试IK求解失败；E=pregrasp姿态/跟踪未到位；F=未建立双指接触；G=接触未通过有效抓取保持；H=交还后未打开；I=时间耗尽。', '',
        'D（实测物理碰撞）不能用碰撞模型拒绝替代。当前记录模型碰撞对与任务双指接触，未检测到的腕部/臂部物理碰撞属于观测限制，不应写成“没有碰撞”。', '',
        '逐控制步保存robot q/qd、EEF/handle pose、相对pose、门角、接触、硬关节余量、IK尝试结果、模型碰撞检查、策略阶段、剩余预算、指令位移及实际位移。未检查模型碰撞的tick明确为null，不使用未来信息补填。', '',
        f'文件：`{directory.relative_to(ROOT)}/natural_severe.trace.json`、`failure_classification.csv`及`phase15_1/results/diagnosis/gate.json`。', '',
        '后续先尝试受检查的撤离，使起点进入自由空间，再做包含朝向的pregrasp规划及有效抓取。不能取消碰撞检查来虚报恢复率。']
    (ROOT / 'docs/phase15_1_severe_failure_analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(diagnosis_counts=dict(counts), episodes=len(rows), trace_rows=len(trace))), flush=True)
