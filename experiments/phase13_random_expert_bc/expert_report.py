"""Generate the expert evidence report without inferring missing results."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


def main():
    summaries=[]
    for name in ['planner_pilot','planner_staged_pilot','planner_free_pilot','expert_validation']:
        path=R/name/'summary.json'
        if path.exists():summaries.append((name,json.loads(path.read_text())))
    validation=next((s for name,s in summaries if name=='expert_validation'),None)
    lines=['# Phase 13 随机工位专家验证报告','',
           '## 固定任务与随机范围','',
           '- 平台：已有 Isaac Lab / Isaac Sim Panda Door Opening，机器人、任务几何、动作接口、物理、Progress Reward 和 reset 流程不变。',
           '- 初始门角均匀分布 0–5°；工装整体 XYZ 平移各均匀分布 ±1cm；摩擦倍率固定 1。对应现有随机化代码 Level 2。',
           '- 工装整体平移同时移动门与把手；没有把手独立几何变形。GT 把手位姿是机器人工作空间坐标、wxyz 四元数。','',
           '## 专家方法','',
           '实测把手/转轴 → 前方上方避障中间点 → 接近 → 对齐 → 夹爪闭合 → 圆弧拉门。动作通过原有 Cartesian differential IK 驱动 Panda，开门依赖真实物理接触。没有门关节强制驱动、夹爪附着或机器人瞬移。',
           '开门阶段解除固定朝向约束。所有变更位于 Phase 13 新规划器；原 door_dataset/planner.py 保持不变。','',
           '## 全尝试执行结果','',
           '| 实验 | 成功 / 全部回合 | 成功率 | Wilson 95% CI | 仿真交互次数 |',
           '|---|---:|---:|---:|---:|']
    for name,s in summaries:
        lo,hi=s['success_95_wilson']
        lines.append(f"| {name} | {s['successes']}/{s['episodes']} | {s['success_rate']:.2%} | {lo:.2%}–{hi:.2%} | {s['simulator_interactions']:,} |")
    lines+=['','预检使用种子 13001，仅用于规划器调试。正式验证使用独立种子 13101，每个并行环境相同回合配额；统计包括所有成功与失败，未按成功筛选。','']
    if validation:
        passed=validation['success_rate']>.9
        lines.append(f"**专家门槛：{'通过' if passed else '未通过'}**；点估计 {validation['success_rate']:.2%}，要求严格高于 90%。置信区间单独报告，不把点估计当成总体确定保证。")
        rows=list(csv.DictReader((R/'expert_validation/attempts.csv').open()))
        lines+=['','### 初始角度分层（全部验证回合）','','| 初始角度 | 回合数 | 成功率 |','|---|---:|---:|']
        for lo,hi in [(0,1),(1,2),(2,3),(3,4),(4,5.001)]:
            selected=[r for r in rows if lo<=np.rad2deg(float(r['angle0_rad']))<hi]
            if selected:lines.append(f"| {lo}–{min(hi,5):g}° | {len(selected)} | {np.mean([int(r['success']) for r in selected]):.2%} |")
        failure=[r for r in rows if not int(r['success'])]
        phases={str(phase):sum(r['final_phase']==str(phase) for r in failure) for phase in range(7)}
        lines+=['',f'失败终止阶段计数（0休息、1避障、2接近、3对齐、4抓取、5拉门、6保持）：`{json.dumps(phases)}`。完整初态、角度和接触记录见 attempts.csv。','']
    collection=ROOT/'datasets/random_door_expert/collection_v1/summary.json'
    if collection.exists():
        s=json.loads(collection.read_text())
        lines+=['## 专家数据','',f"成功轨迹 **{s['saved_trajectories']} 条**，由 {s['episodes']} 次完整尝试得到；尝试成功率 {s['success_rate']:.2%}。仅保存成功轨迹用于 BC，因此这个成功率与独立全尝试验证分开报告。",'',
                '路径：`datasets/random_door_expert/collection_v1/trajectories.h5`。HDF5 每组一条轨迹，包含 observation/robot_state (26)、environment_state (13)、state (11 原始GT)、action (7)、reward、success、next_observation、next_environment_state、next_state、done、terminated、truncated、reset_parameters (5)、planner_phase。终止帧保存 reset 前真实状态。','',
                'environment_state 顺序：door_angle、target_angle、progress、remaining_angle、两指 contact_state、handle XYZ、handle quaternion wxyz。采集专家不加动作噪声。']
    split=ROOT/'datasets/random_door_expert/split_v1/split.json'
    if split.exists():
        m=json.loads(split.read_text());audit=json.loads((split.parent/'data_audit.json').read_text())
        lines+=['',f"整轨迹划分：{len(m['splits']['train'])} train / {len(m['splits']['validation'])} validation / {len(m['splits']['test'])} test，种子 13301。训练集计算归一化参数；测试集不参与拟合或 checkpoint 选择。",'',
                f"数据校验：{audit['trajectories']} 条轨迹、{audit['transitions']:,} 次有效转移；问题数 {len(audit['issues'])}。校验形状、有限值、时间连续性、终止边界、GT对齐、reset参数和采样范围。",'',
                f"HDF5 SHA256：`{m['dataset_sha256']}`。"]
    else:lines+=['','数据采集或划分尚未完成；未填入 BC 结果。']
    lines+=['','## 结论限制','',
            '通过专家门槛只能证明这个随机范围存在可执行控制解，尚不能证明 BC 能学会、环境参数有益或 RL 可以改善。BC 使用相同随机数据做五个配对 seed 的独立物理测试后再判断。',
            '专家成功筛选仍可能减少困难初态覆盖；因此完整随机分布的独立策略测试是必要条件。原有固定工位专家数据和 Phase 1–12 产物保持不变。','']
    settling=R/'initial_angle_settling.json'
    if settling.exists():
        s=json.loads(settling.read_text())
        lines+=['### 初始角扰动持续性','',
                f"采集轨迹的初始角平均{s['mean_initial_deg']:.3f}°，16个REST控制步后平均{s['mean_rest_end_deg']:.3f}°；{s['fraction_rest_end_under_half_degree']:.2%}已回到0.5°以内。",'',
                '继承环境的门关节位置目标为0，并具有恢复关闭位置的驱动。reset即时角度正确、工装XYZ扰动保留，但持续门角随机化的覆盖有限；因此专家成功率不能被解释为持续0–5°门角泛化。没有修改当前环境，详细只读探针与诊断见主报告。','']
    path=ROOT/'docs/random_expert_report.md';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines),encoding='utf-8');print(path)


if __name__=='__main__':main()
