"""Five-seed paired inference, stage diagnostics and explicit phase decisions."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import stats
from phase15.common import ROOT, RESULTS, CHECKPOINTS, save_json

ART = ['ee_offset', 'contact_loss', 'door_regression', 'stagnation']
TESTS = ['fixed', 'random', *ART, 'natural_severe', 'full_loop']


def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    margin = float(stats.t.ppf(.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return dict(mean=mean, sd=sd, ci95=[mean - margin, mean + margin], seeds=values.tolist())


def paired(a, b):
    delta = np.asarray(a) - np.asarray(b)
    if np.allclose(delta, 0):
        p = 1.
    elif delta.std(ddof=1) < 1e-12:
        p = 0.
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    null = np.array([np.mean(delta * sign) for sign in itertools.product([-1, 1], repeat=len(delta))])
    exact = float(np.mean(np.abs(null) >= abs(delta.mean()) - 1e-12))
    return dict(delta=interval(delta), paired_t_p=p, exact_two_sided_signflip_p=exact)


def cell(d):
    m, s = d['mean'] * 100, d['sd'] * 100
    lo, hi = np.clip(np.asarray(d['ci95']) * 100, 0, 100)
    return f'{m:.2f} ± {s:.2f} [{lo:.2f}, {hi:.2f}]'


def main():
    selection = json.loads((RESULTS / 'selection.json').read_text())
    pilot = json.loads((RESULTS / 'pilot_gate.json').read_text())
    metrics = {}
    detailed = {}
    episodes = {}
    for arm in ['A', 'B', 'C', 'D']:
        perseed = {t: [] for t in TESTS}
        detailed[arm] = {}
        episodes[arm] = {}
        for seed in range(5):
            directory = RESULTS / 'test' / f'{arm}_seed{seed}'
            detailed[arm][str(seed)] = {}
            for test in TESTS:
                d = json.loads((directory / (test + '.json')).read_text())
                perseed[test].append(d['success'])
                detailed[arm][str(seed)][test] = d
                episodes[arm].setdefault(test, []).extend(json.loads((directory / (test + '.telemetry.json')).read_text()))
        perseed['artificial_macro'] = np.mean([perseed[t] for t in ART], axis=0).tolist()
        metrics[arm] = {k: interval(v) for k, v in perseed.items()}
    comparisons = []
    for left, right in [('C', 'A'), ('D', 'C'), ('D', 'B')]:
        for metric in ['full_loop', 'random', 'artificial_macro', 'natural_severe']:
            comparisons.append(dict(comparison=left + '-' + right, metric=metric,
                **paired(metrics[left][metric]['seeds'], metrics[right][metric]['seeds'])))
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]['paired_t_p'])
    prior = 0.
    for rank, i in enumerate(order):
        prior = max(prior, min(1., comparisons[i]['paired_t_p'] * (len(comparisons) - rank)))
        comparisons[i]['paired_t_holm_p'] = prior
    ablation = {}
    best = selection['selected_variant']
    for variant in ['R0', 'R1', 'R2']:
        values = {t: [] for t in ART + ['natural_severe']}
        for seed in range(5):
            label = 'C' if variant == best else 'C_' + variant
            directory = RESULTS / 'test' / f'{label}_seed{seed}'
            for test in values:
                values[test].append(json.loads((directory / (test + '.json')).read_text())['success'])
        values['artificial_macro'] = np.mean([values[t] for t in ART], axis=0).tolist()
        ablation[variant] = {k: interval(v) for k, v in values.items()}
    targets = dict(random=metrics['D']['random']['mean'] >= .9,
        fixed=metrics['D']['fixed']['mean'] >= .95, artificial=metrics['D']['artificial_macro']['mean'] >= .9,
        natural_severe=metrics['D']['natural_severe']['mean'] > .5,
        full_loop=metrics['D']['full_loop']['mean'] >= .9,
        actual_reacquire_normal_success=any(r['success'] and any(p.get('handback') for p in r['planner'])
                                          for r in episodes['D']['full_loop']))
    latency = {}
    for arm in ['A', 'B', 'C', 'D']:
        latency[arm] = [json.loads((RESULTS / 'test' / f'{arm}_seed{s}' / 'batch1_latency.json').read_text()) for s in range(5)]
    summary = dict(metrics=metrics, paired_comparisons=comparisons, recovery_ratios=ablation,
        validation_selection=selection, success_targets=targets, algorithm_targets_met=all(targets.values()),
        single_robot_latency=latency, rl_updates=0,
        ci_note='Student-t CI across five training seeds; clipped only in display, raw CI retained',
        signflip_note='Five paired seeds have minimum exact two-sided p=0.0625; do not claim p<0.05 exact evidence')
    save_json(RESULTS / 'summary.json', summary)
    with (RESULTS / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'metric', 'mean', 'sd', 'ci95_low', 'ci95_high', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4'])
        w.writeheader()
        for arm, group in metrics.items():
            for name, d in group.items():
                w.writerow(dict(arm=arm, metric=name, mean=d['mean'], sd=d['sd'], ci95_low=d['ci95'][0], ci95_high=d['ci95'][1],
                    **{f'seed{i}': v for i, v in enumerate(d['seeds'])}))
    paired_index = {(d['comparison'], d['metric']): d for d in comparisons}
    ca_art = paired_index[('C-A', 'artificial_macro')]
    ca_loop = paired_index[('C-A', 'full_loop')]
    db_natural = paired_index[('D-B', 'natural_severe')]
    severe_plans = [p for row in episodes['D']['natural_severe'] for p in row['planner']]
    rejected_starts = sum(bool(p.get('start_collision')) for p in severe_plans)
    lines = ['# Phase 15：Normal / Recovery / GT Re-acquire', '',
        '## 实验边界', '',
        '服务器 b300-1。未运行 SAC、在线 RL、Residual RL、LWD 或 DIVL。保持原 Panda Door 环境、Progress Reward B、reset、动作尺度与 600 tick episode。', '',
        'Normal 为 Phase14.3 C 的逐 seed 验证集选择权重，冻结。Recovery 从同 seed 的 Phase14.3 D 初始化，再以 47 维输入专门训练。GT 是未来工装可以在线提供的状态，评估阶段同样使用。', '',
        '迁移核验见 `phase15/results/environment_audit.json` 和 `phase15/results/bootstrap/`。原始科学源码 142 个文件、S/R 原始划分及 111 个历史已选权重完成 SHA256 核验。两份安装生成的 egg-info 元数据差异已记录。原 π0.5 的 Transformers 分发元数据与被替换源码不一致；复用了全部 2218 个原环境 Python 源文件及实际 tokenizers0.21.4。', '',
        '## 方法与公平性', '',
        '| 组 | Normal | Recovery | Severe |', '|---|---|---|---|',
        '| A | 冻结 Phase14.3 C | 同一 C | 同一 C |',
        '| B | 冻结 Phase14.3 D | 同一 D | 同一 D |',
        '| C | 冻结 C | 新训练专用 Recovery | Recovery |',
        '| D | 冻结 C | 同 C 组 Recovery | GT 碰撞检查规划 + IK |', '',
        f'验证集选择的恢复比例：**{best}**。每种比例 5 seeds、额外 5000 optimizer updates、batch128、rank16 LoRA；全32维 flow loss，非法 action-horizon mask 保留。1250/2500/5000 checkpoint 只用验证集选择，测试集没有用于调整。', '',
        'R0=100% R，R1=80% R+20% S，R2=50% R+50% S；累计采样比例精确。所有组使用相同原始轨迹级划分和 normalization。Recovery 追加 target、handle−EEF XYZ 和相对 quaternion，额外特征统计仅来自 train S/R。', '',
        '**因果限制：**模块组同时改变了恢复输入、优化预算与执行分工；与原 C/D 的差异不能单独证明 S/R 混合存在行为干扰。R 比例消融只支持该专用恢复配方内的比较。', '',
        '## 正式结果', '', '以下均为百分比：5 seed 均值 ± 标准差 [95% CI]。', '',
        '| 组 | 固定 | 随机 | 人工恢复宏平均 | 自然严重偏离 | 新 seed 完整闭环 |', '|---|---|---|---|---|---|']
    for arm in ['A', 'B', 'C', 'D']:
        lines.append('| ' + arm + ' | ' + ' | '.join(cell(metrics[arm][m]) for m in ['fixed', 'random', 'artificial_macro', 'natural_severe', 'full_loop']) + ' |')
    lines += ['', '固定64、随机128、人工每类64、自然严重64 episodes/seed。全闭环128 episodes/seed 从新注册的随机 reset 开始，无人工扰动、无恢复快照写入。历史压力测试保留原 prefix/snapshot 重建，1个 contact warm tick 计入原600 tick；handoff 后没有状态写入。', '',
        f'上述CI以5个训练seed为单位，不是有限evaluation episodes的二项置信区间；固定工位跨seed零方差也不意味着真实成功率绝对为100%。成功标准按预设均值判断。D随机{metrics["D"]["random"]["mean"]*100:.2f}%的95%CI为{metrics["D"]["random"]["ci95"][0]*100:.2f}%–{metrics["D"]["random"]["ci95"][1]*100:.2f}%，达标判断与不确定性需一起解释。', '',
        '## Pilot 与修复记录', '',
        f'正式运行通过的 functional gate：`{pilot["pilot_jobs"]}`。Pilot Recovery 仅50次更新，单seed、每类32 episodes，不能作为正式多seed结果。', '',
        'v4中压力测试已经完成的clone自动reset后再次跨过原handoff时刻，造成额外交接计数及不必要推理。v5将完成clone保持inactive直到下一cohort batch，并逐cohort保存缓存清除和实际刷新证据。独立64样本回归检查和受影响真实物理测试保留在Phase15命名空间，v4失败记录未覆盖。', '',
        '严重偏离起点独立审计见 `phase15/results/pilot/fix_v5/start_collision_audit.json`：32个历史状态中，16个触及规划器6mrad保守关节余量，23个被URDF/把手OBB模型判为场景碰撞，二者有重叠，29个总体被拒绝，没有自碰撞。该模型拒绝不等同于实测穿透。未取消碰撞检查、未放宽关节余量以换取成功率；碰撞起点的安全撤离仍是本轮规划器限制。', '',
        '## 恢复比例消融', '', '| 比例 | 人工恢复 | 自然严重偏离（无规划器） |', '|---|---|---|']
    for variant in ['R0', 'R1', 'R2']:
        lines.append(f'| {variant} | {cell(ablation[variant]["artificial_macro"])} | {cell(ablation[variant]["natural_severe"])} |')
    lines += ['', '## 配对统计', '', '| 比较 | 指标 | 均值差（百分点） | 配对 t p | Holm p | 精确符号翻转 p |', '|---|---|---|---|---|---|']
    for d in comparisons:
        lines.append(f'| {d["comparison"]} | {d["metric"]} | {d["delta"]["mean"]*100:.2f} | {d["paired_t_p"]:.4g} | {d["paired_t_holm_p"]:.4g} | {d["exact_two_sided_signflip_p"]:.4g} |')
    lines += ['', '以训练 seed 为统计单位，12项预设主比较一起 Holm 校正。5个配对 seed 的双侧精确符号翻转 p 最低为0.0625；不能据此宣称精确检验 p<0.05。', '',
        f'C−A人工恢复均值差{ca_art["delta"]["mean"]*100:.2f}个百分点，Holm校正t检验p={ca_art["paired_t_holm_p"]:.4g}；完整闭环差{ca_loop["delta"]["mean"]*100:.2f}个百分点，校正p={ca_loop["paired_t_holm_p"]:.4g}。D−B严重偏离差{db_natural["delta"]["mean"]*100:.2f}个百分点，校正p={db_natural["paired_t_holm_p"]:.4g}。本轮不能宣称稳定的整体显著收益；规划接管对严重偏离的负差也需结合小seed精确检验及失败阶段解释。', '',
        f'{best}通过验证集锁定。R0测试人工恢复均值{ablation["R0"]["artificial_macro"]["mean"]*100:.2f}%，R2为{ablation["R2"]["artificial_macro"]["mean"]*100:.2f}%。测试消融不能用于重新选型；本轮二者CI重叠，不能宣布100%恢复数据最佳。', '',
        '## 恢复阶段与瓶颈', '', '| 组/测试 | 检测事件 | Recovery触发 | Re-acquire触发 | 平均切换 | IK | 规划 | 到达pregrasp | 重抓 | 交还后成功 |', '|---|---|---|---|---|---|---|---|---|---|']
    for arm in ['C', 'D']:
        for test in ['random', 'natural_severe', 'full_loop']:
            rows = episodes[arm][test]
            plans = [p for r in rows for p in r['planner']]
            def rate(key):
                return f'{np.mean([p.get(key,False) for p in plans])*100:.1f}%' if plans else '—'
            handed = [p for p in plans if p.get('handback')]
            after = f'{np.mean([p["final_task_success"] for p in handed])*100:.1f}%' if handed else '—'
            lines.append(f'| {arm}/{test} | {sum(len(r["events"]) for r in rows)} | {sum(r["recovery_triggers"] for r in rows)} | {len(plans)} | {np.mean([r["switches"] for r in rows]):.2f} | {rate("ik_success")} | {rate("planning_success")} | {rate("reached_pregrasp")} | {rate("regrasp")} | {after} |')
    lines += ['', '完整阶段信息见每个测试的 `.telemetry.json` 和结果中的 `recovery_stages`：重新接近持续6 ticks，双指接触/有效抓取持续3 ticks，超过触发门角0.05rad的正向进度持续30 ticks。检测事件指标是条件指标；人工压力测试还保留全样本的原始阶段率，避免只汇报已检测事件。末端扰动 assignment%18 对应XYZ三轴、±5/10/20mm。', '',
        f'正式D组自然严重偏离共有{len(severe_plans)}次规划接管事件，{rejected_starts}次在起点检查被拒绝（{rejected_starts/max(len(severe_plans),1)*100:.2f}%）。这是事件条件率，不能混同episode成功率；同一episode可能多次接管。', '',
        '规划使用 Panda URDF collision geometry、自碰撞对、当前启用柜体碰撞形状的保守 OBB 与地面检查；关节边步长≤0.03rad。FK 一致性 gate 后才执行。仍然是离散规划及局部跟踪检查，不构成连续轨迹的安全证明；模型拒绝碰撞候选的次数也不是实测物理碰撞率。', '',
        'Re-acquire 不设置机器人关节状态/目标，不包含开门 servo；预抓取到位后固定对齐目标，建立抓取后立即交还 Normal。剩余 episode 时间、IK、候选碰撞和跟踪拒绝原因逐事件保存。', '',
        '## 实时性', '', '| 组 | 单机器人Normal平均推理 | Recovery平均推理 | 83.3ms重规划预算 |', '|---|---|---|---|']
    for arm, values in latency.items():
        normal = np.mean([d['normal']['mean_rpc_s'] for d in values])
        rec = [d['recovery']['mean_rpc_s'] for d in values if 'recovery' in d]
        recovered = f'{np.mean(rec)*1000:.1f}ms' if rec else '—'
        met = all(v['budget_met'] for d in values for v in d.values())
        lines.append(f'| {arm} | {normal*1000:.1f}ms | {recovered} | {"满足" if met else "未满足"} |')
    lines += ['', '使用10步 chunk、执行5步、仿真目标重规划12Hz；每次 switch/reset 清缓存。单机器人 benchmark 每模块 warmup3次、计时5次。每 episode 总推理时间记录在 telemetry；向同一 batch 中的每个参与 episode 计入完整请求延迟，因此不能相加理解为独占 GPU 总成本。仿真在推理时等待，仿真成功不代表60Hz实时部署已成立。', '',
        '## 成功标准及下一阶段', '']
    for name, passed in targets.items():
        lines.append(f'- {name}: {"达到" if passed else "未达到"}。')
    passed = all(targets.values())
    lines += ['', f'**可靠 Randomized Anchor 的算法目标：{"本轮全部达到，可作为候选Anchor；仍需依据逐seed波动及实时性判断部署条件" if passed else "本轮尚未全部达到，不能宣称已建立可靠Anchor"}。**', '',
        '### 对用户问题的回答', '',
        f'1. Normal/Recovery是否优于单策略：以C−A的完整闭环及恢复配对结果为依据；完整闭环均值差为{(metrics["C"]["full_loop"]["mean"]-metrics["A"]["full_loop"]["mean"])*100:.2f}个百分点。',
        '2. 是否证明混合数据任务干扰：尚不能隔离因果；需要等输入、等优化预算的单策略/分工对照。',
        f'3. Recovery人工恢复能力：C组{metrics["C"]["artificial_macro"]["mean"]*100:.2f}%，D组{metrics["D"]["artificial_macro"]["mean"]*100:.2f}%。',
        f'4. GT重新抓取是否解决严重偏离：D组{metrics["D"]["natural_severe"]["mean"]*100:.2f}%，D−C为{(metrics["D"]["natural_severe"]["mean"]-metrics["C"]["natural_severe"]["mean"])*100:.2f}个百分点；本轮没有解决该问题，具体失败阶段见上表及逐事件日志。',
        '5. 工装GT的价值：提供确定性触发、相对几何和规划目标；本阶段没有无GT router对照，不能独立量化GT开关的因果贡献。',
        f'6. 完整随机工位是否超过90%：D随机{metrics["D"]["random"]["mean"]*100:.2f}%，新seed全闭环{metrics["D"]["full_loop"]["mean"]*100:.2f}%。',
        '7. 实时性是否瓶颈：由单机器人benchmark与83.3ms预算比较，不能用batch32吞吐替代单机器人延迟。',
        f'8. 是否得到可靠Anchor：{"达到列出的算法均值目标，可冻结为候选系统；需保留seed不确定性与实时限制" if passed else "未全部达到，继续针对恢复阶段瓶颈修复"}。',
        f'9. 是否允许Phase16：{"可讨论小规模受控验证，先冻结Recovery/Re-acquire，且须解决实时执行安排" if passed else "暂不进入在线RL，先补齐未满足的Anchor条件"}。',
        '10. LWD/DIVL条件：本阶段未进行在线改善，尚未证明RL超过冻结Anchor；继续后置。', '',
        '### 建议的下一步', '',
        '先处理严重偏离的碰撞/关节边界起点：区分保守模型误报与实际接触，设计经检查的安全撤离，再接入pregrasp；不能直接取消安全检查。随后定位剩余失败发生在重抓、剩余时间不足还是Normal接管姿态偏离专家分布。保持原600tick预算，并单独记录交接时刻与剩余时间。', '',
        '并行处理单机器人实时推理：当前约210–222ms，超过83.3ms重规划预算。优化推理实现或评估有边界的异步执行安排后，重新验证动作与切换行为。继续保持Normal冻结，以最小受影响的验证集测试确认修复，下一轮使用新的独立测试集；不使用本轮正式测试集反复调参。', '',
        '最终交付独立核验见 `phase15/results/delivery_verification.json`：从570份正式结果、对应episode CSV与telemetry复算30400个episodes，并核验5seed CI、12项配对检验、Holm校正、权重/压力cohort SHA及600tick预算。逐seed规划阶段计数另存 `phase15/results/stage_metrics.csv`。', '',
        '## 复现', '', '```bash', 'source configs/runtime_env.sh', 'python3 -m phase15.audit',
        '.venv_pi05/bin/python -m phase15.recovery_policy.prepare',
        '# 先重新执行并通过独立单seed gate；正式runner拒绝跳过gate',
        f'python3 -m phase15.run --pilot-jobs {pilot["pilot_jobs"]}', '```', '',
        '原Phase1–14.3结果只读保留。所有新训练、评估、selection、CSV、统计和生命周期位于Phase15命名空间。']
    path = ROOT / 'docs/phase15_normal_recovery_report.md'
    path.write_text('\n'.join(lines) + '\n')
    print('REPORT WRITTEN ' + str(path), flush=True)


if __name__ == '__main__':
    main()
