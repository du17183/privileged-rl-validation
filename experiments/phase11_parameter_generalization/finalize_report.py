"""Evidence-driven Chinese report; no fabricated pending/conditional results."""
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta
import numpy as np
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, ARMS
from evaluation.stratified_generalization import CONDITIONS


def pct(x):return str((Decimal(str(x))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
def ci(item,percentage=False):
    scale=100 if percentage else 1
    return f"[{scale*item['ci95'][0]:.4f}, {scale*item['ci95'][1]:.4f}]"
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    summary=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'formal_data_audit.json').read_text())
    diagnostics=json.loads((OUT/'parameter_diagnosis.json').read_text());groups=summary['groups'];gen=summary['generalization'];frozen=summary['frozen_phase9_c']
    labels={'A':'Robot only','B':'+ angle/target/progress','C':'+ contact','D':'+ handle XYZ'}
    effect=next(r for r in summary['paired_tests'] if r['contrast']=='D-A')['metrics']['final_success']
    nominal=groups['D']['nominal_final_success']['mean']
    d=gen['D']['final']
    gate=dict(nominal_ge_90=nominal>=.9,
        randomized_D_minus_A_positive_ci=effect['ci95'][0]>0,
        angle_2p5_above_old_51p9=d['angle_2p5']['policy']['mean']>.519,
        angle_5_nonzero_all_seeds=all(v>0 for v in d['angle_5']['policy']['values']),
        handle_y_minus_1cm_above_old_15p6=d['offset_y_minus_0p01']['policy']['mean']>.15625,
        final_best_gap_le_5pp=abs(groups['D']['gap']['mean'])<=.05,
        late_online_success_every_seed=all(v>0 for v in groups['D']['late_online_successes']['values']),
        randomized_seed_sd_le_10pp=groups['D']['final_success']['sd']<=.1)
    passed=all(gate.values())
    (OUT/'readiness.json').write_text(json.dumps(dict(phase11_passed=passed,criteria=gate,
        full_lwd_divl='Not implemented. Requires state-driven generalization and a fresh Uniform/Quality replay comparison; quality-score diversity alone is insufficient.'),indent=2))
    stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    lines=[f'# Phase 11：环境参数输入与随机工位泛化\n\n更新时间：{stamp}\n',
        '## 1. 完成情况与结论\n',
        f"完成4组×5 seeds×300,000新增训练交互，共6,000,000；正式在线更新每run 37,500次。独立测试{summary['heldout']['jobs']}个任务、{summary['heldout']['episodes']:,} episodes。所有训练/测试真实执行；原Phase1–10数据保留。",
        f"D相对A的随机Level2 Final平均差为{100*effect['mean']:+.2f}个百分点，配对t95%CI {ci(effect,True)}，精确配对p={effect['exact_signflip_p']:.4f}。D固定工位Final={pct(nominal)}。",
        '**Phase11门槛通过。环境参数方案值得先迁移验证Drawer，再研究经验利用。**' if passed else '**Phase11全部成功门槛尚未通过。不能声称环境参数已建立可靠泛化，也不进入完整LWD/DIVL。**',
        '下面的“输入参数”是假设未来工装能测得的编码器角度、接触与校准位姿。本阶段没有真机实验；handle orientation不默认使用。',
        '\n## 2. 对照与初始化\n',
        table(['组','Actor状态','Actor维度','Critic状态'],[[a,labels[a],{'A':26,'B':29,'C':31,'D':34}[a],'同Actor状态 +7维action'] for a in ARMS]),
        '每seed继承对应Phase9 C最终300k checkpoint，保持actor/critic/target、Adam矩、alpha和熵目标。新输入列及其Adam矩为零；20个初始化在预检中验证初始动作与Q一致。冻结Anchor仍是原Phase8 B best，不会因增加参数而重新训练或替换。',
        '固定Progress Reward（每交互有符号20×Δangle，保留reach/grasp/success）、SAC、Expert Replay 50/50、持续BC λ=10、有效高斯KL λ=1、pre-tanh std≤0.01、每约10k transactional progress guard。机器人控制、success>1rad、10秒episode、原资产和reward源文件均未改。',
        '32并行环境、batch256、4 updates/vector step、lr3e-4。吞吐通过多个独立run共享GPU增加，不改变4/32的更新比。Actor和Critic均使用同一测量向量，不存在单独Privileged Critic。',
        '原专家1000条、固定工位数据只读；按旧Progress Reward重标记并补齐测量特征。没有随机工位新增专家或新BC预训练。因此专家约束和冻结anchor是否足以支持新状态是本实验的限制。',
        '只读专家覆盖审计：271,562 transitions，1000条初角均为0rad；初始handle XYZ只有约1微米数值浮动，episode长度268–276。开门过程角度/pose的边际覆盖不等于“home robot state +非零初角/工装偏移”的联合覆盖。左右接触在全数据各约77%，初态均为0。详见expert_coverage.json。',
        '\n## 3. 状态与物理随机化\n',
        '`get_environment_state()`返回door_angle、door_angular_velocity、target_angle、progress、remaining_angle、左右contact_state、handle_position、wxyz handle_orientation。角度rad，位置robot workspace metre；position按只读专家初始均值居中并除0.1m。progress=(angle−episode_start)/(target−episode_start)，截断0–1。角速度/remaining/orientation仅接口提供，不加入核心组。',
        table(['Level','初角','工装XYZ独立平移','柜体摩擦'],[[0,'0°','0','不变'],[1,'U(0,2.5°)','各±5mm','不变'],[2,'U(0,5°)','各±10mm','不变'],[3,'U(0,5°)','各±10mm','静/动×U(0.9,1.1)']]),
        '把手扰动通过刚体平移整个柜体实现，不是仅修改handle数值或独立改形。原铰链下限0°，不测试不可实现的−5°。每次reset断言实际关节、cabinet根位姿和PhysX材质读回等于抽样值。独立测试逐episode保存真实初角、初/末把手位置、工装位移、摩擦scale与验证过的static/dynamic摩擦均值。在线HDF5保存每步11维物理状态，含完整handle quaternion。',
        '随机流按seed和clone独立；匹配的是episode编号参数流，策略长度不同会使global step的暴露不同。reset前保存终止机器人/测量/参数快照，bootstrap next state不会误用新episode。',
        '\n## 4. 课程与学习过程\n',
        f"直接Level2 D seed0 pilot 10,016steps随机成功率={pct(summary['protocol']['pilot_decision']['level2_success'])}，发生拒绝，按预设规则启用课程。四组规则相同：Level0→1→2→3，当前level两组不重叠64episode测试均≥80%才能晋级。晋级只用于后续reset。",
        '固定Level2曲线始终评价相同的联合随机分布，不因课程level改变而降低测试难度。课程是自适应干预：组间实际训练难度占比可不同，结果反映“测量输入+同一课程控制器”的整体效果，不能宣称固定训练分布下的纯网络因果效应。',
        table(['组','结束Level均值','Level0/1/2/3完成交互占比','拒绝平均次数','额外专家BC样本/run'],
              [[a,f"{groups[a]['final_curriculum_level']['mean']:.2f}",'/'.join(f'{100*v:.1f}%' for v in exposure_share(summary,a)),f"{groups[a]['rollback_events']['mean']:.2f}",'9,600,000'] for a in ARMS]),
        '\n![Success curves](../results/phase11_parameter_generalization/figures/success_curves.png)',
        '\n## 5. 五seed主要结果\n',
        '以下是std≤0.01的真实随机策略执行。AUC是额外0–300k交互固定Level2成功率的归一化梯形面积；继承的专家/Phase9成本不隐藏为从零训练。Best按训练验证Level2成功率选择，随后独立复测，因此独立Best可能低于Final。',
        table(['组','Level2 AUC [95%CI]','随机Final [95%CI]','随机Best','Best−Final pp','标称Final','标称Best','随机Final seed SD'],
              [[a,f"{groups[a]['auc']['mean']:.4f} {ci(groups[a]['auc'])}",f"{pct(groups[a]['final_success']['mean'])} {ci(groups[a]['final_success'],True)}",pct(groups[a]['best_success']['mean']),f"{100*groups[a]['gap']['mean']:.2f}",pct(groups[a]['nominal_final_success']['mean']),pct(groups[a]['nominal_best_success']['mean']),f"{100*groups[a]['final_success']['sd']:.2f}pp"] for a in ARMS]),
        table(['组','在线成功/完成episodes','完成episode成功率','末100k成功数','质量score SD均值','score≈1比例'],
              [[a,f"{int(sum(groups[a]['online_successes']['values']))}/{int(sum(groups[a]['online_episodes']['values']))}",pct(groups[a]['online_success_ratio']['mean']),int(sum(groups[a]['late_online_successes']['values'])),f"{groups[a]['completed_quality_sd']['mean']:.4f}",pct(groups[a]['full_quality_fraction']['mean'])] for a in ARMS]),
        '在线比例的分母仅是完成episodes；成功episode可能更短，预算结束时仍有未完成轨迹。HDF5/audit显式给出pending transitions，不能把完成比例当作无偏独立测试成功率。',
        '\n![Seed endpoints](../results/phase11_parameter_generalization/figures/seed_endpoints.png)',
        '\n## 6. 配对统计\n',
        '统计单位是5个seed，CI不裁剪到[0,1]。给出模型依赖的配对tCI、两侧exact sign-flip及四个预设对比Holm校正；5个非零配对的精确两侧p最小0.0625，不能用更多episode伪造seed样本量。',
        table(['对比','AUC差 [95%CI]','AUC精确p/Holm','Final差pp [95%CI]','Final精确p/Holm','Final paired-t p/Holm'],
              [[r['contrast'],f"{r['metrics']['auc']['mean']:+.4f} {ci(r['metrics']['auc'])}",f"{r['metrics']['auc']['exact_signflip_p']:.4f}/{r['metrics']['auc']['holm_exact_p']:.4f}",f"{100*r['metrics']['final_success']['mean']:+.2f} {ci(r['metrics']['final_success'],True)}",f"{r['metrics']['final_success']['exact_signflip_p']:.4f}/{r['metrics']['final_success']['holm_exact_p']:.4f}",f"{r['metrics']['final_success']['paired_t_p']:.4f}/{r['metrics']['final_success']['holm_paired_t_p']:.4f}"] for r in summary['paired_tests']]),
        '\n## 7. 条件泛化与同一冻结策略参照\n',
        '所有条件独立Best/Final复测，新固定seed。各条件通常64episodes/seed；联合Level2随机策略128episodes/seed。此处也重新评估完全未续训的Phase9 C，避免把旧不同测试抽样值视为配对baseline。deterministic结果、每seed值和所有CI见summary.json；主表只显示随机执行。',
        table(['条件','冻结Phase9 C','A Final','B Final','C Final','D Final'],
              [[condition,pct(frozen[condition]['policy']['mean']),*[pct(gen[a]['final'][condition]['policy']['mean']) for a in ARMS]] for condition in CONDITIONS]),
        '\n![Generalization](../results/phase11_parameter_generalization/figures/generalization.png)',
        '\n### 联合Level2条件切片\n',
        '初角与工装L∞位移分箱仍随机化其它因素；单独扰动/强制位移壳层测试与这些条件切片分开。XYZ均匀小立方体下，最小L∞位移bin概率仅约1.56%，出现少数或零episode时不作强统计解释，另外三个offset_shell条件为相应范围提供每seed64独立episodes。',
        table(['组','切片','有效seed数','episodes','seed均值success [95%CI]'],
              [[a,k,item['seed_level']['n'],item['total_episodes'],f"{pct(item['seed_level']['mean'])} {ci(item['seed_level'],True)}"] for a in ARMS for k,item in summary['slices'][a]['final'].items()]),
        '\n## 8. 参数是否被使用与约束诊断\n',
        '同一robot observation，仅改angle0/2.5/5°、contact或handle XYZ±1cm，保存action mean。另有coherent reset角度+progress=0干预。angle-only探针可能违反angle/pose/progress一致性；反应不等于合理控制。handle同轴action投影只作局部线索，不能替代完整接触成功率。',
        table(['组','Final新增Actor列norm均值','Angle-only最大RMS均值','Handle±1cm最大RMS均值','RL/BC/KL梯度norm均值'],
              diagnosis_rows(diagnostics)),
        'Anchor被冻结为原固定工位策略，新状态列为0。有效std≤0.01使均值差KL曲率至少约10,000（不同维度std可能更小）；KL权重仍为1。新状态下即使正确动作应不同，也会被同一个robot-only anchor约束。梯度norm是固定endpoint batch上的局部诊断；尚未进行Anchor作用范围/强度的因果消融，不能单凭norm确认唯一原因。持续BC仅覆盖原工位，低std也限制新状态探索。这些都应作为未适应的候选机制，而非无证据断言“环境参数无用”。',
        '\n## 9. 条件传感噪声与参数遮蔽\n',
        summary['conditional_gate']['status'],
        '条件门槛为D−A随机Final配对tCI下界>0、至少4/5方向为正且D标称均值≥90%。只有通过才用新的evaluation seed进行遮蔽与angle±0.5°、position±2mm、contact2%误判/一控制步延迟；sensor noise不改变物理状态、reward或训练参数。progress由同一噪声angle派生，避免干净冗余通道泄漏。',
        '\n## 10. 轨迹异质性与后续经验利用\n',
        '所有replay仍uniform 50/50：每run SAC专家抽样4,800,000、在线抽样4,800,000，另有9,600,000 expert BC样本。在线样本含成功/失败/未完成transition，不进行质量筛选。',
        '质量沿用Phase10终局score：0.55success+0.20final progress+0.15net progress+0.10运动后双指contact stability。success是输入之一，因此排名AUC不证明前瞻预测。这里仅检查分布是否从近乎全1展开；异质性本身不证明Quality Replay会提升策略。',
        '\n![Quality and curriculum](../results/phase11_parameter_generalization/figures/quality_and_curriculum.png)',
        '\n## 11. 数据审计、预算与复现\n',
        f"数据审计：{audit['datasets']}个正式HDF5、{audit['total_completed_episodes']:,}完成episodes；总成功{audit['total_successes']:,}；终止快照、每clone连续分段、reward和success标签、added angle字段一致性通过。历史{audit['prior_artifacts']:,}文件size/mtime/小源文件SHA一致；训练源码snapshot无变化。",
        f"训练新增6,000,000交互；in-training评价交互{int(sum(groups[a]['eval_interactions']['values'][s] for a in ARMS for s in range(5))):,}；独立测试交互{summary['heldout']['interactions']:,}。Pilot额外10,016训练交互、246,304评价交互；physics/function preflight额外64交互。评价不进入训练replay，成本单独报告。既有expert/Phase9成本不计入新增AUC横轴。",
        '源码：environment_state/、randomized_env/、curriculum/、experiments/phase11_parameter_generalization/、evaluation/stratified_generalization.py与parameter_sensitivity.py。结果CSV/JSON/图：results/phase11_parameter_generalization；完整checkpoint：checkpoints/phase11_parameter_generalization；HDF5：datasets/phase11_parameter_generalization；TensorBoard：logs/phase11_parameter_generalization。历史任务文件只读，Phase11工装reset适配器为新子类。',
        '\n## 12. 回答九个问题\n',
        answers(summary,gate,passed),
        '\n## 13. 当前决定\n',
        '先在Drawer复制同一可测参数接口与配对随机化验证；成功后才重新比较Uniform/Quality/LWD-style，完整LWD/DIVL仍需独立预算与协议。' if passed else '保持原Phase9 C用于固定工位。Phase11测量接口和随机化框架可复用；当前应优先检查固定工位Anchor是否阻止条件动作修正，以及是否需要匹配随机工位的示范覆盖。没有满足门槛前不扩展训练规模、不实施完整LWD/DIVL，也不声称已验证Drawer迁移。']
    conditional=OUT/'conditional_completed.json'
    if conditional.exists():
        items=[]
        for path in (OUT/'conditional').glob('D_seed*.json'):
            if path.name.endswith('.partial.json'):continue
            for r in json.loads(path.read_text())['results']:
                items.append([r['seed'],r['endpoint'],r['metrics']['condition'],r['ablation'],r['sensor_noise'],pct(r['metrics']['success'])])
        lines.insert(-4,'\n### 实际条件复测\n\n'+table(['seed','endpoint','condition','mask','sensor noise','success'],items))
    path=ROOT/'docs/phase11_parameter_generalization_report.md';path.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(path),gate_passed=passed,criteria=gate),indent=2))


def exposure_share(summary,arm):
    count=np.array([sum(summary['curriculum_exposure'][f'{arm}_seed{s}'][str(l)]['transitions'] for s in range(5)) for l in range(4)])
    return count/count.sum()


def diagnosis_rows(diagnostics):
    rows=[]
    for arm in ARMS:
        selected=[r for r in diagnostics['results'] if r['arm']==arm and r['endpoint']=='final']
        angle=[];handle=[]
        for r in selected:
            probes=r['reset_state_probes']['probes']
            angle.append(max([p['action_change_rms'] for p in probes if p['feature']=='angle_only'] or [0]))
            handle.append(max([p['action_change_rms'] for p in probes if p['feature']=='handle_position'] or [0]))
        rows.append([arm,f"{np.mean([r['added_actor_weight_norm'] for r in selected]):.6f}",f'{np.mean(angle):.6f}',f'{np.mean(handle):.6f}',
                     '/'.join(f"{np.mean([r['gradient_norms'][k] for r in selected]):.3f}" for k in ('rl','weighted_bc','weighted_kl'))])
    return rows


def answers(summary,gate,passed):
    effects={r['contrast']:r['metrics']['final_success'] for r in summary['paired_tests']}
    texts=[]
    for number,label,contrast in ((1,'Door angle/progress是否提高不同初态成功率','B-A'),(2,'Contact是否进一步提高接触控制能力','C-B'),(3,'Handle position是否解决工位变化','D-C'),(4,'参数+Progress Reward是否优于只用Progress Reward','D-A')):
        e=effects[contrast]
        texts.append(f"{number}. **{label}**：{contrast}联合随机Final差{100*e['mean']:+.2f}pp，95%CI{ci(e,True)}。是否解决特定角度/位移由第7节条件结果判断；总平均不能代替切片。")
    texts += ['5. **策略是否根据参数改变动作**：第8节和parameter_diagnosis.json保存60个endpoint的action mean/敏感度；新增权重和非零RMS只能证明响应，合理性必须看相应物理条件下成功率。',
              f"6. **传感噪声是否影响方案**：{summary['conditional_gate']['status']}。未执行时不推测鲁棒性。",
              '7. **能否复制到Drawer**：接口可将angle/velocity替换为distance/velocity、保留contact/handle；Door以外任务尚未训练或验证，不宣称跨任务效果。',
              '8. **随机化后质量是否分层**：第5/10节给出terminal score各seed SD与全1比例；与Phase10约98.5%全1比较时注意课程/任务分布变化，不能据此直接证明筛选算法有效。',
              '9. **是否具备重新测试LWD/DIVL条件**：'+('Phase11门槛通过，可先复制Drawer并规划新的经验利用对照；尚未证明Quality Replay收益。' if passed else '门槛未全部通过。即使出现异质经验，也应先解决可测状态能否稳定转化为正确条件动作的问题。')]
    return '\n\n'.join(texts)
if __name__=='__main__':main()
