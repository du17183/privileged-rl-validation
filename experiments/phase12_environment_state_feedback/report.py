"""Chinese report built only from completed raw five-seed results."""
import json
from decimal import Decimal,ROUND_HALF_UP
from datetime import datetime,timezone
import numpy as np
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,CKPT,VARIANTS,run_name
def pct(v):return str((Decimal(str(v))*100).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))+'%'
def effect(m):return f"{100*m['mean']:+.2f} pp [{100*m['ci95'][0]:+.2f}, {100*m['ci95'][1]:+.2f}]"
def main():
 s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'formal_data_audit.json').read_text());g=s['groups'];keys=list(g)
 winner=max(keys,key=lambda k:g[k]['final_success']['mean']);evidence=[r for r in s['efficacy_contrasts'] if r['metrics']['final_success']['ci95'][0]>0]
 anchor_effect=[r for r in s['anchor_contrasts'] if r['metrics']['final_success']['ci95'][0]>0]
 lines=['# Phase 12：Environment State Feedback', '',f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", '',
 '## 1. 范围与结果摘要','',
 '本阶段验证部署时可读取的环境状态反馈；Actor 与 Critic 使用相同的状态向量。没有重新使用训练专用 Privileged Critic，没有 LWD/DIVL/QAM/Quality Replay。', '',
 f"50 次正式训练均完成 300,000 新增交互，共 {s['training']['steps']:,}。5 个匹配 seed 是统计单位。独立复测 {s['heldout']['episodes']:,} episodes；复测交互 {s['heldout']['interactions']:,}。",
 f"按随机 Final 均值，描述性最高组为 **{winner}：{pct(g[winner]['final_success']['mean'])}**；这不是新的独立确认试验。",
 f"环境反馈相对 A 的 8 个候选比较中，Final 配对 t95 CI 下界大于零的数量：{len(evidence)}。是否允许扩至500k：**{s['extension_gate']['passed']}**。", '',
 '## 2. 固定配方、观测接口与随机化','',
 '| 项目 | 设置 |','|---|---|',
 '| 仿真/任务 | 既有 Isaac Lab / Panda Door；同一 USD、控制与成功阈值（门角 >1 rad） |',
 '| 初始化 | 同 seed Phase9 C final，完整网络、target、Adam、alpha；新增输入权重/Adam列置零，初始动作与Q保持一致 |',
 '| Anchor | 原 Phase8 B best，冻结；强1、中0.1、弱0.01、无0 |',
 '| 训练 | 32环境，256 batch，每vector step4次update，LR3e-4；每run37,500 updates |',
 '| 经验/BC | 原1000条expert、271,562 transitions只读；uniform expert/online各50%，另256条expert BC，BC系数10 |',
 '| std/保护 | 全流程std≤0.01；每10k固定nominal与完整随机分布、原threshold0.10 SafeUpdate；拒绝恢复全部学习状态，buffer保留 |',
 '| 训练分布 | 从首个reset起 door angle U(0°,5°)，工装XYZ各U(-1,+1 cm)，摩擦不变；没有课程升级门槛 |',
 '| 测试分布 | 独立种子；主测试同完整随机分布，Best/Final随机各128 episodes/seed；桶测试每64 episodes/seed |',
 '| AUC/Best | 每10k独立于训练rollout的64episode固定validation曲线，AUC归一化到[0,300k]；Best据validation选，另独立复测 |',
 '| 成本 | 300k是新增训练交互；不含继承的expert/Phase9学习成本，也不含评测交互 |','',
 '| 组 | 维数 | Actor/Critic同一输入 |','|---|---:|---|',
 '| A | 26 | 原robot observation：q9、qd9、EE XYZ3、quat4、episode clock1，含夹爪关节 |',
 '| B | 31 | A + angle(rad)、angular velocity(rad/s)、target(rad)、progress[0,1]、remaining(rad) |',
 '| C | 33 | B + 左/右手指contact二值（过滤接触力>0.5N） |',
 '| D | 36 | C + handle workspace XYZ，(XYZ-expert初始均值)/0.1m；未加orientation |','',
 '进度=(angle-episode初角)/(target-episode初角)，截断[0,1]；remaining=target-angle。工装位置变化是整柜刚体平移，不是改把手形状。把手XYZ部署测量能力仍需真机确认；D可作为可测上限。A保留原26D以免同时更改机器人观测。','',
 'Phase11不是纯固定训练：有随机课程，但19/20组停在Level1（0–2.5°、±0.5cm），只有1组到Level2；本阶段所有训练episode均处于完整Level2。与Phase11比较同时含训练曝光和新增velocity/remaining变化，不能把历史差异全部归因于单个输入。','',
 '## 3. 全部方案与五seed统计','',
 '| 方案 | AUC | Independent Best | Independent Final [t95 CI] | Best−Final | nominal Final | 在线成功/完整episode | 拒绝更新均值/30 | Final seed SD | std |',
 '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
 per=json.loads((OUT/'per_seed.json').read_text())
 for k in keys:
  m=g[k];selected=[r for r in per if r['variant']==k];successes=sum(r['online_successes'] for r in selected);episodes=sum(r['online_episodes'] for r in selected)
  ci=m['final_success']['ci95']
  lines.append(f"| {k} | {m['auc']['mean']:.4f} | {pct(m['best_success']['mean'])} | {pct(m['final_success']['mean'])} [{pct(ci[0])}, {pct(ci[1])}] | {100*m['gap']['mean']:+.2f} pp | {pct(m['nominal_final_success']['mean'])} | {successes}/{episodes} | {m['rollback_events']['mean']:.1f} | {100*m['final_success']['sd']:.2f} pp | {m['effective_std']['mean']:.6f} |")
 prior=json.loads((ROOT/'results/phase11_parameter_generalization/summary.json').read_text())
 lines+=['','### 与Phase11的历史比较','', '| 组（强Anchor） | Phase11随机Final | Phase12随机Final | 均值变化 |','|---|---:|---:|---:|']
 for a in ('A','B','C','D'):
  old=prior['groups'][a]['final_success']['mean'];current=g[a+'_strong']['final_success']['mean']
  lines.append(f'| {a} | {pct(old)} | {pct(current)} | {100*(current-old):+.2f} pp |')
 lines+=['','历史比较同时改变了完整随机训练曝光和B/C/D的velocity/remaining输入，不能用它单独识别输入效益；本阶段内部配对比较才隔离新增反馈组合。','',
 '在线成功比例有两种汇总：表中的总成功/总episode是pooled计数；CSV还提供先计算每seed比例再平均的统计。Best由validation选择，独立Best可能低于独立Final，因此gap允许负值。CI不裁剪[0,1]，小样本t假设需谨慎。','',
 '### 配对比较','', '| 比较 | Final差值 t95 CI | Holm t-p | exact sign-flip p | AUC差值 t95 CI |','|---|---|---:|---:|---|']
 for family in ('core_contrasts','anchor_contrasts','efficacy_contrasts'):
  for r in s[family]:
   f=r['metrics']['final_success'];a=r['metrics']['auc']
   lines.append(f"| {r['contrast']} ({family}) | {effect(f)} | {f['holm_paired_t_p']:.4f} | {f['exact_signflip_p']:.4f} | {a['mean']:+.4f} [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}] |")
 lines+=['','五seed的双侧exact sign-flip最小p=0.0625，无法在0.05水平作非参数显著性确认。Holm按预设core、anchor、efficacy三个比较族分别校正；500k门槛是明确标注的模型假设下探索性门槛。episode数量不能替代seed数量。','',
 '## 4. 随机环境分桶','',
 '表中每项均为独立Final的5seed均值。angle分桶固定/限定初角、仍随机XYZ；position分桶固定/限定XYZ、仍随机0–5°初角。nominal同时为零。±0.5/±1cm是三个坐标各自均匀范围（嵌套立方体），另给L∞壳层避免把范围误当单一固定偏移。','',
 '| 方案 | 角0° | 角0–2.5° | 角2.5–5° | 位置0 | XYZ±0.5cm | XYZ±1cm | L∞0.25–0.5cm | L∞0.5–1cm |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 conds=['angle_zero','angle_low','angle_high','position_zero','position_halfcm','position_onecm','position_shell_halfcm','position_shell_onecm']
 for k in keys:lines.append('| '+k+' | '+' | '.join(pct(s['generalization'][k]['final'][c]['policy']['mean']) for c in conds)+' |')
 lines+=['','Best的所有分桶、每seed结果及CI保存在summary.json/generalization；没有用分桶复测重新选择checkpoint。','',
 '## 5. 参数利用：动作敏感性与隐藏输入','',
 '固定1024条按预设transition索引采样的robot state，改变门角0°→5°，分别做angle-only与关联progress/remaining/velocity一致的reset标量干预；记录归一化动作L2差值。不同方法的在线访问状态可能不同；该统计不能证明动作变化正确。','',
 '| 方案 | angle-only L2均值 | coherent-reset L2均值 | 新输入权重norm均值 |','|---|---:|---:|---:|']
 for k in keys:
  probes=[s['sensitivity'][k][str(seed)]['final'] for seed in range(5)]
  lines.append(f"| {k} | {np.mean([p['angle_only_pair_l2_mean'] for p in probes]):.6g} | {np.mean([p['coherent_reset_pair_l2_mean'] for p in probes]):.6g} | {np.mean([p.get('added_actor_weight_norm',0) for p in probes]):.6g} |")
 lines+=['','所有反馈组的Best/Final均实际执行输入隐藏复测；不以D是否显著为前置条件。下表正值=隐藏后成功率下降，负值=隐藏后改善。每个干预与intact使用同seed/clone/episode随机计划。','',
 '| 方案 | 隐藏输入 | masked Final | intact−masked t95 CI |','|---|---|---:|---|']
 for k in keys:
  for mask,m in s['input_masking'][k]['final'].items():lines.append(f"| {k} | {mask} | {pct(m['masked_success']['mean'])} | {effect(m['drop'])} |")
 lines+=['','注意：遮罩区间为探索性的未校正配对t95 CI，不据多项遮罩的单独正区间宣布显著。它们是测试时输入干预，没有额外重训“去掉通道”模型。angle、progress、remaining互相编码相关状态，单通道无下降不能证明未使用；angle-family遮罩同时设angle/velocity/progress=0、remaining=target。all-feedback另把contact=0和handle设expert均值。遮罩可能产生训练分布外输入，下降证明依赖的证据强于证明该传感器必不可少；要独立因果分辨需随后做预注册重训消融。target恒定，本阶段无法验证不同target的条件控制。','',
 '## 6. 六个交付问题','',
 '### 1）环境状态反馈是否提升泛化？','',
 ('有候选Final配对区间正向，但须结合Holm、exact小样本限制、AUC、保护次数及独立泛化桶一起判断。详见比较表，不能把最高seed或checkpoint作为总体结论。' if evidence else '在本次预设配方与预算下，未获得Final配对t95 CI下界为正的反馈优势；不能声称状态反馈已改善随机泛化。这不证明传感器在其他学习配方下无效。'),'',
 '### 2）哪些参数最有效？','',
 'B−A、C−B、D−C分别检验进度状态组合、接触增量、把手位置增量。动作敏感性与mask表给出实际依赖；单项冗余输入的遮罩不等同于重训增量效益。只在成功率增益、稳定性与依赖诊断一致时才推荐部署输入，不依据权重非零宣称有效。','',
 '### 3）为什么Phase11输入无效？','',
 '已确认的限制是完整随机训练曝光不足，原expert只覆盖标称初态，且低std下强Anchor可能把反馈条件动作拉回固定工位策略。Phase12消除了课程卡住，新增angular velocity/remaining并直接测试Anchor强度。历史差异不能唯一归因于观测不足；结合本阶段完整随机A与各反馈组才能区分曝光、约束和状态利用。','',
 '### 4）Anchor是否限制适应？','',
 f"相对强Anchor，减弱/取消Anchor的6个Final配对比较中，有 {len(anchor_effect)} 个t95 CI下界大于零。应结合nominal保持、随机AUC和拒绝率；被guard频繁拒绝时，最终接近原策略并不证明弱Anchor学习本身稳定。",'',
 '### 5）策略是否真正利用输入？','',
 '动作0°/5°对比、单项及关联遮罩均已执行。非零动作变化证明局部函数依赖；隐藏后下降进一步支持执行依赖。两者都不能单独证明适当闭环调整。若反馈组没有独立成功增益，即便输入权重/动作差异非零，也不能宣称学会了有效环境适应。','',
 '### 6）是否满足进入LWD/DIVL？','',
 '仅当反馈相对robot-only有效、完整随机分布持续产生成功/失败经验、轨迹差异可用、动作依状态调整且少量rollback便可保持Final≈Best时才应进入。大量rollback保护得到小gap，属于更新被拒绝，不等价于稳定在线改进；成功/失败两个峰也不自动证明quality replay有效。当前门槛与证据见extension_gate.json；不以固定工位98%替代随机泛化。','',
 '## 7. 成本、保护与完整性','',
 f"训练中评测交互合计 {sum(r['eval_interactions'] for r in per):,}；独立评测 {s['heldout']['interactions']:,}；物理预检64。评测不进入训练buffer。各组random/nominal candidate、拒绝理由、KL与std在updates CSV；raw被拒绝checkpoint亦保留。",
 f"历史文件审计 {audit['prior_artifacts']:,} 项，变更列表 {audit['prior_changed']}；训练源码变更 {audit['training_source_changes']}。50个HDF5完成轨迹/terminal/观测通道审计通过；已完成episode {audit['total_completed_episodes']:,}，成功 {audit['total_successes']:,}。",
 f"训练初态流逐episode匹配 {audit['matched_training_episode_parameters']:,} 项；独立复测条件计划匹配 {s['heldout_parameter_plan_comparisons']:,} 次。保存pending未完成轨迹，不能把其状态伪装成完整成功/失败。",
 'Python/Isaac/GPU与依赖沿用既有报告和configs/runtime_env.sh；精确代码hash和原expert/source/anchor hash在protocol、training_source_manifest和baseline_inventory。GPU队列只对本项目的PID负责，其他项目未停止或修改。','',
 '## 8. 图与可复查产物','',
 '![Learning curves](../results/phase12_environment_state_feedback/figures/learning_curves.png)','',
 '![Endpoints and rollback](../results/phase12_environment_state_feedback/figures/endpoints_and_rollback.png)','',
 '![Buckets](../results/phase12_environment_state_feedback/figures/generalization_buckets.png)','',
 '![Input masking](../results/phase12_environment_state_feedback/figures/input_masking.png)','',
 '- results/phase12_environment_state_feedback/per_seed.csv：全部50run指标；summary.json：全部CI、配对检验、分桶、遮罩、敏感性。',
 '- checkpoints/phase12_environment_state_feedback/：Best、每10k accepted/raw、完整学习状态。',
 '- datasets/phase12_environment_state_feedback/：50个HDF5，实际随机参数、transition、terminal state、episode索引。',
 '- logs/phase12_environment_state_feedback/：训练、持久评测进程日志、TensorBoard；results中另有queue、source与历史审计。','']
 path=ROOT/'docs/phase12_environment_state_feedback_report.md';path.write_text(chr(10).join(lines),encoding='utf-8')
 print(path,flush=True)
if __name__=='__main__':main()
