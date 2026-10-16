"""Evidence-backed working report; pending experiments remain pending."""
from pathlib import Path
import json,csv,collections,hashlib,time
import numpy as np
R=Path.cwd();O=R/'phase15_3/results/retrospective';P=R/'phase15_2/results'
audit=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text());assert audit['passed']
agg=list(csv.DictReader((O/'stage_aggregate.csv').open()));episodes=list(csv.DictReader((P/'metrics/funnel_v2_episode.csv').open()))
mechanism=json.loads((O/'mechanism_scope.json').read_text())['E_summary']
attempts=list(csv.DictReader((O/'planner_attempts.csv').open()))
formal=list(csv.DictReader((P/'metrics/aggregate.csv').open()))
formal_tests=['fixed','random','artificial_macro','natural_severe','full_loop','grade_05','grade_10','grade_15','grade_20']
formal_lines=['|方法|固定|随机|人工宏平均|自然|Full Loop|grade5|grade10|grade15|grade20|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for method in 'ABCDE':
 values=[next(x for x in formal if x['method']==method and x['test']==t) for t in formal_tests]
 formal_lines.append('|'+method+'|'+'|'.join(f"{float(x['mean'])*100:.4f}" for x in values)+'|')
formal_table='\n'.join(formal_lines)
core_lines=['|原正式E端点|五seed均值±SD|t95%CI|门槛（均值/CI下限）|','|---|---:|---:|---|']
for endpoint,threshold in [('fixed',.95),('random',.90),('artificial_macro',.85),('natural_severe',.50),('full_loop',.90)]:
 x=next(x for x in formal if x['method']=='E' and x['test']==endpoint)
 # Use the original aggregate schema; no new confidence interval is inferred.
 low=float(x['low']);high=float(x['high']);mean=float(x['mean'])
 pass_mean=mean>threshold if endpoint=='natural_severe' else mean>=threshold
 pass_low=low>threshold if endpoint=='natural_severe' else low>=threshold
 core_lines.append(f"|{endpoint}|{mean*100:.4f}±{float(x['sd'])*100:.4f}%|[{low*100:.4f}, {high*100:.4f}]%|{'通过' if pass_mean else '未通过'}/{'通过' if pass_low else '未通过'}|")
core_table='\n'.join(core_lines)
experimental=[]
source_summary=R/'phase15_3/diagnostics/cohorts/development/seed1/natural_severe.json'
if source_summary.exists():
 s=json.loads(source_summary.read_text())
 experimental.append(f"新自然开发集：seed1/reset153301001，已保存{s['saved']}个真实失败状态，来源执行{s['completed_source_episodes']}回合、其中失败{s['source_failed_episodes']}，实际交互{s['interactions']} clone-tick。源SHA `{s['cohort_sha256']}`。不含专家动作监督；与旧validation/test身份无重叠。源过程interaction包括未入选筛查回合，不是16个样本的恢复训练成本。")
confirmation_source=R/'phase15_3/diagnostics/cohorts/confirmation/seed1/natural_severe.json'
if confirmation_source.exists():
 c=json.loads(confirmation_source.read_text())
 experimental.append(f"未见自然确认集：seed1/reset153401001，已保存{c['saved']}个真实失败状态；筛查{c['completed_source_episodes']}回合、实际失败{c['source_failed_episodes']}，交互{c['interactions']} clone-tick。源SHA `{c['cohort_sha256']}`。来源reset在观察候选结果之前锁定，与开发集及旧validation/test身份无重叠，不保存训练标签。")
analysis=R/'phase15_3/results/controlled_analysis/analysis_scope.json'
if analysis.exists():
 s=json.loads(analysis.read_text());experimental += ['','|已闭合case|状态数|最终开门|总预算|','|---|---:|---:|---:|']
 for k,v in s['closed_cases'].items():experimental.append(f"|{k}|{v['episodes']}|{v['successes']}|{v['budget']}|")
 experimental.append('所有新case仅1个model seed，统计为探索性状态比例和配对状态差异；不是五model seed方差/CI。尚未完成的case不计为失败或成功。')
 pairing=json.loads((analysis.parent/'initial_pairing.json').read_text())
 experimental.append('已闭合case的actual initial机器人/GT数组、接触和时钟逐bit比较：'+('; '.join(x['baseline']+'→'+x['candidate']+(' passed' if x['passed'] else '未通过') for x in pairing['pairs']) or '待配对case完成')+'。')
 if 'time900' in s['closed_cases']:
  comparisons=json.loads((analysis.parent/'time_trajectory_comparison.json').read_text())['comparisons']
  verified=all(x['key_sets_equal'] and all(v==0 for v in x['max_absolute_difference_by_field'].values()) for x in comparisons)
  assert verified, 'Do not claim identical trajectories without current exact checks'
  extra=(f"1200tick也为{s['closed_cases']['time1200']['successes']}/8，剩余状态仍未开门；延长到1200没有新增成功。" if 'time1200' in s['closed_cases'] else '1200结果待闭合。')
  experimental.append('900tick诊断为7/8开门，成功tick618–717；600复现0/8。已闭合预算组初始机器人/GT状态逐bit一致，前600tick的2999条动作、关节、末端、接触和门角记录均逐值一致。对这7状态，有直接同轨迹反事实证据表明600总预算不足；不能外推全部正式500状态。'+extra+'状态配对exact McNemar仅为探索性状态检验，不能替代五模型seed确认。')
 stage_metrics=list(csv.DictReader((analysis.parent/'stage_metrics.csv').open()))
 experimental += ['### 已完成新开发/确认集：同状态600tick阶段覆盖','|case|N|安全位置|Pre-grasp到达|有效抓取|Normal交还|进度恢复|最终开门|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for case in s['closed_cases']:
  if case=='probe600' or case.startswith('time'):continue
  rr={x['stage']:int(x['coverage_n']) for x in stage_metrics if x['case']==case and x['distance_bin']=='-1' and x['mode']=='raw'}
  experimental.append('|'+case+'|'+str(rr['source'])+'|'+'|'.join(str(rr[k]) for k in ['safe_pose','pregrasp_arrival','valid_grasp','handoff','door_progress','final_success'])+'|')
 experimental.append('这里各列是独立回合覆盖；完整条件分母、Wilson状态区间、距离分桶及raw/ordered定义见`controlled_analysis/stage_metrics.csv`。新实验均单模型seed，五seed方差和CI保持unknown。')
 paired=list(csv.DictReader((analysis.parent/'paired_exploratory.csv').open()))
 experimental += ['### 探索性配对状态统计','|baseline→candidate|端点|赢/输|精确McNemar p|同预算|','|---|---|---:|---:|---|']
 for x in paired:
  experimental.append('|'+x['baseline']+'→'+x['candidate']+'|'+x['endpoint']+'|'+x['candidate_only']+'/'+x['baseline_only']+'|'+x['exploratory_exact_mcnemar_p']+'|'+x['total_budget_same']+'|')
 experimental.append('上述多端点状态检验未作确证性多重校正，样本为一个model seed；不据此宣称五seed显著增益。不同总预算的时间诊断不作为600tick算法改善。')
handoff_path=R/'phase15_3/results/handoff_audit/scope.json'
trajectory_path=R/'phase15_3/results/trajectory_audit/scope.json'
if trajectory_path.exists():
 tr=json.loads(trajectory_path.read_text());experimental.append('独立初始数组字节核验：'+('已完成配对全部dtype及字节一致。' if tr['all_initial_arrays_byte_exact'] else '存在不一致，尚不能解释为公平配对。'))
 for v in tr['comparisons']:
  if v['candidate'].endswith('GT_projected_retreat') and v['key_sets_equal'] and all(x==0 for x in v['maximum_numeric_difference'].values()):
   experimental.append(f"{v['baseline']}→{v['candidate']}的{v['shared_records']}条完整动作/关节/末端/接触/门角记录逐数值一致。因此该投影后检查候选在本cohort没有改变实际行为，也没有闭环收益；它不能证伪其他自然状态中可能存在的投影前后不一致。")
if handoff_path.exists():
 hs=json.loads(handoff_path.read_text())['summaries']
 experimental += ['### 逐步Handoff与时间核验','|case|实际交还|严格探测→过渡→Normal|直接GT过渡|含Recovery桥接|前15步失去双指接触/完整观察|源后活动tick均值|初次Normal剩余tick均值|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for case,v in hs.items():
  experimental.append(f"|{case}|{v['normal_entry_episodes']}|{v['observed_strict_first_handoff']}|{v['direct_planner_transition_normal']}|{v['observed_recovery_bridge_episodes']}|{v['first15_contact_loss']}/{v['first15_fully_observed']}|{v['mean_active_sim_ticks']:.2f}|{v['mean_first_normal_remaining'] if v['mean_first_normal_remaining'] is not None else 'unknown'}|")
 experimental.append('严格序列以每步实际接触、12次连续probe/ready观察、15tick TRANSITION、Normal动作及绝对把手距离<60mm核验。Recovery桥接单列，不能把桥接动作当作纯GT专家标签。前15步没有接触丢失不证明长期抓取稳定。活动tick为源状态后到终点的仿真时间，原总预算仍含源前缀；batch32共享CPU墙钟不是逐回合实测耗时。逐回合及交互成本见`handoff_audit/per_episode.csv`与`scope.json`。')
for name in ['controlled_dev16_seed1_v1','time600_900_1200_seed1_first8_v1','new_natural_confirmation_seed1','confirmation_seed1_v1']:
 d=R/'phase15_3/experiments'/name
 if (d/'lifecycle.json').exists():
  x=json.loads((d/'lifecycle.json').read_text());experimental.append(f"`{name}`：{x['status']}；完整闭合marker={(d/'complete.json').exists()}。" )
experimental += ['','首个8状态探针中7个GT恢复尝试均完成safe/pregrasp/valid/handoff，最终开门0/8；第8状态原Router走Recovery而没有GT尝试。7个GT回合终点仍双指接触，门角约0.45–0.90rad，最后交还仅剩47–171tick。它不能代表正式500自然分布，也不能据此替代正式F2/F7瓶颈。',
 '新开发集四臂：原冻结E；GT直接路由；GT直接路由+投影后撤离候选检查；GT直接路由+原checked waypoint末端位姿反馈。各16同源状态、600tick。两个几何候选与GT直接路由比较，路由干预另与原E比较；不将两种变化混成单因素收益。候选仅构造/边界保留gate已过，闭环收益等待真实结果。',
 '时间诊断使用另一次600复现及900/1200协议，各8相同旧validation状态。330tick局部上限保留，但其可用时间仍为min(330,总预算−触发tick−1)，Normal入口要求至少45剩余tick；因此总预算扩展可以在600之前改变重试/入口可用性。模型RPC时钟仍clip(elapsed/600,0,1)，控制器计时用实际elapsed。不是仅取消600处终止，也不是600tick算法增益。']
summary_path=R/'phase15_3/results/experiment_summary.csv'
if summary_path.exists():
 rows=list(csv.DictReader(summary_path.open()))
 experimental+=['### 新实验最终开门、区间与耗时','|case|开门/N|比例|探索性状态Wilson95%CI|源后活动tick均值|总终止tick均值|','|---|---:|---:|---:|---:|---:|']
 for x in rows:
  experimental.append(f"|{x['case']}|{x['final_successes']}/{x['episodes']}|{float(x['final_rate']):.2%}|[{float(x['exploratory_state_wilson95_low']):.2%}, {float(x['exploratory_state_wilson95_high']):.2%}]|{float(x['mean_active_sim_ticks']):.2f}|{float(x['mean_original_terminal_ticks']):.2f}|")
 experimental.append('这里的Wilson区间只描述这一个模型seed的入选状态比例，不估计模型seed间泛化方差。原正式五seed方差及t区间另见历史aggregate；新实验五seed方差、均值t区间均为unknown。逐回合物理墙钟未单独测量，不能将共享batch32控制时间除以32冒称逐回合实测。')
expert_path=R/'phase15_3/results/expert_usability_audit.json'
expert=json.loads(expert_path.read_text()) if expert_path.exists() else None
complete_analysis=json.loads(analysis.read_text()) if analysis.exists() else {}
all_closed=all(complete_analysis.get(k,False) for k in ['controlled_complete','time_complete','confirmation_complete'])
completed_scope=all_closed and expert is not None
status=('本轮单seed受控诊断及未见确认已闭合；严重恢复算法目标与Anchor门槛尚未达成。' if completed_scope else '进行中，尚不是最终算法验收报告。')
question6='锁定候选的未见确认尚未全部闭合，不能声称提升。'
question3='子任务边界有用，学习策略必要性尚未证明。先验证GT专家；不预设学习优于规划。'
execution_status='最终验收还需要真实新诊断结果、证据选择的候选验证、独立确认与完整统计，当前不宣称完成。'
if expert:
 selected=expert['selected'];dev=expert['development_handoff'];conf=expert['confirmation_handoff'];gate=expert['gate_13_of16_each']
 experimental.append(f"锁定专家候选 `{selected}`：开发集严格交还{dev}/16、未见确认{conf}/16；直接GT过渡分别{expert['pure_planner_direct_dev']}/16与{expert['pure_planner_direct_confirm']}/16。预先声明的探索性可用性门槛是两集各≥13/16且无观察到的关节越界，该门槛{'通过' if gate else '未通过'}。80%左右为本轮设计判断，不是用户给定的Anchor门槛；严格接管成功也不等于最终开门。")
 if not gate:
  question3='Severe Re-acquire是合适的子任务定义，但当前专家未通过两集各≥13/16的预声明可用性门槛；这支持继续修GT几何与执行，不能据此宣称学习策略必要或优于规划。没有采集专家训练标签。'
  execution_status='本轮预声明的四臂开发、600/900/1200时间诊断及锁定候选未见确认均已闭合。交付范围为这轮单seed探索性诊断与候选验证；专家门槛未过，因此条件性BC/π0.5训练和输入GT消融未启动，严重恢复能力修复及可靠Anchor仍未完成。'
 if completed_scope:
  final={x['case']:x for x in rows}
  base=final['confirm_frozen_E'];candidate=final['confirm_'+selected]
  question6=f"同一未见确认集、同600tick，冻结E开门{base['final_successes']}/16，锁定{selected}为{candidate['final_successes']}/16；对应探索性配对状态检验见下表。本轮仅一个模型seed，也未在原正式test上确认，因此没有五seed显著优于原正式对照的证据。"
if all_closed:
 experimental=[x.replace('候选仅构造/边界保留gate已过，闭环收益等待真实结果。','候选构造/边界保留gate只用于接入；真实闭环数据已齐全，未将该gate当成算法有效性证明。') for x in experimental]
 experimental.append('位姿反馈候选开发集最终开门3/16，但Pre-grasp到达、有效抓取、严格交还均6/16，低于GT直接路由的11/16、10/16、10/16。已成功交还的子集首次接管平均剩余207.17tick，GT直接路由为144tick；这是不同成功子集的描述，不能解释为全部状态的配对提速。预声明目标是恢复到Normal可接管状态，因此没有事后按开门最高挑选该候选，也未用确认结果重新选方法。')
 experimental.append('后续修复证据仍指向几何/动作可执行性与仿真恢复时间：在新开发状态中分开检查模型穿透退出、checked path存在但tracking拒绝，以及晚触发后的有效预算。保留600tick正式门槛，时间扩展只做诊断；不放宽碰撞、关节或60mm入口，不将当前单seed少量开门冒称严重恢复已解决。此后若改变候选或数据选择，必须另锁开发协议并使用新的未见确认源，不能继续将本轮确认集称为未见。')
experimental.append('新执行manifest在开发运行中、选择及确认之前保存，覆盖13个执行输入和两份新cohort；它提供从该时点起的字节冻结证据，不能冒称所有源码均有启动前独立验签。所有独立后处理source版本、物理失败接入归档、输入SHA、checkpoint引用及退出证据均保留。')
experimental_text=''
for i,paragraph in enumerate(experimental):
 separator='\n' if i and paragraph.startswith('|') and experimental[i-1].startswith('|') else '\n\n'
 experimental_text+=(separator if i else '')+paragraph
temporal=''
temporal_path=O/'temporal_scope.json'
if temporal_path.exists():
 tt=json.loads(temporal_path.read_text())
 temporal=f"正式自然500状态的源年龄中位数{tt['source_age_median']}tick，源剩余预算中位数{tt['source_available_ticks_median']}tick。128个实际Normal入口回合，从源到第一次接管中位数{tt['source_to_first_normal_median']}tick，第一次接管剩余中位数{tt['remaining_at_first_normal_median']}tick，38个不足120tick；最后一次入口剩余中位数{tt['remaining_at_last_normal_median']}tick。这里只是观察性分层，不证明全部可由延长预算修复。每seed/每回合/每尝试见`temporal_*.csv`。**CPU规划墙钟慢与仿真600tick是不同时间量：墙钟优化不会直接增加每回合可用tick。**"
labels=['源自然严重状态','触发阈值证据（proxy）','安全撤离候选','实际安全位置','Pre-grasp规划','实际到达Pre-grasp','重新接近证据','双指重新接触','有效抓取','交还Normal','Normal接管确认','持续门角增长','最终开门']
stages=['source','trigger_proxy','retreat_candidate','safe_pose','pregrasp_plan','pregrasp_arrival','reapproach','recontact','valid_grasp','handoff','normal_takeover','door_progress','final_success']
def pct(v):return f'{float(v)*100:.2f}%' if v not in ['',None] else 'unknown'
def table(mode):
 text=['|阶段|进入数量|共同成功数量|条件成功率|本阶段独立覆盖/500|累计覆盖率|','|---|---:|---:|---:|---:|---:|']
 for stage,label in zip(stages,labels):
  a=next(x for x in agg if x['method']=='E' and x['population']=='natural' and x['distance_bin']=='-1' and x['mode']==mode and x['stage']==stage)
  text.append(f"|{label}|{a['entered_n']}|{a['success_n']}|{pct(a['conditional_rate_pooled'])}|{a['coverage_n']}|{pct(a['cumulative_rate_pooled'])}|")
 return '\n'.join(text)
e=[x for x in episodes if x['method']=='E' and x['population']=='natural'];counts=collections.Counter(x['primary'] for x in e)
groups=[('F'+str(i),counts['F'+str(i)]) for i in range(1,16)]
tax='\n'.join(['|末次尝试主类|数量/486失败|比例|','|---|---:|---:|']+[f'|{k}|{n}|{n/486:.2%}|' for k,n in groups])
graded=[]
for b,name in enumerate(['[0,5)','[5,10)','[10,15)','[15,20)','[20,+∞)']):
 rr=[x for x in episodes if x['method']=='E' and x['population']=='graded' and int(x['distance_bin'])==b]
 graded.append('|'+name+'|'+str(len(rr))+'|'+'|'.join(str(sum(x[s]=='True' for x in rr)) for s in ['safe_pose','pregrasp_arrival','recontact','valid_grasp','handoff','final_success'])+'|')
report=f'''# Phase15.3：严重自然偏离恢复诊断与专项验证

更新时间：{time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())}。
状态：**{status}** 下面的正式科学数据来自已完成并全量SHA核验的Phase15.2；Phase15.3物理诊断、候选修复和独立确认的状态在最后一节。没有训练新策略，没有启动在线RL。

## 1. 历史基线与隔离

本次逐字节核验通过：2435份最终交付文件、2350份正式原始文件、187份有效冻结源码、10个冻结checkpoint、90份cohort、26份来源文件。300项正式测试、30500回合、A–E各五model seed齐全。自然validation160与test500按model seed/reset seed/原clone/source episode核验唯一，交集0。原审计passed、canonical镜像字节一致。详见`phase15_3/results/phase15_2_baseline_audit.json`，包含每个文件SHA。

正式自然test实际距离0.100094–0.238286m；5–10cm没有自然证据。人工grade使用源H5实测距离分桶。Phase15.1的35.25%与本轮2.8%来自不同cohort，不能作算法增益或退化比较。Phase1–15.2的源码、模型、正式数据与结果不写入。

### 原正式A–E五seed均值（百分比）

{formal_table}

这是同一正式cohort上的原对照。各端点五seed SD、t95%CI见原`phase15_2/results/metrics/aggregate.csv`；13项完整表包括人工四子项，见原最终recovery_chain/latency报告。原25 paired、补充45、阶段7端点各自Holm族独立，不能混合挑选显著端点。

{core_table}

## 2. E自然500回合阶段表

“进入”为前一里程碑被观察到的回合；“共同成功”为前一与当前里程碑均被观察到的回合，条件率=共同成功/已知进入；独立覆盖单独报告。这些是跨多次尝试的raw观察，可能绕过阶段，**不是单次连续因果链**。proxy是保存的阈值违例证据，不能解释为真正误触发率。累计列为各阶段覆盖/source500，不是条件率连乘。

{table('raw')}

例如最终成功14个中有13个记录了持续门角增长，因此最后行条件分子13与最终覆盖14不同；不能简单做计数相减。五seed分别统计的均值、SD和t95%CI保存在`stage_per_seed.csv`、`stage_aggregate.csv`。条件率的五seed均值与上表按分母汇总率分别保存，不能混同。

### 距离断崖（E人工graded，raw数量）

|源H5实测距离cm|源数量|安全位置|Pre-grasp到达|接触|有效抓取|Handoff|最终开门|
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(graded)}

5cm至10cm处安全位置均500/500，损失随后同时发生在到达、接触、有效抓取与交还；不是单一抓取分类器问题。该人工诊断不能填补低距自然测试缺口。

## 3. 损失最大的阶段与具体机制证据

E自然raw集合损失：撤离候选→实际安全位置122；Pre-grasp规划→到达114；有效抓取→交还29；持续增长→最终开门106（包含跨阶段绕过，所以119−14=105并非集合损失）。末次尝试互斥主类F2=122/486=25.10%，F7=115/486=23.66%，F15=117/486=24.07%。F7末次尝试与raw114定义不同，不能互换。

**最大的已归类恢复执行损失在撤离候选到实际安全位置；Pre-grasp到达是第二个证据充分的重点。** F15仍占近四分之一，不能宣布全部失败只有一个原因。

新增对25组自然正式H5与telemetry的只读分析，按末次尝试触发tick筛选：

|E末次失败类|回合|有采样几何模型拒绝|采样arm关节余量<0|采样最小目标位置误差中位数|末次触发剩余tick中位数|
|---|---:|---:|---:|---:|---:|
|F2|122|{mechanism['F2']['episodes_any_sampled_model_reject']}|0|42.02mm|343|
|F7|115|{mechanism['F7']['episodes_any_sampled_model_reject']}|0|26.08mm|259|

F2最后一次尝试中109/122以`time_budget`结束，13个为`safe_retreat_no_monotone_escape`；其法向运动最大值中位数仅8.63mm，小于原25mm撤离到达条件。反复存在候选却未退出模型穿透，不证明真实物理碰撞。F7的115个最后尝试全部以`time_budget`结束。没有采样负关节余量不证明连续轨迹无越界。

旧H5仅每10tick/阶段变化/terminal采样，目标可随waypoint改变；上述“最小目标误差”不是直接规划动作—物理执行残差。真实arm/wrist碰撞未传感，记unknown；有限IK失败不等于数学不可达。

### 需要反事实验证的机制

1. 撤离：候选按投影前预测段选择，实际动作又经过原`project`并被tracking模型检查，需记录投影前后候选、穿透单调性与实际移动；不放宽碰撞或关节边界。
2. Pre-grasp：七关节目标经六维DLS加权拟合，再旋转裁剪与投影；需检查冗余关节路径的不可执行分量、末端位姿误差反馈和真实速度。{('末端位姿反馈已实际验证，但开发集到达/抓取/交还覆盖下降；单独替换该反馈不足以修复整体恢复。' if all_closed else '当前只是机制假设，候选闭环结果尚未齐全。')}
3. 时间：600全回合上限之外，原恢复流程还有330tick局部上限。终止于600不能证明“只差时间”。600/900/1200需作为新协议，保留原动作特征语义并公开600之后的分布外输入处理；局部上限是否改动必须独立标记。
4. 抓取后：E的128个Handoff中119个恢复增长，仅14最终开门，23个5秒内Normal再失败事件（完全观察23、删失107、成功竞争13）；不能只优化抓取后假设Normal必然完成。F12末次主类22与全事件23统计作用域不同。

{temporal}

## 4. 末次尝试互斥归类

{tax}

主类按Phase15.2审计的F1–F15固定定义，逐回合只计一次；joint/model reject、预算耗尽、重复失败等为可重叠次级证据。全部自然失败E486均最终耗尽600tick，显式time_budget次级标记240；这与互斥F14=0并不矛盾。类别详见原`docs/phase15_2_failure_taxonomy.md`与本阶段`exclusive_failures.csv`、`planner_attempts.csv`、`mechanism_per_episode.csv`。

## 5. 受控验证及训练门槛

首个工程探针：冻结E/seed1，自然validation按编号first8，不按结果选择。复制源prefix/reset/handoff数据，32clone接口中只评分8个；600tick、原600归一化、严格60mm、真实15预测suffix、原geometry与投影保持。记录每步q/qd/EEF/contact/action/目标路径/局部elapsed，状态分别标注动作前与动作后。它是**复用validation的单seed探索性工程诊断**，不冒称新独立开发集或正式改善。

候选调参必须使用另行固定的新自然开发集；原正式test不能参与调参。候选在开发集锁定后用未见确认集；源状态、prefix、门/机器人/把手、随机参数、contact warmup、噪声、模型与预算必须匹配，并比较真实initial文件，不能只核对cohort名称。不能保证PhysX缓存接触历史逐bit相同，须报告重放及初始接触核验。

若主要是安全撤离，先修GT几何/执行，暂不训练。若到达/抓取仍是瓶颈，任务定义Severe Re-acquire：只达到Normal可处理状态，不负责整段开门。先验证真实自然开发状态上的专家闭环；专家不可用时继续修规划，不采新动作标签、不训练BC/π0.5。专家可用后单seed链路检查→3seed小训练→稳定趋势后5seed。相同数据与预算比较GT+IK、轻量BC、π0.5及机器人输入/机器人+GT，不能用遮挡推断GT全部收益。

## 6. 必答十问（当前证据状态）

1. **Phase15.2主要失败原因？** 撤离实际安全位置与Pre-grasp实际到达是两个最大的已定位执行瓶颈；末次F2最大122，F7为115，F15为117尚有未细分失败。
2. **规划、抓取、Handoff还是时间？** 正式分布是多阶段问题。新增8状态的600/900反事实显示7个状态前600轨迹相同，900中最终开门，对这7个状态有时间不足的直接证据；严重状态的撤离/到达失败仍未据此排除。模型拒绝不是实测碰撞。
3. **需要独立Re-acquire策略？** {question3}
4. **轻量BC与π0.5谁更有效？** 本阶段尚未训练或公平比较，unknown。没有专家闭环可用证据时不启动训练。
5. **工装GT贡献？** 本阶段同数据同预算无GT对照尚未完成，unknown；当前规划器使用GT不能单独证明学习输入增益。
6. **显著超过原正式对照？** {question6} Phase15.2五seed双侧exact最小p=.0625，原25、补45与阶段7端点Holm族独立；单seed探索不能代替五seed确认。
7. **可靠随机工位Anchor？** 否。E随机83.125%（目标≥90%），自然2.8%（目标>50%），Full Loop82.34375%（目标≥90%）；固定98.4375%不能代替随机泛化。
8. **允许Phase16在线RL？** 否；随机基础、严重自然恢复、Full Loop及GT/动作接口门槛尚未同时通过。
9. **后续π0.5与Residual SAC如何结合？** 仅未来方案：冻结可靠Anchor，Residual在原动作接口输出受约束修正，保持安全投影；与Frozen Anchor/KL在线RL在完全相同随机测试条件配对比较，记录前后性能、成功轨迹及总交互成本。本阶段不启动。
10. **LWD/DIVL继续暂缓？** 是。先证明在线RL对Anchor有真实稳定增量，再研究LWD/DIVL；规划修复收益不能冒充RL收益。

## 7. 工程与当前执行状态

Phase15.2单机器人batch1模型推理约Normal49ms、Recovery58ms；仿真batch32不是每机器人部署实测。独立CPU288诊断证据表明E撤离完整操作墙钟占自然case采集墙钟约84.03%，多包含串行检查；nested inclusive不能相加，fixture3.396×与2.57%开销不能外推完整闭环。新阶段只用有限CPU亲和性与最多2pair，先1pair；不为填满GPU增加worker。

本报告每次生成时附新阶段状态，不把工程接入失败算作科学失败。首次无GPU未启动与第二次socket隔离接入故障的源码、日志、SHA及本任务SIGTERM/SIGKILL处置provenance已独立留档；没有触碰其他项目进程或历史结果。{execution_status}

{experimental_text}

### Ordered覆盖附表

包含所有13里程碑的累积交集，只是严格覆盖描述，也不证明单次尝试连续因果链。

{table('ordered')}
'''
target=R/'docs/phase15_3_reacquire_report.md';target.parent.mkdir(exist_ok=True)
if target.exists():
 arc=R/'phase15_3/results/report_versions';arc.mkdir(exist_ok=True);old=target.read_bytes();(arc/(hashlib.sha256(old).hexdigest()+'.md')).write_bytes(old)
target.write_text(report,encoding='utf-8')
print('WORKING REPORT',target,'sha256',hashlib.sha256(target.read_bytes()).hexdigest())
