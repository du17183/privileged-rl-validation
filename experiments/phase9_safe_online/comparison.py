"""Project-wide descriptive inventory; no cross-phase causal ranking."""
import json
from pathlib import Path
from experiments.phase9_safe_online.current_metrics import main as snapshot
ROOT=Path(__file__).resolve().parents[2]

def main():
    snapshot()
    data=json.loads((ROOT/'results/phase9_safe_online/current_metrics.json').read_text())
    endpoint_best=max((a for a,r in data['arms'].items() if 'policy_final' in r),key=lambda a:data['arms'][a]['policy_final']['mean'])
    endpoint_rate=data['arms'][endpoint_best]['policy_final']['mean']
    all_finished=all(r['training_complete'] and r['independent_seeds_complete']==5 for r in data['arms'].values())
    # Historical point estimates transcribed from the retained reports linked below.
    # I=independent deterministic endpoints; F=fixed-curve endpoints; P=peak on curve.
    history=[
      ('1 Drawer','A','普通SAC','8×200k',.233,.754,.246,'P/I','GT Critic未显示收益'),
      ('1 Drawer','B','完整GT Critic','8×200k',.213,.961,.270,'P/I','终点与A接近'),
      ('1 Drawer','C','GT Actor+Critic','8×200k',.291,.961,.488,'P/I','Actor依赖GT'),
      ('2 Door','A','原BC+SAC配方','5×500k',.183,.906,.453,'I','后期退化'),
      ('2 Door','B','完整GT Critic','5×500k',.089,.694,0,'I','最终全部失败'),
      ('2 Door','C','GT Actor+Critic','5×500k',.148,.731,.250,'I','未稳定优于A'),
      ('2 Door','D','GT Critic+价值加权','5×500k',.049,.566,.166,'I','未改善'),
      ('3','diag_A','普通Critic诊断重跑','5×500k',.117,.719,.038,'I','与历史A分开解读'),
      ('3','diag_B/B3/B4','完整GT Critic重跑','5×500k',.133,.800,.181,'I','三个标签同一组运行'),
      ('3','B1','Critic只加门角','5×500k',.174,.856,.091,'I','有阶段性AUC数值收益'),
      ('3','B2','Critic加角度+角速度','5×500k',.078,.791,.244,'I','收益不随GT维数单调'),
      ('3','B5','50k后冻结Critic','5×500k',.095,.563,0,'I','未阻止退化'),
      ('3','B6','100k后切换普通Critic','5×500k',.153,.969,.028,'I','未阻止退化'),
      ('3','E0','同encoder，无GT辅助','5×500k',.148,.956,.644,'I','辅助任务对照'),
      ('3','E1','持续GT辅助预测','5×500k',.160,.875,0,'I','表示可预测不等于控制稳定'),
      ('3','E2','前100k GT辅助','5×500k',.257,.981,.213,'I','早期收益，终点退化'),
      ('3','E3','只在BC阶段GT辅助','5×500k',.117,.794,.388,'I','未超过E0 AUC'),
      ('4','E0','无GT辅助','5×500k',.172,.994,.328,'I','新评估协议'),
      ('4','E25','前25k GT辅助','5×500k',.213,.975,.250,'I','窗口组AUC数值最高'),
      ('4','E50','前50k GT辅助','5×500k',.157,.938,.231,'I','仍退化'),
      ('4','E100','前100k GT辅助','5×500k',.181,.903,.266,'I','未复现Phase3大幅AUC收益'),
      ('4','E200','前200k GT辅助','5×500k',.153,.919,.400,'I','仍退化'),
      ('4','E100M','角度/接触多head','5×500k',.135,.856,.088,'I','未改善'),
      ('4','E100RB','GT辅助+较宽松回退','5×500k',.434,.856,.363,'I','平均18次回退/seed'),
      ('4','E100R1','GT辅助+快速回退','5×500k',.684,.806,.816,'I','平均40.4次回退/seed；事后探索'),
      ('4','Q0','robot-only共享Q','5×500k',.086,.775,0,'I','Q与回报排序不可靠'),
      ('4','QGT','共享Q+GT辅助','5×500k',.043,.731,0,'I','未改善'),
      ('4','QV','QGT+门控价值回放','5×500k',.043,.731,0,'I','加权几乎未启用'),
      ('5','Uniform','均匀replay','5×100k',.102,.666,.334,'I','短程配对对照'),
      ('5','Quality gate','成功率在线门槛','5×100k',.102,.666,.334,'I','在线成功0，加权未启用'),
      ('5','Return gate','回报相关门槛','5×100k',.099,.563,.244,'I','事后探索'),
      ('5','Quality offline','离线门槛直接质量加权','5×100k',.121,.666,.388,'I','无稳定统计收益；事后探索'),
      ('6','A0','纯在线SAC，随机初始化','5×500k',.021,.203,0,'I','新在线成功81条'),
      ('6','A1','70%专家+30%在线','5×500k',0,0,0,'I','新在线成功0条'),
      ('6','A2/B0','50%专家+50%在线','5×500k',.004,.203,0,'I','新在线成功47条；标签复用'),
      ('6','B1','BC初始化+普通SAC','5×500k',.0006,.059,0,'I','新在线成功0条'),
      ('6','B2','BC初始化+较低学习率','5×500k',.0006,.059,0,'I','新在线成功2条'),
      ('6','B3','BC初始化+持续BC约束λ10','5×500k',.376,.959,.319,'I','新在线成功13条；最佳不等于终点'),
      ('7','E0','Phase6 B3截到300k','5×300k',.358,.994,.572,'F','新在线成功0条'),
      ('7','E1L','固定低温度α=.001','5×300k',.075,.866,.141,'F','未稳定保持'),
      ('7','E2L','温度α退火','5×300k',.543,1,.213,'F','中期强，终点退化；在线成功0'),
      ('7','E3','仅采集std≤.05','5×300k',.002,.066,0,'F','在线成功约4条'),
      ('7','L000','在线BC λ0','5×300k',.001,.056,0,'F','复用Phase6 B1'),
      ('7','L001','在线BC λ.01','5×300k',.001,.056,0,'F','约束过弱'),
      ('7','L005','在线BC λ.05','5×300k',.001,.066,0,'F','约束过弱'),
      ('7','L010','在线BC λ.1','5×300k',.001,.056,0,'F','约束过弱'),
      ('7','L050','在线BC λ.5','5×300k',.002,.128,.069,'F','仍不稳定'),
      ('7','G1','完整checkpoint回退','5×300k',.733,.825,.797,'F','独立最终85.3%；平均23.6次回退'),
      ('7','E1 exploratory','更高目标熵探索组','5×300k',.633,1,.800,'F','在线成功0；非低熵组'),
      ('7','E2 exploratory','更高目标熵退火探索组','5×300k',.485,1,.803,'F','在线成功0'),
      ('8','A','原奖励baseline+持续BC+保护','5×300k',.759,.800,.800,'I','原始随机在线成功0'),
      ('8','B','A+Progress Reward','5×300k',.870,.991,.959,'I','原始随机在线成功0'),
      ('8','C','A+progress状态输入','5×300k',.741,.800,.791,'I','Actor使用环境状态'),
      ('8','D','progress输入+奖励','5×300k',.711,.797,.784,'I','没有超过B'),
    ]
    lines=['# 全部有效方案与当前指标','',f"快照UTC：{data['utc']}。历史值来自保留报告，显示精度与源报告一致。",'',
      '**当前优先候选：Phase9 C（Phase8 B锚点 + KL约束 + 全流程std上限0.01）。**它以较简单的replay实现标称工位随机成功与低退化；D的额外成功replay没有显示净收益。'+('全部7组已完成5seed×300k及独立复测。' if all_finished else '补充组未完成时不能纳入最终排名。'), '',
      f'按已完成五seed独立随机Final点估计，最高组为**{endpoint_best}：{100*endpoint_rate:.2f}%**；这个排序不等于统计显著领先。D100只从专家replay更新，采到的在线数据不参与梯度，因此不能以其高分证明新在线经验带来改进。C是在线混合经验候选，并非声称它在所有单指标上最高。', '',
      '## 统一解释','',
      '- 表中的AUC是策略成功率随训练交互的归一化面积，不是质量模型的分类ROC AUC。历史Phase1–8为确定性评估，Phase9主表为实际训练采样分布。',
      '- 阶段间训练预算、评估频率、奖励、保护规则等不同，不能把跨阶段AUC差当作单因素的因果收益。Phase9从已经训练好的锚点继续，0步已经有能力。',
      '- I=独立确定性复测；F=训练固定评估；P/I=最佳是训练峰值、最终是独立复测，所以不计算混合口径gap。负gap可来自独立复测波动。',
      '- 全部数值均为seed均值；完整95%区间、逐seed值及配对统计见源报告和对应results。正式结果不包含无效协议和接口pilot；同义标签没有重复计数。', '',
      '## 全部阶段与有效策略方案：统一总表','',
      '| 阶段/任务 | ID | 方法 | seeds×训练步/seed | 成功率AUC | 最佳成功 | 最终成功 | Best−Final（百分点） | 动作/测试口径 | 说明 |',
      '|---|---|---|---|---:|---:|---:|---:|---|---|']
    for phase,arm,name,budget,auc,best,final,mode,note in history:
        gap=f'{100*(best-final):.1f}' if mode!='P/I' else '—'
        lines.append(f'| {phase} | {arm} | {name} | {budget} | {auc:.4f} | {100*best:.1f}% | {100*final:.1f}% | {gap} | {mode} | {note} |')
    names=dict(A='原始std，无anchor',B='KL anchor，原始std',C='anchor+std≤.01',D='C+冻结anchor成功replay',CANN='anchor+std .1→.01退火',D100='D的100%专家replay',D30='D的约30%专家replay')
    for arm,r in data['arms'].items():
        if r['training_complete'] and 'policy_final' in r:
            lines.append(f"| 9 | {arm} | {names[arm]} | 5×300k | {r['policy_auc']['mean']:.4f} | {100*r['policy_best']['mean']:.2f}% | {100*r['policy_final']['mean']:.2f}% | {100*r['policy_gap']['mean']:.2f} | 随机/I | 新在线成功{r['online_successes']:,}/{r['online_episodes']:,}条 |")
    lines+=['','## Phase9：当前全部组，实际训练采样策略','',
      '| 组 | 方法 | 每seed训练步范围 | 独立复测完成 | 随机AUC | 随机Best | 随机Final | 随机Best−Final（pp） | 确定性Final | 新在线成功/完成episode | 平均拒绝次数 |',
      '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in data['arms'].items():
        complete=r['training_complete'] and r['independent_seeds_complete']==5
        def value(k,pct=True):return f"{r[k]['mean']*(100 if pct else 1):.2f}" if k in r else '待完成'
        def percentage(k):return value(k)+'%' if k in r else '待完成'
        auc=f"{r['policy_auc']['mean']:.4f}" if 'policy_auc' in r else '待完成'
        lines.append(f"| {arm} | {names[arm]} | {min(r['steps']):,}–{max(r['steps']):,} | {r['independent_seeds_complete']}/5 | {auc} | {percentage('policy_best')} | {percentage('policy_final')} | {value('policy_gap')} | {percentage('deterministic_final')} | {r['online_successes']:,}/{r['online_episodes']:,}{'' if complete else '（累计中）'} | {value('rejections',False)} |")
    lines+=['','### 当前主组95% CI与多seed稳定性','',
      '| 组 | 随机Final均值 [t95 CI，%] | Final seed SD（pp） | 门+2.5°随机Final | 门+5°随机Final | 把手−1cm随机Final | 把手+1cm随机Final |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('A','B','C','D'):
        r=data['arms'][arm];m=r['policy_final'];ci=m['ci95'];p=r['perturbed_policy_final']
        lines.append(f"| {arm} | {100*m['mean']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {100*m['sd']:.2f} | "+' | '.join(f"{100*p[c]['mean']:.2f}%" for c in ('angle_2p5','angle_5','handle_y_minus_1cm','handle_y_plus_1cm'))+' |')
    difference=data['primary_paired_final']['D-C']
    d=difference['difference'];lo,hi=d['ci95']
    lines+=['',f"D−C随机Final配对差={100*d['mean']:.3f}pp，95% CI [{100*lo:.3f},{100*hi:.3f}]pp，精确双侧p={difference['exact_signflip_p_two_sided']:.4f}。t区间未裁剪，超过0–100%是小样本正态区间性质，不表示成功概率能超过100%。",'',
      'C/D都未通过小扰动标准：门+5°为0%，把手−1cm明显失败。C/D某些seed拒绝超过20%的候选区间；不能说全seed都不依赖回退。当前不进入LWD/DIVL。', '',
      '## Phase5：离线质量/价值模型（不是策略成功AUC）','',
      '| 模型 | 主测试ROC AUC | 压力测试ROC AUC | 同来源ROC AUC | 压力Top20%成功率 |',
      '|---|---:|---:|---:|---:|',
      '| E2 SAC Q | .489 | .417 | .571 | 50.0% |',
      '| Q0共享Q | .505 | .643 | .485 | 79.8% |',
      '| QGT共享辅助Q | .464 | .650 | .486 | 85.0% |',
      '| MC回报预测 | .720 | .338 | .548 | 58.3% |',
      '| 离线TD0 | .709 | .480 | .561 | 47.5% |',
      '| TD+MC | .562 | .383 | .564 | 57.0% |',
      '| 固定前50步质量模型 | .976 | .793 | .808 | 79.1% |',
      '| 固定前100步质量模型 | 1.000 | .888 | .878 | 93.6% |',
      '| 完整轨迹事后模型（上限） | 1.000 | 1.000 | 1.000 | 100% |','',
      '前100步质量模型压力测试ROC AUC的95% CI为[.860,.916]；排序有效不意味着质量replay已提升策略。完整轨迹模型在episode结束后评分，不是在线提前选择器。','',
      '## BC与baseline口径','',
      '- Phase6纯BC起点独立确定性19/320=5.94%；Phase8纯BC起点固定评测18/320=5.625%。',
      '- Phase8 A baseline最终80%是持续BC+SAC+专家回放+保护后的独立确定性结果；不是纯BC，也不是普通无保护SAC。',
      '- 当前缺少同等总更新次数的BC-only对照，所以不能把6%→80%全部归因于RL。高成功率还需注明best/final、确定性/真实采样、固定工位/扰动。','',
      '## 来源','']
    for name in ('final_report_zh','door_rl_report','privileged_analysis_report','stable_privileged_report','value_quality_report','phase6_stable_rl_report','phase7_stable_policy_report','phase8_progress_rl_report'):
        lines.append(f'- [{name}]({name}.md)')
    lines+=['- Phase9当前原始快照：results/phase9_safe_online/current_metrics.json；各组eval/episodes/updates CSV与heldout JSON。','']
    destination=ROOT/'docs/method_comparison.md'
    destination.write_text('\n'.join(lines),encoding='utf-8')
    print(destination)

if __name__=='__main__':main()
