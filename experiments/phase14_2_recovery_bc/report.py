"""Scientific report generated solely from completed artifacts."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'results/phase14_2_recovery_bc'

def pct(x):return f'{100*x:.2f}%'
def interval(d):return f'{pct(d["mean"])} [{100*d["ci95"][0]:.2f}, {100*d["ci95"][1]:.2f}]'
def delta(d):return f'{100*d["mean"]:+.2f} pp [{100*d["ci95"][0]:+.2f}, {100*d["ci95"][1]:+.2f}]'

def main():
    all=json.loads((R/'summary.json').read_text());s=all['summary'];paired=all['paired'];gate=all['anchor_gate']
    m=json.loads((ROOT/'datasets/phase14_2/prepared_v1/manifest.json').read_text());diag=json.loads((R/'diagnosis.json').read_text())
    n=json.loads((ROOT/'datasets/phase14_2/success_size_matched/collection_v1/summary.json').read_text())
    audit=json.loads((R/'preservation_after.json').read_text())
    lines=['# Phase14.2：恢复数据与参数条件 BC','',
       '## 结论','',
       f'主方案 D 随机工位成功率为 **{interval(s["D"]["random"])}**（5个训练seed，Student-t95%CI）。',
       f'D−B={delta(paired["D-B"]["random"])}；D−F={delta(paired["D-F"]["random"])}；D−C={delta(paired["D-C"]["random"])}。pp表示百分点。',
       f'**按固定20k最终权重，D={pct(all["final_update_random"]["D"]["mean"])}、B={pct(all["final_update_random"]["B"]["mean"])}，D−B={delta(all["final_update_paired"]["D-B"])}。这一终点有正向配对证据，但收益不跨选模协议稳定，也未证明优于普通增量数据。不能笼统宣称恢复数据完全无用。**',
       '本轮不训练RL；SAC/Residual/KL RL/LWD/DIVL更新数均为0。',
       ('恢复增强方法通过预设候选门槛，但仍需独立复测后才能命名新的Anchor。' if gate['candidate_qualified'] else '**未通过可靠随机工位Anchor门槛，因此不进入Phase14.3，不进入LWD/DIVL。保留全部模型作为候选和诊断产物。**'),
       '', '## 数据、来源与划分','',
       '|来源|成功轨迹|train轨迹/transition|validation轨迹/transition|test轨迹/transition|',
       '|---|---:|---:|---:|---:|']
    for name,key in [('S：Phase13原随机成功','base'),('R：Phase14.1正式人工恢复成功','recovery'),('N：普通随机成功增量','ordinary')]:
        c=m['counts'][key];total=sum(v['trajectories'] for v in c.values())
        lines.append(f'|{name}|{total}|{c["train"]["trajectories"]}/{c["train"]["transitions"]}|{c["validation"]["trajectories"]}/{c["validation"]["transitions"]}|{c["test"]["trajectories"]}/{c["test"]["transitions"]}|')
    lines+=['',f'N从正常Level2初态开始，使用与R完全相同的Phase14.1恢复专家类和动作补偿，采集成功 **{n["successes"]}/{n["episodes"]}={pct(n["success_rate"])}**，保存484成功轨迹；失败尝试保留在attempts.csv。没有主动失败注入、策略前缀或人工遥操作。',
        '', 'S原始文件不修改，沿用原200/50/50整轨迹划分。R按四类扰动分层、整轨迹划分；N独立整轨迹划分。R含全部484条正式成功恢复轨迹，16失败恢复不参与BC。各源test轨迹均不训练。',
        '', 'R只取专家接管后的顶层纠正transition，不读取嵌套policy_and_injection_prefix。Phase14旧自然数据与Phase14.1自然crosscheck均不参与训练或本轮最终压力测试。',
        '', '继承任务的限制：reset门角为0–5°，但原门关节驱动的目标为0、stiffness=10、damping=2.5，在接近过程中会使初始门角趋向关闭；把手XYZ随机偏移持续存在。因此完整随机成绩主要证明该既有工位随机协议下的适应，不能解释为持续多门角状态的泛化。此轮未改变门驱动、reward、reset或机器人配置。',
        '', '### 等采样预算与重复使用','',
        '每组20000更新、batch512，总10240000样本抽取。A/B全部抽S；C/D与E/F均每batch256S+256新增源，新增源各抽5120000次。N与R采用用户允许的训练采样量匹配，而非声称原始transition数完全相同。',
        '', '|组|S抽取数|新增源抽取数|S平均重复次数|新增源平均重复次数|','|---|---:|---:|---:|---:|']
    for arm,v in diag['sampling'].items():lines.append(f'|{arm}|{v["S_examples"]}|{v["added_examples"]}|{v["S_average_reuse"]:.2f}|{v["added_average_reuse"]:.2f}|')
    lines+=['','## 输入、模型与公平性','',
        '复用Phase13的Policy：39→256→256→7，ReLU/tanh，连续动作MSE、Adam3e−4；77838总参数（含7个冻结log_std）。所有组容量、训练seed0–4、更新次数、优化器、动作接口均一致。7组按arm维向量化执行，独立权重与Adam状态；初始化前校验向量化forward与原Policy相同。相同GT对照共用S与新增源抽样索引。',
        '', '输入为robot26与13个有效GT特征：门角、门角速度、进度、剩余角、两指接触、把手XYZ+四元数。任务目标角始终1rad：作为显式配置常量保存，亦可从门角+剩余角恢复。将原恒定目标角槽用于角速度，保持原网络结构和容量；本接口不用于变化目标角任务。',
        '', '归一化沿用原train-S统计，仅速度槽用train-S原始物理速度重新拟合；所有组共享。Robot-only在归一化后屏蔽全部GT；D_no_handle只屏蔽把手7个通道。没有planner阶段/专家内部计时/未来动作输入。完整GT是假设工装能稳定提供标定位姿的条件结果；实际硬件位姿传感尚未验证，另列去位姿结果，不能将完整GT直接视为部署已解决。',
        '', '每1000更新，以三源validation的等权动作MSE选择best，所有组使用同一评估集合与选择规则，不以测试成功率选模。这个共同验证目标同时覆盖恢复与普通状态，与Phase13仅S验证不完全相同。因此另列所有A–F最终20k权重的随机成功率，检查选模敏感性；历史9.06/47.34/63.28%仅作背景，不能作本轮配对对照。',
        '', '## 五seed闭环结果','',
        '数值为均值%，方括号为训练seed95%CI；人工恢复列为四类型等权平均。CI可超出0–100，是未截断的Student-t区间。',
        '', '|组|数据/输入|固定|完整随机（95%CI）|人工恢复（95%CI）|自然严重偏离（95%CI）|随机seed SD|',
        '|---|---|---:|---:|---:|---:|---:|']
    labels={'A':'S/robot','B':'S/GT','C':'S+R/robot','D':'S+R/GT','E':'S+N/robot','F':'S+N/GT','D_no_handle':'S+R/GT去把手pose'}
    for arm in labels:lines.append(f'|{arm}|{labels[arm]}|{pct(s[arm]["fixed"]["mean"])}|{interval(s[arm]["random"])}|{interval(s[arm]["artificial_mean"])}|{interval(s[arm]["natural_severe"])}|{pct(s[arm]["random"]["sd"])}|')
    lines+=['','### 随机工位seed明细与最终20k敏感性','',
        '|组|best-validation seed0–4（%）|final20k seed0–4（%）|final20k均值与95%CI|','|---|---|---|---|']
    for arm in labels:
        d=all['final_update_random'].get(arm);f=' / '.join(f'{v*100:.2f}' for v in d['values']) if d else '未追加此敏感性对照'
        lines.append(f'|{arm}|'+ ' / '.join(f'{v*100:.2f}' for v in s[arm]['random']['values'])+f'|{f}|{interval(d) if d else "—"}|')
    lines+=['','### 固定20k最终权重的配对敏感性','', '|比较|随机差值pp（95%CI）|paired t p|敏感性家族Holm p|exact signflip p|','|---|---:|---:|---:|---:|']
    for k,d in all['final_update_paired'].items():lines.append(f'|{k}|{delta(d)}|{d["paired_t_p"]:.5g}|{d["holm_p"]:.5g}|{d["exact_signflip_p"]:.5g}|')
    lines+=['','D的最低验证MSE权重比最终20k权重的闭环成绩更低，B也有选模与最终性能差距。共同动作MSE不等同于多步任务可靠性。未来需要独立closed-loop validation选择checkpoint，当前不得用本轮测试集再挑高分权重作为Anchor。',
            '', '### 预设核心配对比较','', '|比较|随机成功差值pp（95%CI）|正向seed数|paired t p|Holm p|exact signflip p|','|---|---:|---:|---:|---:|---:|']
    for k,v in paired.items():
        d=v['random'];lines.append(f'|{k}|{delta(d)}|{d["positive_seeds"]}/5|{d["paired_t_p"]:.5g}|{d["holm_p"]:.5g}|{d["exact_signflip_p"]:.5g}|')
    lines+=['','统计单位为训练seed；不能把同一策略的数百episode当作数百独立训练seed。所有组采用同一未见过的reset序列。n=5的精确双侧signflip最小p=0.0625，因此即使t检验显著，也明确报告精确检验限制；四个主比较同时报告Holm校正。',
        '', '## 恢复阶段与独立压力测试','',
        '每模型固定64、随机128 episodes；人工四类型各64；自然严重偏离64。共享新测试seed生成的物理失败状态，参考前缀来自冻结Phase13候选，并非恢复专家。自然状态从未主动注入扰动的参考策略rollout在tick169筛选，末端距把手0.10–0.20m；不是旧0/5状态或训练R状态。',
        '', '压力测试采用同样32-clone物理布局并打包状态：重放真实动作历史，恢复记录的robot q/qdot及门q/qdot，保持1tick重建真实接触测量，再在策略接管边界恢复一次记录的位置和速度。自然失败速度较大，单保持步可移动数厘米，因此这一步重建是测试初始化；学习策略接管后没有状态修正或专家动作。保持tick计入原600tick时限。接触输入来自真实保持步的测量，PhysX内部求解器历史不能完整序列化；不声称所有内部物理变量比特级相同。每个模型实际初始化robot26/GT11都保存并检查，此轮35模型的可观察初始化状态与接触标签完全一致。',
        '', '阶段定义：进入把手6cm内连续6tick；双指接触连续3tick；有效抓取代理=双指接触+6cm内+夹爪关闭动作+夹爪宽0.5–7.9cm连续3tick（并非完整力闭合证明）；门角高于接管时0.05rad连续30tick；最终门角>1rad。阶段只统计策略动作后的观测，不将初始化保持动作当作学习策略恢复动作。',
        '', '|测试/组|重新接近|重新接触|抓取代理|持续进度恢复|最终成功|','|---|---:|---:|---:|---:|---:|']
    for test in ['ee_offset','contact_loss','door_regression','stagnation','natural_severe']:
        for arm in ['B','C','D','F','D_no_handle']:
            ds=[json.loads((R/'evaluation'/f'{arm}_seed{i}'/(test+'.json')).read_text()) for i in range(5)]
            lines.append('|'+test+'/'+arm+'|'+'|'.join(pct(np.mean([d[k] for d in ds])) for k in ['reapproach','recontact','regrasp','progress_resumed','success'])+'|')
    lines+=['','已在邻域/已有接触的状态可能使阶段总比例较高；每episode CSV保留initially_near/initial_contact，JSON另给when_needed条件率。自然严重偏离全部来源于自然前缀，但压力测试是冻结状态分布上的条件恢复，不等同于各策略自然rollout的失败频率或端到端总体恢复。',
            '', '### 实际压力状态匹配审计','', '|测试|max EEF组间跨度mm|max handle跨度mm|max q跨度rad|接触标签完全一致比例|','|---|---:|---:|---:|---:|']
    for test,d in all['physical_state_matching'].items():lines.append(f'|{test}|{1000*d["max_eef_span_m"]:.4f}|{1000*d["max_handle_span_m"]:.4f}|{d["max_q_span_rad"]:.5f}|{pct(d["contact_consistent_fraction"])}|')
    q=all['same_S_expert_quantity_control'];effects=all['same_S_expert_effects']
    ns=json.loads((ROOT/'datasets/phase14_2/success_size_matched/same_S_expert_v1/summary.json').read_text())
    lines+=['','## 同原专家的数据量补充对照','',
        '原六组N与R使用相同Phase14.1专家，能控制新增专家版本；该专家与S的Phase13专家存在正常接近路径/阶段时序差异。因此F−B同时含“更多普通数据”和“专家风格混合”，不能直接称为纯数量效应。发现这个方法风险后，在读取同原专家新数据的结果之前，追加一个有记录的协议修正：再生成484正常成功轨迹N_sameS，用完全不修改的Phase13 staged_free专家。S/R及原N不改、不替换、不择优删除。',
        '', '新增E_S/F_S为S+N_sameS，五个相同seed、相同网络、20k更新、512batch、50/50源采样。选模仍用原冻结共同S/R/N_v5验证集，避免改变已声明的选择目标；新N的独立validation额外保留。两组均测固定/随机，最终20k再测随机，提供普通数据规模的严格源风格对照。',
        '', f'N_sameS采集{ns["successes"]}/{ns["episodes"]}={pct(ns["success_rate"])}，484条成功轨迹按338/72/74整轨迹划分；采集174304次仿真交互。原始文件哈希和抽样预算写入`datasets/phase14_2/quantity_control_v1/manifest.json`。补充量对照和主对照一起完整保留。',
        '', '|补充组|固定best|随机best（95%CI）|随机final20k（95%CI）|','|---|---:|---:|---:|']
    for arm,d in q.items():lines.append(f'|{arm}|{pct(d["best_fixed"]["mean"])}|{interval(d["best_random"])}|{interval(d["final_random"])}|')
    lines+=['','|补充比较|随机差值pp（95%CI）|paired t p|补充家族Holm p|exact signflip p|','|---|---:|---:|---:|---:|']
    for k,d in effects.items():lines.append(f'|{k}|{delta(d)}|{d["paired_t_p"]:.5g}|{d["holm_p"]:.5g}|{d["exact_signflip_p"]:.5g}|')
    lines+=['','F_S−B控制原专家风格，用于判断只增加普通成功数据的效果；D−F_S与D−F同时报告，分别对应原专家数量对照与新增专家版本对照。补充比较是诊断性协议修正，不能替代预设主比较来宣布D通过Anchor门槛。']
    lines+=['','## 专家标签、MSE与失败诊断','',
         '恢复训练占50%，并不属于恢复数据权重过低的设置。动作MSE与实际闭环成功分开报告：bc_seed*_summary.json提供三源未见轨迹的transition/trajectory MSE；diagnosis.json按release/clear/orient/approach/align/grasp/open分解位置、旋转、夹爪误差与夹爪符号准确率。',
         '', '下表为5seed的未见轨迹动作MSE均值，格式为transition等权 / trajectory等权；MSE基于归一化动作，不是实际末端跟踪位置误差。',
         '', '|组|S test MSE|R test MSE|N test MSE|','|---|---:|---:|---:|']
    heldout=[row for seed in range(5) for row in json.loads((ROOT/f'checkpoints/phase14_2_recovery_bc/bc_seed{seed}_summary.json').read_text())['test']]
    for arm in labels:
        cells=[]
        for source in ['base','recovery','ordinary']:
            rows=[v for v in heldout if v['variant']==arm and v['source']==source]
            assert len(rows)==5
            cells.append(f'{np.mean([v["transition_mse"] for v in rows]):.6f} / {np.mean([v["trajectory_mse"] for v in rows]):.6f}')
        lines.append('|'+arm+'|'+'|'.join(cells)+'|')
    lines+=['', 'R/N共用修复后的专家，S使用原Phase13专家，因此D−F控制了新增专家控制器变化；D−B同时涉及新增恢复覆盖和新增标签来源，不能仅凭这个差值归因恢复覆盖。最近邻动作分歧是近似状态比较，不是精确状态动作矛盾或动作多模态的因果证明。',
         '', '|R训练阶段|transition|占R%|占整体batch%|每512batch期望样本|','|---|---:|---:|---:|---:|']
    for phase,v in diag['R_train_phase_exposure'].items():lines.append(f'|{phase}|{v["transitions"]}|{100*v["R_fraction"]:.2f}|{100*v["whole_batch_fraction"]:.2f}|{v["expected_samples_per512"]:.2f}|')
    lines += [
         '', 'R中open约66.6%，clear+approach+orient在整个batch合计约4.4%。恢复源占50%并不保证最关键重新接近/重定向状态占足够比重。阶段平衡抽样是待验证的假设，本轮没有更改主采样和loss。',
         '', '下一步：先以独立闭环validation选模，保存完整checkpoint轨迹并核对阶段MSE与部署成功的关系；再单独对照恢复前段分层采样。若仍失败，检查专家阶段/阶段内计时的不可观察性、单步MSE对不同纠正动作的平均化、相对EEF–handle几何与短历史。现有近邻证据和专家路径差异不能单独证明这些因果解释。不要直接增加恢复数据或进入RL。',
         '', '## Anchor门槛与八个交付问题','',f'机器可读门槛：`{json.dumps(gate,ensure_ascii=False)}`。','']
    lines += [
       f'1. **恢复数据是否提高随机BC？** 验证MSE选模D−B={delta(paired["D-B"]["random"])}；最终20k的D−B={delta(all["final_update_paired"]["D-B"])}。终点有正向证据，主选模比较没有稳定正向证据，不能把这两种口径混用。',
       f'2. **是否只是数据更多？** D−F={delta(paired["D-F"]["random"])}；同原专家数量对照D−F_S={delta(effects["D-F_S"])}、F_S−B={delta(effects["F_S-B"])}。未证明恢复覆盖稳定优于普通增量数据。',
       f'3. **GT在恢复学习中是否有效？** D−C={delta(paired["D-C"]["random"])}；'+('GT条件输入仍有正向证据，但包含完整把手位姿，去位姿模型的主随机指标仅9.38%。' if paired['D-C']['random']['ci95'][0]>0 else '本轮配对比较尚不足以证明稳定GT收益。'),
       f'4. **是否学会完整纠正链？** D人工恢复={interval(s["D"]["artificial_mean"])}；自然恢复={interval(s["D"]["natural_severe"])}。必须结合上表逐阶段瓶颈，不能用接近率或专家成功率代替策略最终恢复率。',
       f'5. **新Anchor成功率？** 当前D五seed主指标={interval(s["D"]["random"])}，最终20k={interval(all["final_update_random"]["D"])}；'+('候选仍需独立复测，不能先宣称已建立Anchor。' if gate['candidate_qualified'] else '没有通过门槛的新Anchor；已有历史候选保持原样。'),
       f'6. **严重自然偏离的问题？** 自然压力测试逐阶段如上；D最终={pct(s["D"]["natural_severe"]["mean"])}。这一独立压力域不要求90%，也不与旧5例不同状态的专家结果作因果涨跌比较。',
       '7. **是否进入14.3？** '+('必须先完成候选独立复测并更新资格文件；本轮不训练RL。' if gate['candidate_qualified'] else '不允许：可靠随机恢复Anchor的预设门槛未满足。'),
       '8. **是否进入LWD/DIVL？** 不满足：本轮只验证BC，还缺随机工位稳定在线成功经验及RL相对冻结Anchor的正向增益证据。',
       '', '## 资源、完整性与复现','',
       '训练5seed×7主组与5seed×2补充组，共45模型，每模型20k更新；没有RL训练。主评估8进程×32环境；数量对照与补齐初版归档项各最多4进程，峰值16进程×32=512环境，单GPU最多2进程，各4 CPU线程、nice10，共享8×B300，无终止其他项目。全部最终有效评估24960 episodes。仿真版本、任务与runtime_env.sh沿用原项目。',
       '', f'历史保护检查{audit["checked"]}个文件；变化{len(audit["changed"])}。源数据SHA256如下：','',
       '```json',json.dumps(m['raw_sha256'],indent=2),'```','',
       '结果：`results/phase14_2_recovery_bc/{summary.json,metrics.csv,diagnosis.json,evaluation/,evaluation_final/}`；权重和训练曲线：`checkpoints/phase14_2_recovery_bc/`；日志：`logs/phase14_2_recovery_bc/`；划分与归一化：`datasets/phase14_2/prepared_v1/manifest.json`。',
       '', '复现入口：`source configs/runtime_env.sh; .venv/bin/python -m experiments.phase14_2_recovery_bc.run --eval-gpus 0,1,2,3,4,5,6,7`。保留完成项并检查全部预期结果，而非仅凭仿真进程退出码0判断完成。',
       '', '![BC比较](../results/phase14_2_recovery_bc/figures/paired_bc_comparison.png)','']
    (ROOT/'docs/phase14_2_recovery_bc_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (R/'anchor_qualification.json').write_text(json.dumps(gate,indent=2))
    print('REPORT WRITTEN',flush=True)

if __name__=='__main__':main()
