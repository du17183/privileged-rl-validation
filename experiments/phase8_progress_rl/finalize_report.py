"""Build the final report exclusively from completed measured artifacts."""
from runtime_paths import project_path
import json
from pathlib import Path
from experiments.phase8_progress_rl.analyze import read
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results"/"phase8_progress_rl"
PUBLIC = project_path('/home/xiaolong/privileged_rl_validation')


def main():
    manifest = json.loads((OUT/"completion_manifest.json").read_text())
    assert manifest["primary_runs"] == 20 and manifest["independent_tests"] == 80
    summary = json.loads((OUT/"summary.json").read_text())
    pairs = json.loads((OUT/"paired_tests.json").read_text())
    heldout = json.loads((OUT/"heldout_summary.json").read_text())
    diagnosis = json.loads((OUT/"update_diagnosis.json").read_text())
    table = (OUT/"report_tables.md").read_text()
    delta = pairs["comparisons"]["B-A"]["auc"]["difference"]
    total_eval = sum(s["evaluation_steps"]["mean"]*5 for s in summary.values())
    frozen_steps = sum(int(read(p)[0]["eval_env_steps"])
                       for p in (OUT/"heldout").glob("*.csv"))
    prose = f"""# Phase 8：可部署进度参数驱动 Panda Door RL

完成日期：2026-09-30。服务器入口 b300-2；项目路径 `{PUBLIC}/`。

## 1. 本轮结论

完成 A–D 各五 seed、每 run 300,000 次训练交互，共 **20 run / 6,000,000 次训练交互**；另完成最佳与最终 checkpoint 的 80 个独立测试，共 **5,120 个评估回合**。全部进程退出成功、曲线和预算完整，专家及原 Door 环境文件哈希保持一致。

仅增加有符号进度奖励的 B 在当前配方中表现最好：受保护成功率 AUC **0.870**，独立最终 checkpoint 确定性成功率 **0.959**，附加 std=0.01 时 **0.938**。A 相应为 0.759 / 0.800 / 0.778。C（状态输入）与 D（状态+奖励）没有提高整体成绩。

**这些结果尚未建立稳定在线学习闭环。**所有 20 run 的随机在线采集成功轨迹数均为零；成功能力来自专家回放参与的更新及 checkpoint 保留，回退前策略仍反复退化。B 的平均成绩优势没有通过五 seed 配对统计证明，不能表述为普遍提高样本效率。D 未通过预设稳定性门槛，E、500k 延长、轻度随机化、Drawer 复制均未启动；这属于条件实验未开展，没有给它们赋予零分或虚构结果。

## 2. 环境、数据与共同配方

沿用 Panda Door Opening 和已有 cabinet 抓握条资产、reward 基线、reset、机器人配置与 1,000 条专家轨迹；本阶段只在独立包装器中施加状态/奖励干预。Isaac Sim 5.1.0.0；Isaac Lab 源码 v2.3.0、commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`，包元数据 0.47.2；Python 3.11.15、PyTorch 2.7.0+cu128、CUDA runtime 12.8、NumPy 1.26.0。8×NVIDIA B300 SXM6 AC，每卡 275,040 MiB；记录的驱动 580.82.07。版本来源为既有 `results/environment.json` 与 `docs/door_environment_report.md`。

任务为 6 维相对末端差分 IK + 1 维二值夹爪动作，60Hz，最长 600 步 / 10 秒，右门 hinge 范围 0–1.57 rad，关闭初态 0，完整成功门槛沿用 **angle >1 rad（约57.3°）**。新增监测没有改变 A–D 的提前终止规则。训练与本轮复测保持固定门初角和把手位置，保留原配置已有的物理材料设定；没有新增位姿/摩擦泛化干预。

| 参数 | 所有 A–D |
|---|---|
| 训练 seed | 0,1,2,3,4 |
| 并行环境 / run | 32 |
| BC 初始化 | 3,000 更新，相同专家 minibatch seed |
| SAC learning rate / batch | 3e-4 / 256 |
| 每个32环境向量步 | 4次优化更新 |
| 专家 replay 比例 | 50% |
| 持续 BC 约束 | raw MSE 系数 λ=10 |
| 熵 | 沿用自动温度，目标熵 −7 |
| 评估 | 每10k训练交互、固定64回合 |
| 独立复测 | 新进程，seed=90000+训练seed；每 checkpoint/动作模式64回合 |

专家 HDF5 有 **271,562 条 transition / 1,000 条轨迹**。B/D 在内存中重标注专家奖励，文件只读；A–D 使用相同数据。Phase 1–7 结果目录不写入。本阶段预检和最终审计验证以下 SHA256：

| 文件 | SHA256 |
|---|---|
| door_dataset/door_expert_1000.h5 | a17c857eb760517bd0d59e266c4f008c5fcf980faf8b8052d4343b57c7b069a8 |
| door_env/door.py | b9e710e6195d5cfba46fffc23afd8896871e2b70a6c1ae6d8185625dbe68fb69 |
| door_env/isaac_env.py | d5efaf1041361547bc6e865ffebca801556eebfcf4306b1bcb3001aad3b80fa6 |

## 3. 进度接口和消融

`ProgressDoorEnv.get_progress_state()` 返回 door_angle、door_angular_velocity、target_angle、progress、remaining_angle、contact_state。progress=(angle−start)/(target−start)，限制到[0,1]；每次 reset 保存真实起始角。成功独立使用未裁剪门角与目标比较。接触定义为左右手指到把手的过滤接触力均>0.5N，真机须由可靠的接触测量替换。

| 组 | Actor/Critic 输入 | Reward |
|---|---|---|
| A | 原26维机器人观测 | 原奖励 |
| B | 与A相同 | 有符号角差替换正向速度项 |
| C | 26维+5维进度状态 | 与A相同 |
| D | 与C相同 | 与B相同 |
| E（条件实验） | 与D相同，目标可变 | 与D相同，分阶段目标 |

机器人观测为9关节位置、9关节速度、TCP位置3、四元数4、episode时钟1。新增五维为角度rad、角速度/5、目标rad、完成比例、双侧接触。C/D Actor 和 Critic 都使用31维，没有 Critic 单独可见的GT。核心输入不含把手位姿。新增输入列先置零，其他网络权重从同 seed 的26维基线复制，预检初始 Actor/Q 输出一致；随后相同3,000次 BC 会学习新增输入。故 C/D 的 step0能力允许与 A/B不同，必须同时阅读初始值，不能把全部差异归为在线更新。

原奖励已有进度信号：`reach +0.5 grasp +20 max(angular_velocity,0)+600 success`，RewardManager 按 dt=1/60积分。B/D 保留 reach/grasp/终奖，将积分后的速度项换为 `20*(angle_next-angle)`。这是**有符号角差相对正向速度奖励**的对照。未验证其为折扣意义下的策略不变 shaping。专家前后角与下一角速度用于相同重标注，避免线上/线下 reward 不一致。

预检4,096条专家样本的原奖励重建误差为0；在线奖励最大误差约9.54e-7；8个显式成功/超时事件确认 terminal 快照在 reset 前捕获，关门与 home reset 正常。监测记录最大/最终角、累计正向角差、倒退量、success、return、回合长度。停滞为连续120步角增量<1e-4 rad，倒退回合为累计负角差>0.05rad。CSV `lost_contact` 实际表示120步无双侧有效接触，包含初始尚未抓握阶段，不能直接解释为已经抓住后丢失接触；仅作诊断，不新增提前 reset。

## 4. 保护器与三种动作分布

所有 A–D 使用同一保护器：相对历史锚点 success下降>.25、progress下降>.15、最大角下降>.15rad、或倒退量增加>.10rad时，恢复 Actor、双Q、目标Q、三个优化器和温度参数。替换最佳还需满足成功/进度容差与倒退约束。每10k保存回退前 `raw_step_*` 和回退后 `step_*`；`best.pt` 为保护器接受的文件。

训练采集和 Actor loss 均用原 SAC squashed Gaussian。BC loss 约束 `tanh(mean)` 与专家动作的 MSE，**没有直接限制 log_std**。本轮不通过简单降温或改变训练std改善结果。

部署评估分别使用确定性 `tanh(mean)`，以及对 mean 添加固定 **pre-tanh Gaussian std=0.01** 后执行。后者不等于用学习到的 SAC σ采样，更不代表随机训练策略已经稳定。

raw 曲线表示**受保护训练历史中每个10k区间的回退前候选策略**，不能当作一整轮无保护SAC的反事实结果。受保护曲线的最高值也可能不等于保护器最终选择的 best.pt 验证值；报告两者及独立复测，避免混用。

## 5. 完整五seed结果

统计单位为训练seed，t分布95% CI、样本方差和同seed配对精确双侧符号翻转检验。仅5seed、32种符号组合，最小非零双侧p=0.0625；多臂探索结果不宣称p<0.05。表中概率CI为便于阅读裁剪到[0,1]，JSON保留未经裁剪的t区间。CI刻画这五个训练seed的不确定性，不能把重复固定工位回合视为独立任务配置。

主AUC是0–300k **训练交互**上的归一化梯形面积；step0是3,000次离线BC后的测量，不包含专家数据生成成本。达到阈值的0步表示BC已有该能力。未达到的seed保留未达到，不能删除失败seed后计算平均到达时间。CSV另存首次阈值对应的训练+保护评估交互数。

{table}

五seed final样本方差 A/B/C/D 为 **0.198486 / 0.000415 / 0.192676 / 0.158374**。B的受保护输出跨seed最集中；这仍需要频繁回退。

### 配对效应的实际含义

B−A受保护AUC差为 **{delta['mean']:.3f}**，95% CI **[{delta['ci95'][0]:.3f},{delta['ci95'][1]:.3f}]**，exact p=1.000。seed0–4差为 **0.000 / +0.803 / −0.003 / −0.007 / −0.237**。优势主要来自seed1从失败变为成功，seed4更慢；没有一致的逐seed提速。Raw AUC差+0.089，CI[−0.073,0.252]，p=0.375，同样未证明明确改善。

状态与奖励的受保护AUC交互项 `D−B−C+A` 为−0.141，CI[−0.612,0.331]、p=0.750，没有协同效应证据。独立最终确定性复测B−A均值+0.159，配对p=1.000；应报告为观察到的均值优势，而非已证明普遍提高成功率。

### 交互成本

训练总计6,000,000步，训练期间保护/固定评估总计 **{total_eval:,.0f}步**，80个独立测试另计 **{frozen_steps:,}步**。因此回退保护并非免费，在真机上尤其不能把“300k训练步”解释为全部工装交互成本。这里只报告相同训练预算下的消融，没有证明同等总物理交互预算下的最优效率。

## 6. 更新和随机性诊断

| 组 | Raw best−final均值 | 最后候选策略参考集pre-tanh σ均值 | 动作均值漂移MSE | 独立最终策略实际状态上σ | tanh后动作MC std |
|---|---:|---:|---:|---:|---:|
"""
    for arm in ("A", "B", "C", "D"):
        diag = diagnosis["groups"][arm]["late_diagnostics"]
        independent = next(r for r in heldout if r["variant"] == arm and r["mode"] == "final" and r["noise"] == 0)
        prose += (f"| {arm} | {summary[arm]['raw_gap']['mean']:.3f} | {diag['policy_std_mean']['mean']:.3f} | "
                  f"{diag['action_drift_mse']['mean']:.6f} | {independent['mean_raw_policy_std']['mean']:.3f} | "
                  f"{independent['mean_action_std_mc']['mean']:.3f} |\n")
    prose += f"""
MC std是在独立测试访问的状态上，从学习到的策略分布额外采样8次估计，不用于执行；单独的torch.Generator避免改变执行噪声随机序列，GPU审计通过。实际执行仍是表中确定性/0.01模式。参考集诊断与实际执行访问状态不同，不混为同一测量。

策略分布仍明显宽于部署0.01设定，角度输入/有符号奖励没有使其自然收敛到精细操作分布。B随机在线采集仍为零成功，但确定性与小噪声策略有高成功率。该结果支持“训练采集与部署分布仍不匹配”，没有单独证明Q误差或某个状态维度是退化的因果原因。

Raw最后候选的seed成功率：A=0.594/0/0/0/0；B=1/0/0.953/0.359/0；C=0.953/0.016/0/0.828/0；D=0.344/0.984/0/1/0.078。保护后gap很小，同时原候选仍失败，这说明保护器保存了能力，更新本身尚未稳定。

保护器也可能保护很弱的锚点：A seed1未建立可靠成功能力，C seed4始终保留step0低成功锚点。早期一些回退只因累计倒退超过阈值；本轮没有训练“取消某一条回退规则”的反事实，不能把所有退化归因于探索或认定当前保护阈值最优。下一轮应区分“拒绝部署替换”和“回滚学习器”，并要求恢复锚点先通过独立验收。

## 7. 曲线

阴影为五seed t 95% CI；图示概率区间裁剪到[0,1]。所有横轴均为训练交互次数。

![保护后的成功曲线]({PUBLIC}/results/phase8_progress_rl/figures/protected_success.png)

![每次回退前候选成功曲线]({PUBLIC}/results/phase8_progress_rl/figures/raw_success.png)

![学习到的策略标准差]({PUBLIC}/results/phase8_progress_rl/figures/policy_std.png)

progress、最大门角、倒退量及0.01噪声曲线保存在同一figures目录。

## 8. 七个研究问题

### 问题1：门角作为Reward是否提升RL？

B是本轮最值得保留的候选：受保护AUC和独立最终成功率均值最高，五seed部署输出方差明显小。但配对CI跨零、效应主要来自一个seed，随机在线采集无成功，尚不能确认进度奖励普遍提升在线RL样本效率。原reward已有正向角速度信号，本轮新增的是有符号反馈。

### 问题2：作为Actor输入是否提升RL？

C相对A受保护AUC−0.019、独立最终确定性成功率−0.009；没有整体提升证据。C/D的BC初始平均成功率0.250，相比A/B的0.056更高，说明额外状态可改变离线学习；其后仍有失败seed。当前结果只否定“在本配方中直接追加这五维即可可靠改善”的结论，不证明真实进度信息没有用途。

### 问题3：同时用于state+reward是否最好？

否。D的受保护AUC0.711、独立最终成功率0.784，均低于B；未发现正向协同。D的seed4仍低成功，seed2未稳定达到90%，不能将小best-final gap本身当作学会且稳定。

### 问题4：Curriculum是否进一步提高sample efficiency？

本轮未验证。D的raw最终平均0.481（门槛0.8）、最弱seed0（门槛0.6）、raw gap均值0.284（门槛≤0.1）、每seed在线成功0（门槛≥5），四项均未满足。按“D稳定后再启动”的条件，没有开展E。已预留10°/30°/50°/完整1rad、连续两次≥80%晋级以及部分目标专家轨迹前缀重标注接口，代码存在不代表实验已完成。

### 问题5：参数驱动方法是否减少后期退化？

保护后的部署文件可保持能力，B独立best0.991→final0.959，差0.031；附加0.01时0.988→0.938，差0.050。参数驱动D没有建立稳定更新和连续成功rollout。Raw gap有所降低伴随不同上限与失败seed，无法据此证明更新稳定。实际仍需要16.6–20.4次平均回退/run。

### 问题6：真机应使用哪些参数、怎样进入RL？

工装接口建议统一输出真实角/距离、速度、起始与目标、完成比例、剩余量和可靠接触；单位、时间戳与reset起点要一致。当前证据优先支持保留有符号进度reward、直接物理成功判定和进度保护作为候选；Actor状态追加需要继续受控验证。若接触传感不可靠，应独立消融，不用仿真接触位代替已验证的真机可观测信号。把手位姿不是本轮核心输入。

当前只测试完整目标1rad；输入中包含target不代表已学会多个目标。传感误差/延迟和新几何尚未验证。B最佳文件可用于仿真固定工位演示与下一轮对照，不能由其高成功率直接推导真实机器人在线学习已可用。

### 问题7：是否满足进入LWD/DIVL条件？

否。已有五seed完整测量和保护后能力保留；仍缺稳定成功的在线rollout、明确优于baseline的配对证据、进度组合的跨seed稳定、轻度新初态测试及Drawer复现。继续做经验筛选无法补足当前采集与更新问题，本轮没有实现LWD/DIVL/QAM。

## 9. 下一步受控验证建议

保留B作为候选和A作为对照。先用小规模检查对齐采集、Actor更新与目标Q采样的动作分布，记录双侧接触阶段的实际末端扰动及夹爪翻转概率；约束均值的BC loss之外，还要验证分布约束的作用。只改采集噪声却继续用宽分布Actor/Q目标，不能宣称解决整个闭环。

保护器以真实进度验收部署候选，并对学习器回退与不回退做独立小规模对照；拒绝把低成功初始模型作为“安全最佳”。先要求连续产生成功轨迹并保持final接近独立best，再按原条件开启课程、初角/把手/摩擦干预及Drawer复制。本报告提出下一步实验，不额外扩大本轮训练。

## 10. 吞吐与可复现产物

按用户要求，调度器从8个并行run扩到20个，每卡最多3个训练run，容量上限24槽位；本轮32环境/run，峰值640训练环境。每个run另有32环境的独立固定评估进程。已运行的8个训练进程原样接管，没有重启、改变学习参数或增加训练预算。独立best/final复测在run结束后流水执行，使用空出的GPU槽位。

`throughput_measurement.json`记录采样时8个未完成run最近10k区间的异步合计估计：训练约790 steps/s，包含它们的固定评估约7,636 steps/s；最快已记录单run区间约358训练steps/s。这些速率包含更新、评估等待和checkpoint I/O，不含随后独立复测，不能解释为仿真纯FPS或8卡硬件最大吞吐。报告主要比较交互次数，不用共享服务器上的墙钟时间宣称算法更快。

| 产物 | 位置（项目根目录下） |
|---|---|
| 主训练与配置 | experiments/phase8_progress_rl/、progress_rl/ |
| 进度/动作分布评价 | evaluation/progress_metrics.py |
| 检查点 | checkpoints/phase8_progress_rl/P8{{A–D}}_seed{{0–4}}/ |
| TensorBoard/进程日志 | logs/phase8_progress_rl/ |
| 逐回合/评估/诊断CSV | results/phase8_progress_rl/ |
| 五seed与配对统计 | summary.json、per_seed.csv、paired_tests.json |
| 独立复测 | heldout/、heldout_summary.json、heldout_status.csv |
| 原始候选与锚点审计 | update_diagnosis.json |
| 阶段决策与完成验收 | stage_decision.json、completion_manifest.json |
| 预检与缓存等价性 | preflight_audit.json、expert_cache_audit.json |

正式20run加载的代码保存为 `results/phase8_progress_rl/primary_source_snapshot.tar.gz`，SHA256=`247a028e10c7884ed3bf7ea026d027372db82b9ff59a7752124dbd267628768a`。之后的专家静态特征缓存优化经GPU逐字段bitwise检查，仅用于未来启动；在途训练保留原实现。独立测试新增分布诊断使用独立RNG，未改变执行噪声序列。汇总脚本适配现有NumPy1.26的梯形积分接口，没有改动训练结果。

统计复现命令（项目根目录执行）：

```bash
source configs/runtime_env.sh
for script in analyze analyze_heldout diagnose report_tables validate_results finalize_report; do
  .venv/bin/python "experiments/phase8_progress_rl/$script.py" || break
done
```

不要直接重复启动已存在checkpoint的训练；训练器和复测队列均检查唯一输出路径。

本轮完整交付的是进度接口、受控消融、保护与分布诊断、五seed统计和条件扩展框架。**观察到进度奖励候选的部署能力改善，尚未证明工装参数使在线RL更新稳定或普遍更高效。**
"""
    target = ROOT/"docs"/"phase8_progress_rl_report.md"
    target.write_text(prose, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
