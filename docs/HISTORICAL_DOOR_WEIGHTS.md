# 历史Door权重补充：普通SAC、Privileged Critic与GT辅助E2

## 1. 保存目的与项目背景

本项目研究真实工装提供的环境GT、任务进度及自动reset如何帮助Franka Panda在线学习。原始实验在b300-2的8张NVIDIA B300上，使用Isaac Sim5.1及固定Isaac Lab源码进行。

2026-10-05新增30个历史checkpoint，保存项目最初的三组证据：普通BC+SAC出现退化；Critic直接输入GT没有稳定收益；GT辅助表示能形成较好的早期策略，但继续训练仍会失去能力。它们支持独立复测与价值/表示诊断，**不是当前稳定执行方法，也不是π0.5模型**。

这次迁移只读取原二进制、CSV和配置，不重新训练，不改任务、reward、reset，不用新评测选择模型。先前数据Release和81模型Release保持原样。新增包：[Historical Door weights](https://github.com/du17183/privileged-rl-validation/releases/tag/historical-door-weights-20261005)。新版恢复脚本默认同时恢复两批，共110个任务checkpoint和1个共享π0.5 base，合计111个模型。

## 2. 三组模型是什么

| 组 | 阶段 / 方法 | Actor输入 | Critic与GT路线 | 文件数 | 原字节 / MiB | 用途 |
| --- | --- | --- | --- | ---: | ---: | --- |
| `history_door_sac` | Phase2 A，BC初始化SAC | robot26 | robot26 | 10 | 33,360,096 / 31.81 | 原始Door机器人观测对照与后期退化 |
| `history_door_privileged` | Phase2 B，Privileged Critic | robot26 | robot26+GT11 | 10 | 34,261,216 / 32.67 | 直接GT Critic对照与Q诊断 |
| `history_gt_e2` | Phase3 E2，前100k GT辅助 | robot26 | Critic仅robot26；GT监督Actor encoder的预测头 | 10 | 37,402,520 / 35.67 | 早期表示收益、最佳策略与终点退化 |
| 合计 | 各组5seed×Best/Final | | | 30 | 105,023,832 / 100.16 | |

“普通SAC”不代表纯在线随机初始化：三组共享1000条成功专家、3000次BC初始化、1000次离线Critic更新，在线专家replay比例25%、BC约束权重10。各运行500k在线transition、32并行环境、batch256、每向量步4次SAC更新。原配置为`configs/door_experiment.json`；E2继承它，增加`configs/door_phase3.json`的GT辅助协议。

E2辅助权重0.1，目标为门角MSE和左右双指contact BCE。到100k后关闭辅助loss并冻结GT head。Actor和Critic的外部输入均不含GT；GT只作为训练标签。E2文件中的`variant=A`是网络路由标识，实际方案由`variant_label=E2`、目录和配置确定。不要把它误当Phase2 A。

## 3. 选模规则、逐seed文件与结果

Best按原周期评估成功率最大值选择，同分取最早步数；Final固定500000步。独立heldout只复核，不能用heldout重新选择Best。部分25k间隔因32环境批量计数得到75008、150016等精确步数，必须使用实际文件名。

| Seed | A Best步数 | A Best / Final成功率 | B Best步数 | B Best / Final成功率 | E2 Best步数 | E2 Best / Final成功率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 75008 | 90.625% / 0% | 50016 | 98.438% / 0% | 100000 | 100% / 0% |
| 1 | 425024 | 100% / 20.313% | 275008 | 100% / 0% | 150016 | 98.438% / 28.125% |
| 2 | 450016 | 62.5% / 6.25% | 100000 | 26.563% / 0% | 75008 | 100% / 0% |
| 3 | 150016 | 100% / 100% | 175008 | 25% / 0% | 125024 | 93.75% / 78.125% |
| 4 | 375008 | 100% / 100% | 400000 | 96.875% / 0% | 400000 | 98.438% / 0% |

原路径模板：

```text
/home/xiaolong/privileged_rl_validation/checkpoints/door/A_seed{seed}/step_{step}.pt
/home/xiaolong/privileged_rl_validation/checkpoints/door/B_seed{seed}/step_{step}.pt
/home/xiaolong/privileged_rl_validation/checkpoints/door_privileged_ablation/E2_seed{seed}/step_{step}.pt
```

换服务器后，restore按相同项目相对路径保存，项目根目录可改变。此包保留完整SAC checkpoint，不是`selected/E2/`中的actor-only导出。

| 方法 | 原训练成功率AUC | Best独立成功率 | Final独立成功率 | Best−Final |
| --- | ---: | ---: | ---: | ---: |
| Phase2 A | 0.183 | 90.625% | 45.313% | 45.313个百分点 |
| Phase2 B | 0.089 | 69.375% | 0% | 69.375个百分点 |
| Phase3 E2 | 0.257 | 98.125% | 21.25% | 76.875个百分点 |

每个checkpoint独立确定性评估64回合，reset seed=90000+训练seed，5seed均值。AUC是原0–500k在线交互曲线梯形积分/500k，**不是ROC AUC，也不是随机工位成功率**。原训练周期评估与新reset seed独立复测不同，因此报告中的训练终点和heldout终点会有不同值。

E2相对同encoder的Phase3 E0 AUC差+0.1084，bootstrap区间[0.0350,0.1817]，精确配对p=0.125，属于探索性证据。不能把跨阶段E2−Phase2 A的差当成GT单因素收益；E0匹配模型未包含在这次30文件补充包，E0代码、指标和原始数据已在仓库与数据Release中。Phase4新协议未确立稳定辅助窗口。因此保留E2不等于证明持续GT监督或持续在线更新有效。

原报告：[Door RL](door_rl_report.md)、[Privileged分析](privileged_analysis_report.md)、[后续稳定表示](stable_privileged_report.md)。所有结果为原实验值，本次未做GPU复测。

## 4. 原观测、动作、奖励与环境

- robot26：Panda关节位置9、速度9、相对环境原点EEF xyz3、EEF四元数4、episode时钟1。最后一维是episode时间比例，不是后来引入的门开度progress。
- GT11：门角1、角速度1、把手相对环境原点xyz3、四元数4、左右接触位2。接触定义为对应过滤力>0.5N；字段顺序按`door_env/door.py`。
- 动作7：归一化相对EEF平移xyz3、旋转轴角3、夹爪1；平移尺度0.05m、旋转尺度0.3rad；保持原控制频率60Hz。
- 成功门槛：door angle>1rad；每episode最多600tick。标称固定Panda Door工位，原reset/home、资产及专家数据不变。
- **原奖励**为`door_env.door.door_reward`：接近把手项 + 0.5×双指抓取 + 20×正门角速度 + 600×成功。不能用Phase8/9 Progress Reward替代后仍声称原实验复现。
- 原策略是tanh Gaussian，log_std裁剪[-5,2]；没有Phase9全流程std≤0.01，也没有Phase9 Anchor约束。确定性表格采用tanh(mean)，随机采样要另行报告。
- 不需要π0.5基础模型或OpenPI环境。使用主仿真环境：Python3.11.15、Torch2.7.0+cu128、Isaac Sim5.1及固定Isaac Lab源码，完整安装见[重建指南](REPRODUCE.md)。

## 5. 下载与完整性

只恢复这30个补充权重：

```bash
python3 scripts/restore_weights.py --groups history_door_sac,history_door_privileged,history_gt_e2
```

恢复全部111模型与原始数据：

```bash
python3 scripts/restore_release.py --groups all
python3 scripts/restore_weights.py --groups all
python3 scripts/verify_export.py --data --weights
```

restore对每个附件及每个模型验证SHA256，不同字节的既有文件默认拒绝覆盖。原81模型的分卷、catalog和manifest未变，三个新增组从独立Release读取。`verify_export.py --weights`检查两份catalog声明的全部111模型；只下载历史子集时使用restore自身的30文件校验，不能把其余81缺失当成历史包损坏。

逐模型背景与SHA见[`weight_catalog.csv`](../reproducibility/weight_catalog.csv)；原选择、heldout数值、文件指纹在[`historical_door_weights_catalog.jsonl`](../reproducibility/historical_door_weights_catalog.jsonl)与[`historical_door_metrics.csv`](../reproducibility/historical_door_metrics.csv)。分卷信息在`historical_door_weights_manifest.json`；CPU载荷键、张量shape和原参数在`historical_door_checkpoint_metadata.json`；上传与公开恢复验证另有对应JSON。

## 6. CPU加载与正确模型类

完整payload包含actor、critic、target_critic、Actor/Critic/温度优化器、log_alpha、env_steps、seed以及原训练配置等。CPU检查不启动仿真：

```python
from pathlib import Path
import torch
from algorithms.asymmetric_sac import GaussianActor
from auxiliary_learning.gt_prediction import EncodedGaussianActor

root = Path.cwd()  # 先进入重建后的项目根目录
path = root / "checkpoints/door_privileged_ablation/E2_seed0/step_100000.pt"
payload = torch.load(path, map_location="cpu", weights_only=False)
# A/B使用GaussianActor(26,7)；E2使用EncodedGaussianActor(26,7)。
actor = EncodedGaussianActor(26, 7)
actor.load_state_dict(payload["actor"], strict=True)
actor.eval()
print(payload["env_steps"], payload["seed"], payload.get("variant_label"))
```

Critic诊断：A构造`AsymmetricSAC(26,11,7,"A")`，B构造同尺寸`"B"`，E2构造`AuxiliarySAC(26,11,7,aux_weight=0)`。从payload加载相应Actor/Critic/target键，不能按同一个Actor类读取E2。A/E2的Q输入宽度33（robot26+action7），B为44（robot26+GT11+action7）。GT仅供B的训练/Q分析及E2监督/诊断，三组Actor推理均不接收GT。

这些checkpoint不含完整replay、RNG、collector和PhysX状态，不能凭它们逐位恢复原训练。新增微调须单独记录协议，E2的辅助开关还要按累计在线步数处理。

## 7. 独立仿真复测与输出隔离

原接口为`experiments.door.heldout_eval`（A/B，分别接收--mode best/final）和`experiments.door_privileged_ablation.heldout_eval`（E2，一次测best/final）。这些接口由原CSV解析checkpoint并写入当前项目的results，因此在**干净的新运行目录**中复测，保留原报告与结果。

```bash
python3 scripts/prepare_run.py --name historical-door-retest
```

进入新运行目录，用该目录中的restore脚本恢复三个历史权重组；原成功曲线CSV作为只读选择依据复制到新运行的相同results路径；保留原独立reset种子。每次传入单个seed，不用原训练launch脚本。A/B复测命令示例（在新运行目录内）：

```bash
source configs/runtime_env.sh
.venv/bin/python -m experiments.door.heldout_eval --variant A --seed 0 --mode best --device cuda:0 --headless
.venv/bin/python -m experiments.door.heldout_eval --variant A --seed 0 --mode final --device cuda:0 --headless
# B同理；E2的接口一次评测best和final。
.venv/bin/python -m experiments.door_privileged_ablation.heldout_eval --variant E2 --seed 0 --device cuda:0 --headless
```

选择CSV应来自恢复的数据Release，不从本次结果生成。跨驱动/GPU的分数不能保证逐位一致；本次公开下载/CPU/Hash验证不等于第二服务器GPU结果复现。
