# 精简权重：任务背景、模型索引和重建

## 1. 这是什么项目

项目研究：真实工装提供的环境状态、任务进度和自动reset，能否帮助Franka Panda稳定在线学习。平台为Isaac Sim5.1 / 固定Isaac Lab源码，最初在b300-2的8张B300上实验。

Phase1是抽屉打开，之后是Panda Door Opening。先测试GT直接输入Critic，再研究GT辅助表示、BC约束、进度奖励、探索噪声与Anchor保护，最后生成随机工位和失败恢复专家，比较参数条件BC与state-only π0.5。

原精简Release保存**80个任务checkpoint + 1个共享π0.5基础模型**，附件保持原样。2026-10-05追加独立历史Door Release，提供30个原始SAC/Privileged Critic/GT辅助E2的Best/Final；两份合计**110个任务checkpoint + 1个base = 111个模型**。它们不是全部历史模型，也不是重新训练结果。所有二进制保持原字节；原服务器、数据Release和科学报告没有被覆盖。

## 2. 保存哪些模型，为什么

| 组 | 数量 | 大小（未压缩） | 任务 / 用途 |
| --- | ---: | ---: | --- |
| Phase8 B Best | 5 | 17.81MiB | 固定Door，Progress Reward策略；Phase9实际使用的同seed冻结Anchor |
| Phase9 C Best + Final | 10 | 35.66MiB | 固定Door，Anchor + Progress Reward + std≤.01；复测稳定采样执行与Best/Final差距 |
| Phase13 A/B Best + 参考Anchor | 11 | 3.31MiB | 随机Door；Robot-only与Robot+GT的同容量BC对照；重新生成恢复数据/压力状态所需参考策略 |
| Phase14.3 A/B已选模型 | 10 | 3.01MiB | 随机Door；小型MSE BC，成功专家S与成功+恢复专家S+R的对照 |
| Phase6 B3 Best + Final | 10 | 35.67MiB | 固定Door；持续BC约束SAC，保留学会后退化与执行噪声诊断 |
| Phase14.3 C/D已选π0.5增量模型 | 10 | 1.07GiB | 随机Door；state-only π0.5，S与S+R数据对照 |
| Phase1 Drawer A/B/C Final | 24 | 77.99MiB | 抽屉；Robot-only、Privileged Critic、Privileged Policy各8seed终点对照 |
| 共享π0.5基础模型 | 1 | 13.47GiB | 所有10个π0.5微调模型共同依赖的一份已转换PyTorch base；本身不是Door策略 |
| Phase2 Door A Best + Final | 10 | 31.81MiB | 原始机器人观测BC+SAC，后期退化与普通Critic对照 |
| Phase2 Door B Best + Final | 10 | 32.67MiB | 完整GT Critic，直接GT路线与Q稳定性对照 |
| Phase3 E2 Best + Final | 10 | 35.67MiB | 前100k GT辅助监督，早期高成功率与500k退化对照 |

原81模型共15,799,223,865 bytes（14.71GiB），追加30模型105,023,832 bytes（100.16MiB），合计15,904,247,697 bytes（约14.81GiB）。压缩分卷的准确大小见`reproducibility/selected_weights_manifest.json`及`historical_door_weights_manifest.json`。同一个参考Anchor与别名checkpoint可能字节相同；保留原路径满足既有流程。

模型索引：[`weight_catalog.csv`](../reproducibility/weight_catalog.csv)，共111行。每行记录原路径、SHA256、任务/阶段/方案、seed、checkpoint实际训练步数、输入边界、数据来源、报告/指标/选模来源及依赖，`release_tag`区分来源。原81归档与CPU索引保持不变；历史补充对应`historical_door_weights_catalog.jsonl`、`historical_door_checkpoint_metadata.json`及`historical_door_metrics.csv`。详细历史背景、原奖励和逐seedBest/Final见[HISTORICAL_DOOR_WEIGHTS.md](HISTORICAL_DOOR_WEIGHTS.md)。

注意：Phase8/9 SAC载荷中的`variant=A`是沿用的网络结构标识，**不能据此把P8B或P9C误认为实验A**。真实实验方案由索引的`experiment_arm`、目录和阶段协议确定，`payload_variant`单独保留。

## 3. 关键结果与解释边界

数字为原报告的独立评测，不是在新服务器重新测得。

| 模型 | 固定工位 | 随机工位 | 人工恢复宏平均 | 自然严重偏离 |
| --- | ---: | ---: | ---: | ---: |
| Phase9 C | Best采样96.56%，Final采样97.50% | 与下列Level2协议不同，不混列 | — | — |
| Phase14.3 A：MSE BC + S | 100.00% | 84.22% [77.01,91.43] | 65.23% | 6.25% |
| Phase14.3 B：MSE BC + S+R | 85.94% | 70.31% [54.93,85.70] | 46.25% | 15.00% |
| Phase14.3 C：π0.5 + S | 100.00% | 89.69% [85.91,93.47] | 52.89% | 5.31% |
| Phase14.3 D：π0.5 + S+R | 94.06% | 81.25% [70.94,91.56] | 85.55% | 6.88% |

方括号为随机工位五训练seed的Student-t95%CI，完整SD/CI/配对检验见原报告。Phase9 C的采样成功率AUC=.9584，新在线成功轨迹5001/5044；同cap冻结Anchor的成功率约97.8%，因此不能把保持高成功率解释成已证明RL净提升。Phase6 B3的Best约95.9%、Final约31.9%，用于诊断而非部署。

Phase14.3的π0.5随机提升尚未达到校正后统计显著，自然严重偏离恢复仍弱，未通过可靠随机Anchor门槛。四组都使用相同Robot+GT、无RGB；这不是完整视觉VLA对照，也不能把不同容量、预训练、动作块和优化预算的配方差异单独归因于模型架构。

随机Level2为初角0–5°、工装XYZ±1cm。原门驱动仍会使初角趋向关闭，因此成绩不能解释为保持不同门角的全面泛化。所有模型均为仿真策略；该实验没有证明实时硬件部署。

## 4. 正确观测、动作和依赖

### Drawer / 固定Door SAC

Actor机器人输入26维：q9、qd9、EEF xyz3、四元数4、归一化episode时钟1。Drawer C额外使用11维工装GT；A/B的Actor不使用这些GT。Drawer任务`Isaac-Open-Drawer-Franka-IK-Rel-v0`，成功位移>.30m，480tick；Door成功角>1rad，600tick。原60Hz、7维归一化相对EEF xyz/轴角+夹爪动作保持不变。

SAC文件保留Actor、Critic、target Critic、优化器和温度等原状态，但不能仅靠checkpoint恢复replay、RNG、collector和仿真内部状态。它们支持模型复测和另起严格记录的continuation，不应声称原训练进程逐位续跑。

Phase9 C执行必须保留**pre-tanh std上限.01**，同时保留Progress Reward和原控制尺度。KL锚点必须是同seed的Phase8 B，不能在加载Final后把当前Actor复制成新的“原Anchor”。构造`AnchoredSAC`时先载Phase8 B，再`load_state`载Phase9状态；未提供原Replay/RNG时，新的微调需要单独记录协议。

### 随机工位BC

39个输入槽：robot26+environment13。Phase13 A把GT归一化槽置零，B使用测量GT；使用checkpoint内的mean/std和模式。Phase14.3 A/B两组都使用GT，其A/B标签与Phase13的A/B含义不同。

Phase14.3中13个有效环境字段为门角、角速度、progress、remaining、左右contact、handle xyz和四元数。目标角固定1rad；与Phase13相比，恒定target输入槽被角速度代替。不得把两个阶段的环境字段顺序混用。正确入口见`experiments/phase13_random_expert_bc/model.py`和`experiments/phase14_2_recovery_bc/model.py`。

### π0.5微调模型

10个C/D `.pt`文件保存28,707,872个可训练参数（512个tensor）：rank16 LoRA和训练过的action/time投影。它们**必须与共享base和项目适配代码组合**，不能作为完整独立网络直接加载。

依赖：固定OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`及已保留的源码改动/Transformers补丁、对应tokenizer、独立π0.5环境。输入使用checkpoint归一化，再atan映射/离散语言前缀；没有RGB。模型产生10步、32维内部动作，前7维为真实动作、其余为受监督零padding，执行前5步后重新规划；reset/恢复接管需清空旧chunk。使用10个Euler flow steps。

共享base恢复到`external_weights/pi05_base/model.safetensors`，SHA256为：

```text
0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
```

`pi05.model.build()`加载base、建立LoRA结构；`pi05.model.restore(net, task_checkpoint)`严格核对可训练参数集合后加载微调参数。不要以别的任务模型或任意版本base替代。

OpenPI/Gemma原始条款及修改说明见[`WEIGHT_NOTICE.txt`](WEIGHT_NOTICE.txt)、`third_party/openpi/LICENSE`和`LICENSE_GEMMA.txt`；Release附有同样的许可证。restore脚本将三份通知同时放入base目录。基础模型为本实验已使用的PyTorch格式转换副本，格式来源及具体源码版本已记录，不声明重新转换后必然逐位相同。

## 5. 另一台服务器怎样恢复

```bash
git clone https://github.com/du17183/privileged-rl-validation.git
cd privileged-rl-validation
# 原科学数据、指标、日志和依赖源码
python3 scripts/restore_release.py --groups all
# 全部111个已选模型，包含共享base与历史Door补充
python3 scripts/restore_weights.py --groups all
# CPU校验；不启动训练或GPU实验
python3 scripts/verify_export.py --data --weights
```

只需要原小模型时：`--groups core,context,drawer`；π0.5另外恢复`pi05_adapters,pi05_base`。新增历史组为`history_door_sac,history_door_privileged,history_gt_e2`，可单独下载；`all`恢复两份Release。分卷无需手工拼接；每卷及每个恢复模型都核对SHA256。已有正确文件可以复用，不同字节的现有文件默认拒绝覆盖。

原运行环境的重建见[`REPRODUCE.md`](REPRODUCE.md)。权重不进入Git对象，原81在[精简权重Release](https://github.com/du17183/privileged-rl-validation/releases/tag/selected-weights-phase1-14-3-20261004)，新增30在[历史Door Release](https://github.com/du17183/privileged-rl-validation/releases/tag/historical-door-weights-20261005)；代码、索引和说明进入Git。

独立复测Phase14.3时只生成所选20个模型的evaluation jobs，复用原test cohort、test reset seed=143502，使用已有`pi05.inference.server`及`evaluation.pi05_closed_loop_eval`接口。其job字段见`experiments/phase14_3_pi05_bc/run.py`的`job()`。不要启动训练总控来“评估”已经恢复的模型，也不要用test结果重新选checkpoint。

重新训练时用`prepare_run.py`创建空输出目录，避免恢复的completed标记跳过未包含的其他历史权重。恢复模型不代表恢复全部训练历史。

## 6. 选模与校验记录

每组保留全部原seed，不挑单seed高分。Phase14.3使用独立closed-loop validation已冻结的`selection.json`：A/B按1k/5k/10k/15k/20k候选，C/D按250/1250/2500/3750/5000候选；test从不参与选模。Phase13保留原validation MSE选择，不能以本次test重选。Phase6 Best来自原主评估曲线，Final为500k；Drawer为200k最终模型。

迁移任务已做原文件稳定性检查、每模型SHA256、gzip分卷和每个tar成员完整比对、80个任务模型CPU载荷读取及base safetensors头检查。上传后核对GitHub每个附件的state/size/SHA256，公开恢复测试的证据单独保存。迁移不启动新训练，不改reward/reset/任务，也不声称已在第二台服务器完成GPU复测。
