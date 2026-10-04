# 在另一台服务器重建与复现

## 范围

此仓库覆盖Phase1–14.3的代码、配置和报告。Release保留全部非权重数据、指标、逐episode结果、图表、日志和历史源码包；无效实验保留原有目录标记，不混入有效结论。

原数据Release不包含权重；原精简权重Release包含80个已选任务checkpoint和1份共享π0.5预训练base；2026-10-05独立补充30个早期Door SAC/Privileged Critic/GT辅助E2的Best/Final，两份权重Release合计111模型。背景及索引见[WEIGHTS.md](WEIGHTS.md)与[历史Door说明](HISTORICAL_DOOR_WEIGHTS.md)。虚拟环境、Python缓存、旧PID/socket/lock、嵌套Git对象和旧NVRTC备份不上传。原服务器文件未改动。两个旧交付压缩包中的权重已移除；修改后的hash及原始hash记录在`reproducibility/original_files.json`。

恢复精简权重后可直接复测所选模型；其余历史模型仍需重新训练。Git保存源码和索引，模型二进制只在Release。

## 1. 克隆并恢复所有数据、指标和日志

```bash
git clone https://github.com/du17183/privileged-rl-validation.git
cd privileged-rl-validation
python3 scripts/restore_release.py --groups all
python3 scripts/restore_weights.py --groups all
```

默认下载所有Release分卷，核对每卷SHA256，再流式恢复原目录。公开仓库无需token。下载使用小分块并支持HTTP Range断点续传，网络错误自动重试；已有正确分卷会跳过下载。现有恢复文件必须与归档字节一致，避免无意覆盖新实验。

分组：`datasets`、`results`、`logs`、`checkpoints`（只含CSV/JSON等元数据）、`third_party`（Isaac Lab源码）、`historical_and_door_data`（Door专家数据和历史交付包）。

权重分组：`core`（16）、`context`（30）、`drawer`（24）、`pi05_adapters`（10）、`pi05_base`（1）；历史补充为`history_door_sac`（10）、`history_door_privileged`（10）、`history_gt_e2`（10）。`--groups all`读取两份Release；只运行原小模型可先恢复`core,context,drawer`。每个恢复的模型也核对原文件SHA256，并保留原checkpoint路径。

原始数据/结果/日志约16GiB，下载缓存另占压缩分卷体积。Isaac、π0.5环境、预训练权重和新训练checkpoint需要额外空间。完整迁移建议预留100GiB以上；重跑所有历史实验还需为新checkpoint预留空间。

```bash
# 校验导出源码；不需要GPU，不会启动训练
python3 scripts/verify_export.py
# 恢复全部分组后，对所有归档文件做完整SHA256核对
python3 scripts/verify_export.py --data
# 恢复全部权重后，检查111个模型原文件SHA256
python3 scripts/verify_export.py --weights
```

## 2. Isaac运行环境

原实验：Linux x86_64、Python3.11、Isaac Sim5.1、Isaac Lab固定commit、Torch2.7.0+cu128、8×B300。详细包版本见`reproducibility/isaac-runtime-freeze.txt/json`，硬件/驱动见历史环境报告和provenance。

预先安装可用的NVIDIA驱动、Python3.11和uv；阅读并接受Isaac Sim许可条款后执行：

```bash
PYTHON_BIN=3.11 bash configs/setup_isaac.sh
source configs/runtime_env.sh
.venv/bin/python -m pip install 'numpy==1.26.0' 'h5py==3.16.0' 'scipy==1.15.3'
.venv/bin/python -m pip check
```

必须先恢复`third_party`分组。安装脚本在新`.venv`内重建B300所需的NVRTC12.9兼容替换，并按需获取libGLU。发行版和GPU不同可能需要适配系统依赖；不要通过修改reward/reset或训练预算解决安装问题。全部原始freeze是版本证据，不应直接照搬其中旧服务器的editable绝对路径。

每次Isaac进程运行前都需要`source configs/runtime_env.sh`。安装脚本末尾的preflight会启动仿真；它不训练策略。此上传任务未启动任何新的GPU实验。

## 3. π0.5单独的Python环境

Phase14.3使用与Isaac隔离的PyTorch环境，通过Unix socket进行推理通信：Torch2.14.0+cu130、Transformers4.57.6、JAX0.5.3。实际运行用到的OpenPI源码（包括本地改动）与Transformers模型源码已经保留；无需复制原RLinf项目。

```bash
PYTHON_BIN=3.11 bash scripts/setup_pi05.sh
```

脚本只修改新建`.venv_pi05`，安装使用uv的copy模式，避免后续模型源码/NVRTC替换影响共享uv缓存。Torch及torchvision必须使用匹配的CUDA13.0构建；安装源可通过`PI05_TORCH_INDEX`调整。若指定历史wheel不再可下载，应使用自己保存的同版本wheel或重新审定版本，不能把任意新版本宣称为严格复现。完整环境快照见`reproducibility/pi05-runtime-freeze.txt/json`。

精简权重Release已提供本实验使用的同SHA转换base，`restore_weights.py --groups all`会恢复它及许可证。也可另行获取：官方OpenPI列出的base是`gs://openpi-assets/checkpoints/pi05_base`；如果拿到的是JAX格式，需要按[官方PyTorch转换说明](https://github.com/Physical-Intelligence/openpi#converting-jax-models-to-pytorch)，在固定OpenPI版本的完整环境中使用`examples/convert_jax_model_to_pytorch.py`转换。不能直接把JAX文件改名为safetensors。本轮使用的是已经转换好的task-independent PyTorch base，放到：

```text
external_weights/pi05_base/model.safetensors
```

本实验使用base SHA256：

```text
0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
```

不要用其他任务的fine-tuned checkpoint替代。严格按同一二进制复现必须匹配上述SHA256；自行转换后的文件若hash不同，需核查转换版本、参数映射和dtype，不能未经核验宣称逐位一致。tokenizer是分词资源，不是策略权重，已保留，并有hash。可覆盖的路径变量：`OPENPI_SOURCE`、`PI05_BASE`、`PI05_TOKENIZER`、`PI05_PYTHON`；默认都指向本仓库内的对应位置。

## 4. 新实验目录与历史完成标记

恢复的结果包含旧实验完成标记，而精简包只含所选权重。**不能直接在历史目录运行旧训练总控脚本**：它可能依据完成标记跳过未包含的其他模型。重新训练必须创建干净的重跑目录；直接复测所选模型按WEIGHTS.md的评估接口执行。

```bash
python3 scripts/prepare_run.py --name phase14-3-retrain --dry-run
python3 scripts/prepare_run.py --name phase14-3-retrain
cd runs/phase14-3-retrain
source configs/runtime_env.sh
```

该操作复制当前源码，创建空的训练输出目录，共享只读数据/依赖/预训练权重位置，复用独立评估cohort，不启动训练。

Phase14.3可分步骤运行：

```bash
# 如已恢复派生chunks，此步骤可省略；需要重建时使用π0.5环境
.venv_pi05/bin/python -m pi05.data_adapter
# 单链路预检查，需GPU与已恢复的base
.venv_pi05/bin/python -m experiments.phase14_3_pi05_bc.preflight
# 总控：训练→validation选模→独立test；默认使用8GPU
.venv/bin/python -u -m experiments.phase14_3_pi05_bc.run
# 主流水线完成后，生成GT遮挡、动作诊断、统计与报告
.venv/bin/python -u -m experiments.phase14_3_pi05_bc.finalize
```

恢复的cohort含动作history、初始参数和接管状态，因此可以复用原独立压力分布。若重新生成cohort，精简包提供原Phase13参考Anchor；保持validation/test独立。不要用本次测试结果重新选超参数。

早期各阶段入口位于各自`experiments/`目录、`train/`和`evaluation/`中；阶段协议与完整报告见`docs/`。精简包提供Phase8 B及Phase13参考Anchor；依赖其余未包含前置模型的实验仍需先重训。

## 5. 结果解释与验证边界

Phase14.3是state-only π0.5 LoRA与单步MSE BC配方比较，不是完整视觉VLA对照。网络容量、预训练、状态量化、动作块和优化预算都不同。仿真等待推理输出，不证明实时硬件部署。

导出只修改部署路径和无`.git`依赖的provenance读取；diff见`reproducibility/source_portability.patch`。没有改动任务、reward、reset、动作/观测、训练loss、seed或预算。历史报告记录的是原服务器源码hash；导出源码hash另行记录。

迁移包已做源码语法、Git二进制排除、归档完整性、原文件SHA256、所选checkpoint CPU载荷及路径校验。**没有在第二台服务器重新跑完整训练**，跨驱动/GPU结果也不能保证逐位一致。
