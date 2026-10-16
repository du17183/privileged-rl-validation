# Phase15–16-R：b300-1 暂停实验迁移

快照日期：2026-10-16。Git 基线：`b5385c24dce5226e74862533397919e659434b32`。

本次仅整理、校验和发布已有文件。没有启动、恢复、终止训练、仿真、推理或数据采集；当前 `/proc` 核验没有活动训练/仿真/推理进程，两条历史监测/读取进程仍为 `T` 暂停状态。完整 argv、PPID 与出生时刻见 `final_source_and_pause_verified.json`。原始暂停记录和旧 lifecycle 按原字节保留，旧文件中的 `alive=true` 是当时的历史记录。

## 1. 包含内容

- Phase15、15.1、15.2、15.3 的最新代码、配置、cohort、原始结果、日志、图表、失败归档和审计记录。
- Phase16-R 的算法、评估、监督与协议源码；现有训练轨迹、Replay、checkpoint、验证/测试结果及暂停记录。
- 全部现有 `checkpoints/phase15*`、`checkpoints/phase16*` 文件，以及已解析到的冻结输入依赖。同 SHA 权重按清单去重存储，恢复时复制回全部原路径。
- 当前项目源码、工装资产、实际 OpenPI 源码、历史环境快照与当前两个 Python 环境的 pip freeze。
- 2026-10-16 奖励/Residual SAC 离线审计的脚本、完整结果和摘要。

共享 π0.5 Base 和固定 Isaac Lab 源码引用已有未改动的 Release；不会重新上传同一份大模型。Base 必须为：

```text
external_weights/pi05_base/model.safetensors
0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
```

Git 中的 `migration/phase15_16r_20261016/source/` 提供便于浏览的阶段源码子集。完整原路径文件以 Release 的 `migration_manifest.json` 为准。私有原始 `user_request` 文本直接文件不公开，保留其 SHA 和排除原因；原文仍在 b300-1。若以后需要重跑依赖这些原文的输入审计，需另行私下迁移。socket、锁、缓存及非普通运行时文件不作为可恢复状态导出。

## 2. 实验完成状态

| 阶段 | 实际状态 | 解释边界 |
| --- | --- | --- |
| Phase15 / 15.1 | 历史结果和完成/审计记录完整保存 | 不跨 cohort 直接计算方法增益 |
| Phase15.2 | 300 项正式测试、30,500 回合、288 独立 CPU 诊断；最终审计和全交付标记通过 | E 自然严重偏离 2.8%、Full Loop 82.34375%；没有达到可靠 Anchor 门槛 |
| Phase15.3 | 受控诊断与独立最终审计通过 | 单 model seed 的探索性结果；专家准入未过，未训练新的 Re-acquire 策略；不能把诊断完成称为恢复问题已解决 |
| Phase16-R | 用户暂停的未完成探索实验 | seed0 C/D 已到50k交互；seed1 部分完成；多 seed 最终比较未完成，没有证明 RL 净提升 |

Phase16-R 的基线是同 seed 的 Phase14.3 C Normal π0.5，不能把 Phase15.2 E 的恢复结果拼成同一条 RL 学习曲线。

历史 seed0 独立随机测试（每组128回合，sampled Residual 为主要结果）：

| 方法 | 成功数 | 成功率 |
| --- | ---: | ---: |
| 冻结 π0.5 SFT | 109 | 85.15625% |
| Residual SAC / 原奖励 | 108 | 84.375% |
| Residual SAC / 净进度奖励 | 107 | 83.59375% |

单 seed 的少量回合差异不能证明进度奖励造成稳定退化。净进度版本替换了原奖励中的正角速度项，不是简单追加奖励。

## 3. 克隆与恢复

需要 Python 3.11 和足够磁盘空间。完整新快照原始文件约14.12GB，压缩分卷约10.97GB；共享 Base、Isaac/PyTorch 环境、下载缓存和后续新结果需要额外空间，建议预留至少100GB。

```bash
git clone https://github.com/du17183/privileged-rl-validation.git
cd privileged-rl-validation
git checkout migration-phase15-16r-20261016

# 新目录；只下载、恢复与校验，不安装依赖、不启动实验。
python3 scripts/restore_phase15_16r.py \
  --destination /data/privileged_rl_snapshot_20261016 \
  --with-dependencies

# 之后可再次逐文件复核。
python3 scripts/restore_phase15_16r.py \
  --destination /data/privileged_rl_snapshot_20261016 \
  --verify-only
```

恢复脚本拒绝写入没有本快照标记的非空目录；遇到不同 SHA 的现有文件报错。每卷至多512MiB；先验证分卷 SHA，再验证 tar 每个成员和最终全部原文件 SHA。目录层级与清单之外的条目、路径越界和符号链接不会被直接解包。

`--with-dependencies` 通过已有恢复脚本下载同 SHA Base 和固定 Isaac Lab 源码；不会下载全部早期历史模型。省略该选项也可恢复本次新增快照，外部依赖是否就绪会单独列出。

## 4. 文件校验

本次清单 SHA256：

```text
10939fd5bfff9620354cef8dcfb124e09e8e22b389076523faf125903bb22567  migration_manifest.json
```

迁移元数据目录包含：逐文件清单、分卷 `SHA256SUMS`、权重路径/SHA索引、打包前后进程记录、源文件前后复核结果及 GitHub/独立恢复验证凭据。文件完整性通过不等于新的科学结论通过；本次没有重跑历史物理评估，也没有把未完成的 Phase16-R 标成完整审计通过。

## 5. A100 上的运行前检查

原版本为 Isaac Sim5.1、固定 Isaac Lab、两个独立 Python 环境。NVIDIA 官方要求列明 A100/H100 因无 RT Core 不受支持：[Isaac Sim5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。本快照没有声称已完成 A100 运行验证。

新服务器需先检查驱动、系统依赖、资产路径、无界面仿真、reset/contact/IK，再测冻结 SFT 与零 Residual。保留 H10/执行5步、60Hz/600tick、7维动作、冻结夹爪及原物理协议。原 B300 专用 NVRTC 替换、CUDA graph 缓存与65GB空闲显存门槛不能直接照搬至 A10040GB。

源码保留历史绝对路径用于 provenance。部署时应在新命名空间配置依赖路径并记录补丁；不要覆盖历史源文件或直接运行旧总控。checkpoint 不包含完整的在线仿真/收集器/RNG 状态，不能宣称原进程逐位续跑。

## 6. 后续修复建议（本包未实施）

2026-10-16 离线审计未发现奖励积分、相邻 Replay、动作合成或 checkpoint 加载的直接错误；发现需要受控验证的风险：Critic 同时读取未执行 residual 和裁剪后的执行动作；gamma=.99 在60Hz的有效时间尺度短；Actor 在首个完整回合前已更新；Q 的小 TD 误差不能证明回报校准正确；持续 residual 会扰动强 SFT。

后续在新目录逐项验证：执行动作 Critic、完整回合与 Critic 预热、折扣时间尺度、原奖励对净进度奖励、Residual 约束。先单 seed 短实验与中途评测，再依据趋势扩展。改变 Critic 架构或奖励协议后不能把旧优化器状态直接当成同一训练继续。LWD/DIVL 不随本迁移自动启动。
