# Phase15.2 阶段失败诊断

逐步状态在 *.stage_trace.h5：每10tick、Router/阶段变化、终点记录；状态为动作前，终点/阶段为动作后。
阶段率按 episode 和条件交集分别报告，允许 Recovery 路径跳过规划阶段；重新抓取不等于最终成功。

|方法|失败分类|数量|
|---|---|---|
|A|success|21|
|A|J_episode_timeout|89|
|A|A_retreat_failed|175|
|A|D_pregrasp_failed|122|
|A|J_episode_timeout_after_resumed_progress|88|
|A|H_Normal_refailure|5|
|B|success|9|
|B|J_episode_timeout|306|
|B|D_pregrasp_failed|73|
|B|J_episode_timeout_after_resumed_progress|95|
|B|H_Normal_refailure|4|
|B|I_progress_not_resumed|2|
|B|A_retreat_failed|11|
|C|success|9|
|C|J_episode_timeout|326|
|C|D_pregrasp_failed|68|
|C|J_episode_timeout_after_resumed_progress|63|
|C|G_handoff_failed|18|
|C|A_retreat_failed|13|
|C|H_Normal_refailure|3|
|D|success|9|
|D|J_episode_timeout|331|
|D|D_pregrasp_failed|68|
|D|J_episode_timeout_after_resumed_progress|55|
|D|G_handoff_failed|24|
|D|A_retreat_failed|13|
|E|success|14|
|E|J_episode_timeout|239|
|E|J_episode_timeout_after_resumed_progress|106|
|E|D_pregrasp_failed|101|
|E|H_Normal_refailure|8|
|E|G_handoff_failed|17|
|E|A_retreat_failed|14|
|E|I_progress_not_resumed|1|

所有源样本、失败日志、原 collision/600tick 约束保留。详见 metrics/stage.csv、failure.csv、router.csv、distance_bins.csv。
发生计划阶段超时、交还后停滞或 Normal 再次失败时，优先查看剩余预算、关节/相对姿态支持与阶段耗时。不能将规划失败归因于未执行的 Recovery 模型。

## 完整CPU与决策证据

[独立288诊断真实CPU报告](phase15_2_cpu_profile.md)与[17/18/13逐项回答](phase15_2_question_answers.md)保留同线程exclusive、分层分位数、shared/unknown与开销限制。正式300/30500数据和全部负结果未变；未启动新训练、训练数据采集或Phase16。
