# Phase15.2 用户问题逐项证据回答

正式300/30500与独立诊断288的scope严格分开。三套问题分别逐项回答，最终报告与审计SHA见delivery manifest。raw多次尝试里程碑不等于单次连续因果链，pre-action状态与post-action阶段/terminal分开。

## 2026-10-08新增17问

### 1. 哪种方案最好

A在随机、人工宏平均、新自然和Full Loop点估计最高，B接触丢失点估计最高；没有单一总体赢家。五seed双侧exact最小p=.0625，原25/补45/阶段7 Holm各为独立族，不支持exact p<.05。旧15.1自然35.25%是不同cohort，本轮配对A自然4.2%。

### 2. 固定是否≥95%

E均值98.4375%，五seed t95%CI[95.6938,101.1812]%；均值门槛通过，CI下限门槛通过。CI保留未裁剪值。

### 3. 随机是否≥90%

E均值83.1250%，五seed t95%CI[74.8199,91.4301]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 4. 人工是否≥85%

E均值63.0000%，五seed t95%CI[60.4591,65.5409]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 5. 自然是否严格>50%/偏好60%

E均值2.8000%，五seed t95%CI[0.9583,4.6417]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。偏好60%也失败。

### 6. Full Loop是否≥90%

E均值82.3438%，五seed t95%CI[70.1004,94.5871]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 7. 5–10cm断崖阶段

断崖跨Pregrasp到达、contact、valid及handoff多段，不能单一归于撤离。下表按源H5实际距离；自然只有10–24cm，无低距自然证据，不能用名义grade代替距离。

|实际源距离|N|safe|到达|contact|valid|handoff|final|
|---|---|---|---|---|---|---|---|
|[0,5)cm|500|500|395|395|395|330|96|
|[5,10)cm|500|500|270|190|150|55|0|
|[10,15)cm|500|500|50|15|10|0|0|
|[15,20)cm|500|500|5|0|0|0|0|

### 8. R2改善了什么

A→B模型出口320/500→469/500，实际safe 293→354，计划195→306，到达117→130，contact 149→149，handoff 114→110，最终21→9。前端覆盖改善，最终未改善；validation64.375% safe/0.625% final不是正式test率，正式不重选R2。阶段7族exact/Holm限制保留。

### 9. 最大失败类型

E自然最大末次互斥主类F2=122/486，随后F15=117/486、F7=115/486、F6=75/486。各方法主损失不同；F3零不证明无实测碰撞，F5零不证明数学可达。

### 10. Timeout占多少

预算耗尽是次级事件，不与F1–F15互斥主类相加。下表报告实际自然失败中的数量；F14为0不代表没有timeout。600tick终点不能证明timeout是唯一因果，也不能据此自动延长预算。

|方法|自然失败N|budget_exhausted次级|explicit_time_budget次级|F14唯一预算主类|
|---|---|---|---|---|
|A|479|479|91|0|
|B|491|491|307|0|
|C|491|491|329|0|
|D|491|491|331|0|
|E|486|486|240|0|

### 11. CPU串行几何占比

288个独立诊断已经关闭；以下为逐父线程实测份额，分母分别是该线程已观测exclusive wall/CPU和更宽process CPU。几何标签是已注入模块的下界，other_control/未注入几何unknown；不合成30500个正式episode的全局份额，不以batch32除32。

|任务|PID/TID|几何/观测wall|几何/观测CPU|几何thread CPU/process CPU|
|---|---|---|---|---|
|E|959513/138758855829312|2.443%|2.570%|2.214%|
|A|2111766/133090788378432|2.301%|6.478%|4.139%|
|B|2158540/137261947852608|2.647%|2.735%|2.385%|
|C|2158890/123467920791360|2.592%|2.672%|2.329%|
|D|2759676/127009501652800|2.656%|2.765%|2.367%|
|E_grades|2760066/130893714069312|2.317%|2.389%|1.729%|

原始cohort/stage/thread P50/P95/P99和count见cpu_context_timing.csv；未保存逐call，跨层池化quantile unknown，不能平均分位数。E_grades分类计时混合四grade，grade/距离/结果归因unknown；child CPU单列，nested inclusive不可相加。

### 12. CPU优先优化模块

按各父线程真实exclusive CPU排序如下，不能将inclusive retreat与内层collision/FK重复相加。若other_control领先，先补该范围的独立归因；已识别的collision/FK/IK/joint热点再针对重复构建、不变量缓存或批检查设计等价验证。tracking/Normal检查保留原clone提交顺序。

|任务|PID/TID|序|类别|count|exclusive CPU s|
|---|---|---|---|---|---|
|E|959513/138758855829312|1|other_control|517829|2540.250656|
|E|959513/138758855829312|2|tracking_check|104186|26.069732|
|E|959513/138758855829312|3|safe_retreat_planning|8400|22.967375|
|A|2111766/133090788378432|1|other_control|8223|32.370928|
|A|2111766/133090788378432|2|isaac_step|1200|13.036478|
|A|2111766/133090788378432|3|collision_check|9489|1.758545|
|B|2158540/137261947852608|1|other_control|514379|2473.339045|
|B|2158540/137261947852608|2|tracking_check|105311|27.357588|
|B|2158540/137261947852608|3|safe_retreat_planning|5290|21.820302|
|C|2158890/123467920791360|1|other_control|514688|2485.519545|
|C|2158890/123467920791360|2|tracking_check|105432|26.840872|
|C|2158890/123467920791360|3|safe_retreat_planning|5290|21.043655|
|D|2759676/127009501652800|1|other_control|514820|2543.584122|
|D|2759676/127009501652800|2|isaac_step|1200|33.923257|
|D|2759676/127009501652800|3|tracking_check|105574|27.621877|
|E_grades|2760066/130893714069312|1|other_control|1573071|7749.250791|
|E_grades|2760066/130893714069312|2|isaac_step|4800|118.304642|
|E_grades|2760066/130893714069312|3|tracking_check|253089|77.942939|

完整串行retreat操作wall为E自然2574.083732秒、四grade混合7803.258549秒，分别占更宽collector wall约84.03%/71.16%；这不是纯CPU份额。other_control合并profile/world/goal/forward等几何及控制范围，约2–6%的已命名几何份额只是下界，全部串行几何CPU占比仍unknown。优先独立拆分这一retreat/profile范围并验证重复构建/不变量缓存，再对tracking/joint和FK/collision设计等价批处理；完整注入开销unknown。详见 [CPU实测作用域](phase15_2_cpu_profile.md)。这些是未来工程方案，不改本轮冻结协议。dispatch包含IPC/wait，pure IPC未知；startup/shadow/串行retreat作用域分开。原生fixture约2.57%开销不代表闭环，闭环因果开销unknown；v1 3.396倍不外推。

### 13. π0.5独占推理延迟

10角色每100个独占GPU batch1样本。下表为五seed各自分位数的最小–最大范围（ms），不是池化分位数。

|角色|clock|P50范围|P95范围|P99范围|
|---|---|---|---|---|
|normal|model_s|48.721–50.111|48.755–50.140|48.858–50.169|
|normal|service_s|49.248–50.689|49.343–50.990|49.778–53.863|
|normal|rpc_s|49.562–51.048|49.748–51.488|50.122–54.177|
|recovery|model_s|57.554–58.716|57.599–58.749|58.018–58.820|
|recovery|service_s|58.113–59.342|58.226–59.749|58.679–59.875|
|recovery|rpc_s|58.465–59.779|58.608–60.189|59.090–60.334|

model_s是返回的forward wall；service/RPC各自作用域。预测/执行15、名义4Hz，显存allocated/reserved逐角色见CSV。CUDA-event、机器人端到端unknown。

### 14. Anchor资格

否。E仅fixed均值与下限通过，random/artificial/natural/full主要门槛失败，既定重复失败下降亦未通过；不冻结为Reliable Randomized Anchor。

### 15. Phase16资格

否。没有通过Anchor的Phase16/RL启动资格。仅未来满足完整门槛后考虑Frozen Anchor、KL约束RL/Residual SAC公平验证，本轮不启动训练。

### 16. 独立Re-acquire路线

证据支持把Severe Re-acquire作为后续独立候选子任务计划，同时保留撤离执行、到达与严格入口的多阶段问题。目标只返回Normal可接管状态，不负责完整开门；GT状态→pregrasp/valid-grasp/normal-entry，未来500–1000成功专家轨迹，同数据同test比较GT Planner、轻量state BC与pi0.5。当前不自动采训练数据、不训练新策略、不预设VLA最好。

### 17. LWD/DIVL条件

仍不具备LWD/DIVL条件。可靠随机Anchor、在线RL稳定正向增量及多质量在线经验均未成立。

## 2026-10-07此前18问

### 1. 哪种方案最好

A在随机、人工宏平均、新自然和Full Loop点估计最高，B接触丢失点估计最高；没有单一总体赢家。五seed双侧exact最小p=.0625，原25/补45/阶段7 Holm各为独立族，不支持exact p<.05。旧15.1自然35.25%是不同cohort，本轮配对A自然4.2%。

### 2. 随机是否≥90%

E均值83.1250%，五seed t95%CI[74.8199,91.4301]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 3. 人工是否≥85%

E均值63.0000%，五seed t95%CI[60.4591,65.5409]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 4. 自然是否严格>50%/偏好60%

E均值2.8000%，五seed t95%CI[0.9583,4.6417]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。偏好60%也失败。

### 5. Full Loop是否≥90%

E均值82.3438%，五seed t95%CI[70.1004,94.5871]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 6. 5–10cm断崖阶段

断崖跨Pregrasp到达、contact、valid及handoff多段，不能单一归于撤离。下表按源H5实际距离；自然只有10–24cm，无低距自然证据，不能用名义grade代替距离。

|实际源距离|N|safe|到达|contact|valid|handoff|final|
|---|---|---|---|---|---|---|---|
|[0,5)cm|500|500|395|395|395|330|96|
|[5,10)cm|500|500|270|190|150|55|0|
|[10,15)cm|500|500|50|15|10|0|0|
|[15,20)cm|500|500|5|0|0|0|0|

### 7. R2改善了什么

A→B模型出口320/500→469/500，实际safe 293→354，计划195→306，到达117→130，contact 149→149，handoff 114→110，最终21→9。前端覆盖改善，最终未改善；validation64.375% safe/0.625% final不是正式test率，正式不重选R2。阶段7族exact/Holm限制保留。

### 8. Safe Retreat是否主瓶颈

E自然模型出口470→实际safe 348，raw损失122，F2为122/486=25.10%，仍是重要瓶颈；graded低距safe都500/500，瓶颈因scope而异。模型clear不是完整物理安全。

### 9. Pre-grasp是否主瓶颈

E自然计划273→实际到达159，raw损失114；末次尝试F7=115/486，F6=75/486。规划可生成与物理到达必须分别解释，有限IK未解不证明数学不可达。

### 10. Re-grasp是否主瓶颈

E自然contact 181、valid 157；graded [0,5)→[5,10) contact395→190、valid395→150，支持重新接近/严格抓取问题。A/B历史stage5 proxy与CDE hold/probe/60mm不是同一分类器；actual arm/wrist collision unknown。

### 11. Handoff是否主瓶颈

E自然valid 157→handoff 128→持续门角119；graded低距valid395→handoff330，高一桶valid150→handoff55。实际Normal入口、transition与持续门角不同；E−D同时改变pregrasp与progress，不能单因素归因。

### 12. Timeout占多少

预算耗尽是次级事件，不与F1–F15互斥主类相加。下表报告实际自然失败中的数量；F14为0不代表没有timeout。600tick终点不能证明timeout是唯一因果，也不能据此自动延长预算。

|方法|自然失败N|budget_exhausted次级|explicit_time_budget次级|F14唯一预算主类|
|---|---|---|---|---|
|A|479|479|91|0|
|B|491|491|307|0|
|C|491|491|329|0|
|D|491|491|331|0|
|E|486|486|240|0|

### 13. CPU串行几何占比

288个独立诊断已经关闭；以下为逐父线程实测份额，分母分别是该线程已观测exclusive wall/CPU和更宽process CPU。几何标签是已注入模块的下界，other_control/未注入几何unknown；不合成30500个正式episode的全局份额，不以batch32除32。

|任务|PID/TID|几何/观测wall|几何/观测CPU|几何thread CPU/process CPU|
|---|---|---|---|---|
|E|959513/138758855829312|2.443%|2.570%|2.214%|
|A|2111766/133090788378432|2.301%|6.478%|4.139%|
|B|2158540/137261947852608|2.647%|2.735%|2.385%|
|C|2158890/123467920791360|2.592%|2.672%|2.329%|
|D|2759676/127009501652800|2.656%|2.765%|2.367%|
|E_grades|2760066/130893714069312|2.317%|2.389%|1.729%|

原始cohort/stage/thread P50/P95/P99和count见cpu_context_timing.csv；未保存逐call，跨层池化quantile unknown，不能平均分位数。E_grades分类计时混合四grade，grade/距离/结果归因unknown；child CPU单列，nested inclusive不可相加。

### 14. CPU优先优化模块

按各父线程真实exclusive CPU排序如下，不能将inclusive retreat与内层collision/FK重复相加。若other_control领先，先补该范围的独立归因；已识别的collision/FK/IK/joint热点再针对重复构建、不变量缓存或批检查设计等价验证。tracking/Normal检查保留原clone提交顺序。

|任务|PID/TID|序|类别|count|exclusive CPU s|
|---|---|---|---|---|---|
|E|959513/138758855829312|1|other_control|517829|2540.250656|
|E|959513/138758855829312|2|tracking_check|104186|26.069732|
|E|959513/138758855829312|3|safe_retreat_planning|8400|22.967375|
|A|2111766/133090788378432|1|other_control|8223|32.370928|
|A|2111766/133090788378432|2|isaac_step|1200|13.036478|
|A|2111766/133090788378432|3|collision_check|9489|1.758545|
|B|2158540/137261947852608|1|other_control|514379|2473.339045|
|B|2158540/137261947852608|2|tracking_check|105311|27.357588|
|B|2158540/137261947852608|3|safe_retreat_planning|5290|21.820302|
|C|2158890/123467920791360|1|other_control|514688|2485.519545|
|C|2158890/123467920791360|2|tracking_check|105432|26.840872|
|C|2158890/123467920791360|3|safe_retreat_planning|5290|21.043655|
|D|2759676/127009501652800|1|other_control|514820|2543.584122|
|D|2759676/127009501652800|2|isaac_step|1200|33.923257|
|D|2759676/127009501652800|3|tracking_check|105574|27.621877|
|E_grades|2760066/130893714069312|1|other_control|1573071|7749.250791|
|E_grades|2760066/130893714069312|2|isaac_step|4800|118.304642|
|E_grades|2760066/130893714069312|3|tracking_check|253089|77.942939|

完整串行retreat操作wall为E自然2574.083732秒、四grade混合7803.258549秒，分别占更宽collector wall约84.03%/71.16%；这不是纯CPU份额。other_control合并profile/world/goal/forward等几何及控制范围，约2–6%的已命名几何份额只是下界，全部串行几何CPU占比仍unknown。优先独立拆分这一retreat/profile范围并验证重复构建/不变量缓存，再对tracking/joint和FK/collision设计等价批处理；完整注入开销unknown。详见 [CPU实测作用域](phase15_2_cpu_profile.md)。这些是未来工程方案，不改本轮冻结协议。dispatch包含IPC/wait，pure IPC未知；startup/shadow/串行retreat作用域分开。原生fixture约2.57%开销不代表闭环，闭环因果开销unknown；v1 3.396倍不外推。

### 15. Anchor资格

否。E仅fixed均值与下限通过，random/artificial/natural/full主要门槛失败，既定重复失败下降亦未通过；不冻结为Reliable Randomized Anchor。

### 16. Phase16资格

否。没有通过Anchor的Phase16/RL启动资格。仅未来满足完整门槛后考虑Frozen Anchor、KL约束RL/Residual SAC公平验证，本轮不启动训练。

### 17. 独立Re-acquire路线

证据支持把Severe Re-acquire作为后续独立候选子任务计划，同时保留撤离执行、到达与严格入口的多阶段问题。目标只返回Normal可接管状态，不负责完整开门；GT状态→pregrasp/valid-grasp/normal-entry，未来500–1000成功专家轨迹，同数据同test比较GT Planner、轻量state BC与pi0.5。当前不自动采训练数据、不训练新策略、不预设VLA最好。

### 18. LWD/DIVL条件

仍不具备LWD/DIVL条件。可靠随机Anchor、在线RL稳定正向增量及多质量在线经验均未成立。

## 原恢复漏斗13问

### 1. 哪种方案最好

A在随机、人工宏平均、新自然和Full Loop点估计最高，B接触丢失点估计最高；没有单一总体赢家。五seed双侧exact最小p=.0625，原25/补45/阶段7 Holm各为独立族，不支持exact p<.05。旧15.1自然35.25%是不同cohort，本轮配对A自然4.2%。

### 2. R2改善了什么

A→B模型出口320/500→469/500，实际safe 293→354，计划195→306，到达117→130，contact 149→149，handoff 114→110，最终21→9。前端覆盖改善，最终未改善；validation64.375% safe/0.625% final不是正式test率，正式不重选R2。阶段7族exact/Holm限制保留。

### 3. 5–10cm断崖阶段

断崖跨Pregrasp到达、contact、valid及handoff多段，不能单一归于撤离。下表按源H5实际距离；自然只有10–24cm，无低距自然证据，不能用名义grade代替距离。

|实际源距离|N|safe|到达|contact|valid|handoff|final|
|---|---|---|---|---|---|---|---|
|[0,5)cm|500|500|395|395|395|330|96|
|[5,10)cm|500|500|270|190|150|55|0|
|[10,15)cm|500|500|50|15|10|0|0|
|[15,20)cm|500|500|5|0|0|0|0|

### 4. Pre-grasp是否主瓶颈

E自然计划273→实际到达159，raw损失114；末次尝试F7=115/486，F6=75/486。规划可生成与物理到达必须分别解释，有限IK未解不证明数学不可达。

### 5. Re-grasp是否主瓶颈

E自然contact 181、valid 157；graded [0,5)→[5,10) contact395→190、valid395→150，支持重新接近/严格抓取问题。A/B历史stage5 proxy与CDE hold/probe/60mm不是同一分类器；actual arm/wrist collision unknown。

### 6. Handoff是否主瓶颈

E自然valid 157→handoff 128→持续门角119；graded低距valid395→handoff330，高一桶valid150→handoff55。实际Normal入口、transition与持续门角不同；E−D同时改变pregrasp与progress，不能单因素归因。

### 7. Normal再失败原因

以下按funnel_v2_episode的5秒Normal事件定义累计，重复入口事件与唯一episode/末次F12不同；删失、成功竞争终点不得当作观察到未失败。样本与阶段证据支持入口/接管后持续推进困难，不能识别唯一物理原因。

|方法|再失败事件|完整观察分母|删失|成功竞争终点|
|---|---|---|---|---|
|A|5|5|88|21|
|B|4|4|97|9|
|C|4|4|63|8|
|D|0|0|55|8|
|E|23|23|107|13|

### 8. 新自然结果

E均值2.8000%，五seed t95%CI[0.9583,4.6417]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。

### 9. Full Loop稳定90%

E均值82.3438%，五seed t95%CI[70.1004,94.5871]%；均值门槛失败，CI下限门槛失败。CI保留未裁剪值。均值与下限失败，不能称稳定90%。

### 10. Anchor资格

否。E仅fixed均值与下限通过，random/artificial/natural/full主要门槛失败，既定重复失败下降亦未通过；不冻结为Reliable Randomized Anchor。

### 11. Phase16资格

否。没有通过Anchor的Phase16/RL启动资格。仅未来满足完整门槛后考虑Frozen Anchor、KL约束RL/Residual SAC公平验证，本轮不启动训练。

### 12. 独立Re-acquire路线

证据支持把Severe Re-acquire作为后续独立候选子任务计划，同时保留撤离执行、到达与严格入口的多阶段问题。目标只返回Normal可接管状态，不负责完整开门；GT状态→pregrasp/valid-grasp/normal-entry，未来500–1000成功专家轨迹，同数据同test比较GT Planner、轻量state BC与pi0.5。当前不自动采训练数据、不训练新策略、不预设VLA最好。

### 13. 推理与CPU工程瓶颈

10角色每100个独占GPU batch1样本。下表为五seed各自分位数的最小–最大范围（ms），不是池化分位数。

|角色|clock|P50范围|P95范围|P99范围|
|---|---|---|---|---|
|normal|model_s|48.721–50.111|48.755–50.140|48.858–50.169|
|normal|service_s|49.248–50.689|49.343–50.990|49.778–53.863|
|normal|rpc_s|49.562–51.048|49.748–51.488|50.122–54.177|
|recovery|model_s|57.554–58.716|57.599–58.749|58.018–58.820|
|recovery|service_s|58.113–59.342|58.226–59.749|58.679–59.875|
|recovery|rpc_s|58.465–59.779|58.608–60.189|59.090–60.334|

model_s是返回的forward wall；service/RPC各自作用域。预测/执行15、名义4Hz，显存allocated/reserved逐角色见CSV。CUDA-event、机器人端到端unknown。

CPU工程瓶颈与上述独占推理分开：按各父线程真实exclusive CPU排序如下，不能将inclusive retreat与内层collision/FK重复相加。若other_control领先，先补该范围的独立归因；已识别的collision/FK/IK/joint热点再针对重复构建、不变量缓存或批检查设计等价验证。tracking/Normal检查保留原clone提交顺序。

|任务|PID/TID|序|类别|count|exclusive CPU s|
|---|---|---|---|---|---|
|E|959513/138758855829312|1|other_control|517829|2540.250656|
|E|959513/138758855829312|2|tracking_check|104186|26.069732|
|E|959513/138758855829312|3|safe_retreat_planning|8400|22.967375|
|A|2111766/133090788378432|1|other_control|8223|32.370928|
|A|2111766/133090788378432|2|isaac_step|1200|13.036478|
|A|2111766/133090788378432|3|collision_check|9489|1.758545|
|B|2158540/137261947852608|1|other_control|514379|2473.339045|
|B|2158540/137261947852608|2|tracking_check|105311|27.357588|
|B|2158540/137261947852608|3|safe_retreat_planning|5290|21.820302|
|C|2158890/123467920791360|1|other_control|514688|2485.519545|
|C|2158890/123467920791360|2|tracking_check|105432|26.840872|
|C|2158890/123467920791360|3|safe_retreat_planning|5290|21.043655|
|D|2759676/127009501652800|1|other_control|514820|2543.584122|
|D|2759676/127009501652800|2|isaac_step|1200|33.923257|
|D|2759676/127009501652800|3|tracking_check|105574|27.621877|
|E_grades|2760066/130893714069312|1|other_control|1573071|7749.250791|
|E_grades|2760066/130893714069312|2|isaac_step|4800|118.304642|
|E_grades|2760066/130893714069312|3|tracking_check|253089|77.942939|

完整串行retreat操作wall为E自然2574.083732秒、四grade混合7803.258549秒，分别占更宽collector wall约84.03%/71.16%；这不是纯CPU份额。other_control合并profile/world/goal/forward等几何及控制范围，约2–6%的已命名几何份额只是下界，全部串行几何CPU占比仍unknown。优先独立拆分这一retreat/profile范围并验证重复构建/不变量缓存，再对tracking/joint和FK/collision设计等价批处理；完整注入开销unknown。详见 [CPU实测作用域](phase15_2_cpu_profile.md)。这些是未来工程方案，不改本轮冻结协议。dispatch包含IPC/wait，pure IPC未知；startup/shadow/串行retreat作用域分开。原生fixture约2.57%开销不代表闭环，闭环因果开销unknown；v1 3.396倍不外推。

288个独立诊断已经关闭；以下为逐父线程实测份额，分母分别是该线程已观测exclusive wall/CPU和更宽process CPU。几何标签是已注入模块的下界，other_control/未注入几何unknown；不合成30500个正式episode的全局份额，不以batch32除32。

|任务|PID/TID|几何/观测wall|几何/观测CPU|几何thread CPU/process CPU|
|---|---|---|---|---|
|E|959513/138758855829312|2.443%|2.570%|2.214%|
|A|2111766/133090788378432|2.301%|6.478%|4.139%|
|B|2158540/137261947852608|2.647%|2.735%|2.385%|
|C|2158890/123467920791360|2.592%|2.672%|2.329%|
|D|2759676/127009501652800|2.656%|2.765%|2.367%|
|E_grades|2760066/130893714069312|2.317%|2.389%|1.729%|

原始cohort/stage/thread P50/P95/P99和count见cpu_context_timing.csv；未保存逐call，跨层池化quantile unknown，不能平均分位数。E_grades分类计时混合四grade，grade/距离/结果归因unknown；child CPU单列，nested inclusive不可相加。

## 直接证据文件

五seed总表与各Holm族：aggregate.csv、paired.csv、funnel_paired_45.csv、r2_paired_stage7.csv。源H5距离、预算与事件：funnel_v2_episode.csv；CPU：独立analysis目录的五CSV与cpu_analysis.json；独占10角色：exclusive_latency_all_clocks.csv。所有输入SHA见本文件manifest。

[CPU完整报告](phase15_2_cpu_profile.md) · [独占延迟](phase15_2_latency_report.md) · [13阶段漏斗](phase15_2_recovery_funnel.md) · [F1–F15](phase15_2_failure_taxonomy.md)。
