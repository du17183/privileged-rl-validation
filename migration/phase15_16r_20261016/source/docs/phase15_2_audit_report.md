# Phase15.2 独立审计说明

正式完整性审计 passed；CPU独立288物理诊断已完成，12主worker及8CPU均exit0；本地全交付由最终marker单独核验。审计程序源码未修改。

原正式监督57078已退出，不重新启动。六个旧失败标记010/012/014/016/017/043来自H5排他创建技术失败或v1精确shadow失败；后来同任务在保留已完成结果后续跑，存在更新的真实complete marker和exit0。它们表示历史技术失败，不表示正式结果缺失。
原marker逐项SHA与当时日志/源码/jobs/报告已保存在analysis_v2_audit_markers，移动而非删除。原独立auditor重跑成功后才按原run.py完成逻辑保存experiment_complete；pipeline_failed原件亦保留，未伪造物理或成功率gate。

|检查|证据|
|---|---|
|正式300项/30500episode|原独立final_audit与50个真实任务complete；A–E各60项/6100episode|
|五seed/配对|seed0–4全部；A–E每seed原始reset/prefix/noise相同；不重选R2|
|episode唯一/完整/预算|每case原serialized telemetry唯一身份；固定64/random128/full128，其余100；全部<=600tick|
|数据源/泄漏|90cohort H5 SHA；660新自然160val+500test实际失败源唯一且零重叠；cohort_integrity_gate passed；evaluation-only/no training labels|
|checkpoint/source/历史|原frozen_inputs、有效code_manifest逐SHA复核；Phase1–15.1历史冻结文件未覆盖；10角色权重冻结|
|R2 SHA|e773e9fe20862e3bcf821e77507d2b9eca9b2211201f29fe0e5dfee6a14e6573|
|原始2350正式artifact SHA|技术marker修复前后逐项一致；保存archive_manifest.raw_formal_sha256|
|退出/后台|readiness_20261008T020056Z ready=true、live_processes=[]；84 lifecycle原worker记录exit0，v1/v2全部CPU实际退出；新增诊断调度器另记，不混为正式残留|
|统计|原65 aggregate五seedtCI+25 paired/Holm；补充45端点独立族；7个R2阶段为另列描述族|
|scope|21pilot仅技术；raw diagnostic false历史metadata bug不改变scope；只取results/test|

## 授权计算版本

E原042–049（seed1–4）共48 case使用v2；其他252case原串行。v1在clone29/tick162出现3.469446951953614e-18精确shadow差，容差未放宽。v2保留原完整串行撤离，只并行tracking/Normal前三候选checks，按clone原序提交且真实精确shadow。原202完整case未重算。v1 3.396× fixture不适用于v2/完整流水线。
code_manifest保留prior_activation历史；metadata旧version字符串须结合revision/activation及三个实际v2 SHA读取，不判作未授权漂移。

canonical audit为phase15_2/results/final_audit.json，用户要求镜像results/phase15_2/final_audit.json逐字节一致。补充所有报告后再重跑同一原auditor，更新funnel审计与本地交付SHA。CPU synthetic/native gate与真实288诊断分开记录，最终全交付以逐项本地SHA证明。

## CPU关闭与最终报告刷新

不改原auditor源码，在本次最终CPU与48问报告追加之后重跑；canonical与results/phase15_2镜像逐字节一致。独立诊断全部关闭、600tick/clock、冻结输入与真实pool exact shadow均通过。CPU跨grade细粒度归因与闭环开销缺证据项保持unknown，不以工具门槛或formal原complete代替全部交付。
