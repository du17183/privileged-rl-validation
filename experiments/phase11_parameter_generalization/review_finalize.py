"""Preserve original delivery, append reviewed evidence, create fresh bundle."""
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime,timezone
from experiments.phase11_parameter_generalization.protocol import ROOT, OUT, CKPT, LOG, ARMS
from experiments.phase11_parameter_generalization.finalize_report import table,pct,ci
report=ROOT/'docs/phase11_parameter_generalization_report.md'
old_sha=hashlib.sha256(report.read_bytes()).hexdigest()
shutil.copy2(report,OUT/'report_before_review.md')
evidence=json.loads((OUT/'review_evidence.json').read_text())
audit=json.loads((OUT/'formal_data_audit.json').read_text())
summary=json.loads((OUT/'summary.json').read_text());groups=summary['groups']
manifest=json.loads((OUT/'training_source_manifest.json').read_text())
changes=[name for name,digest in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
if not evidence['passed'] or not audit['passed'] or changes:raise RuntimeError('Review gate failed')
subprocess.run([str(ROOT/'.venv/bin/python'),'-m','experiments.phase11_parameter_generalization.finalize_report'],cwd=ROOT,check=True)
content=report.read_text(encoding='utf-8')
old_label='完成episode成功率'
if old_label not in content:raise RuntimeError('Expected report table label missing')
content=content.replace(old_label,'成功率（5seed等权均值）')
content=content.replace('在线比例的分母仅是完成episodes；','上表成功率给各seed等权，不等于合计成功/合计episode；汇总比例另见复核表。在线比例的分母仅是完成episodes；')
supplement=[
    '## 14. 复核补充：分母、保护机制与证据限制',
    '### 额外置信区间',
    table(['组','随机Best 95%CI（%）','标称Best 95%CI（%）','标称Final 95%CI（%）','Best−Final 95%CI（pp）'],
          [[a,ci(groups[a]['best_success'],True),ci(groups[a]['nominal_best_success'],True),ci(groups[a]['nominal_final_success'],True),ci(groups[a]['gap'],True)] for a in ARMS]),
    '### 在线比例的两个统计口径',
    table(['组','合计成功/完成','汇总比例','5seed比例均值','score<0.1','score在[0.1,0.9)','score≥0.9'],
          [[a,f"{r['successes']}/{r['episodes']}",pct(r['pooled_success_ratio']),pct(r['seed_mean_success_ratio']),pct(r['pooled_low_score_lt_0p1']),pct(r['pooled_middle_score_0p1_0p9']),pct(r['pooled_high_score_ge_0p9'])] for a,r in evidence['quality_analysis'].items()]),
    '随机化让score≈1的seed均值占比降至约45.6%–50.7%，但分布主要是“近0失败”和“≥0.9成功”两峰，不是丰富连续进度层级。13,654条完成轨迹中仅1条score位于[0.1,0.9)。简单筛选仍接近成功/失败分类；必须防止仅优先easy success、排除困难状态的学习经验。',
    '### 更新拒绝来自哪些条件',
    table(['组','拒绝blocks','拒绝比例','仅标称触发','仅课程触发','两者均触发','被拒候选课程success≥80%'],
          [[a,f"{r['rejections']}/{r['blocks']}",pct(r['fraction']),r['nominal_only'],r['curriculum_only'],r['both'],r['rejected_candidates_with_curriculum_success_ge80']] for a,r in evidence['rejection_analysis'].items()]),
    '大部分拒绝由课程条件下降或regression触发；少数候选课程成功≥80%时仍被保护机制拒绝。不能把所有拒绝都归因于标称保护，也不能把拒绝次数直接当作Anchor因果效应。',
    'D的KL梯度norm均值830.503、RL梯度6.162，二者均值之比约135。参数列已学习非零权重，但angle/handle探针对归一化action的RMS响应只有约0.000876/0.000584。有参数响应，却没有足够的条件控制收益。证据支持进一步诊断约束与适应冲突；尚不能证明解除Anchor一定改善。',
    '19/20 runs最终仍在Level1；只有C seed0达到Level2，没有Level3训练暴露。因此不把+5°失败解释为充分Level2训练后环境参数必然无效。固定工位能力通过，但随机增益失败，传感噪声/输入遮蔽按预设条件暂缓。',
    f"复核验证{evidence['matched_condition_comparisons']}个Best/Final与冻结baseline的物理条件计划完全匹配：每seed/clone/episode的初角、XYZ位移、摩擦相同。NumPy1.26的AUC接口修复仅在后处理中采用同一梯形公式np.trapz；没有重跑训练或复测，首次失败日志保留。",
    '### 下一轮建议（尚未执行）',
    '用小规模严格配对消融分别检查“仅标称状态施加Anchor/有界环境参数残差”和“随机工位示范覆盖”，避免同时改两者。原Phase9 C继续用于固定工位；没有泛化证据前不进入完整LWD/DIVL、不宣称Drawer迁移效果。']
report.write_text(content+chr(10)*2+(chr(10)*2).join(supplement)+chr(10),encoding='utf-8')
marker=OUT/'phase11_completed.json';completion=json.loads(marker.read_text())
completion.update(review_passed=True,reviewed_at=datetime.now(timezone.utc).isoformat(),
    reviewed_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),initial_report_sha256=old_sha,
    matched_physical_plans=evidence['matched_condition_comparisons'],learning_source_changes=changes)
marker.write_text(json.dumps(completion,indent=2))
folders=['environment_state','randomized_env','curriculum','experiments/phase11_parameter_generalization']
files=[]
for name in folders:files += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [ROOT/'evaluation/stratified_generalization.py',ROOT/'evaluation/parameter_sensitivity.py',report,ROOT/'docs/phase11_parameter_generalization_protocol.md']
files += [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.png','.py','.md') and 'ipc' not in p.parts and p.name!='verified_review_bundle.json']
files += list(CKPT.glob('*/completed.json'))
files += [p for p in LOG.glob('*.log') if p.stat().st_size<2000000]
archive=OUT/'phase11_review_artifacts_verified.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
    for path in sorted(set(files)):stream.write(path,path.relative_to(ROOT))
bundle=dict(archive=str(archive),size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    reviewed_report_sha256=completion['reviewed_report_sha256'],reviewed_at=completion['reviewed_at'],
    full_checkpoints_datasets='Preserved on b300-2; review package excludes large binaries and HDF5')
(OUT/'verified_review_bundle.json').write_text(json.dumps(bundle,indent=2));print(json.dumps(bundle))
