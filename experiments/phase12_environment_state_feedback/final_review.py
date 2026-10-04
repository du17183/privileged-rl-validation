"""Presentation-only QA amendment; original experiment sources and data remain fixed."""
import hashlib,json,zipfile
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,TAG

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if not json.loads((OUT/'phase12_completed.json').read_text())['completed']:raise RuntimeError('Pipeline incomplete')
 if not json.loads((OUT/'formal_data_audit.json').read_text())['passed']:raise RuntimeError('Data audit failed')
 if (OUT/'final_review.json').exists():raise RuntimeError('Final review already applied')
 manifest=json.loads((OUT/'training_source_manifest.json').read_text())
 changes=[name for name,sha in manifest.items() if digest(ROOT/name)!=sha]
 if changes:raise RuntimeError(f'Frozen source changes: {changes}')
 immutable={str(p.relative_to(ROOT)):digest(p) for p in (OUT/'summary.json',OUT/'per_seed.csv',OUT/'curves.json')}
 summary=json.loads((OUT/'summary.json').read_text());keys=list(summary['groups'])
 columns=['door_angle','door_angular_velocity','progress','remaining_angle','angle_family','contact_state','handle_position','all_feedback']
 array=100*np.array([[summary['input_masking'][k]['final'].get(c,{}).get('drop',{}).get('mean',np.nan) for c in columns] for k in keys])
 image=OUT/'figures/input_masking.png';image_before=digest(image)
 plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(11,6));im=ax.imshow(array,vmin=-30,vmax=30,cmap='coolwarm',aspect='auto')
 ax.set_xticks(range(len(columns)),columns,rotation=40,ha='right');ax.set_yticks(range(len(keys)),keys)
 for i in range(len(keys)):
  for j in range(len(columns)):
   if np.isfinite(array[i,j]):ax.text(j,i,f'{array[i,j]:+.1f}',ha='center',va='center',fontsize=8)
 ax.set_title('Input masking: intact minus masked success (percentage points)')
 fig.colorbar(im,ax=ax,label='Success drop (percentage points)');fig.tight_layout();fig.savefig(image);plt.close(fig)
 report=ROOT/'docs/phase12_environment_state_feedback_report.md';report_before=digest(report)
 note='''
## 10. 最终图表核对与遮罩解释补充

遮罩热图的单元格数值、色标和标签统一使用百分点；原先色标采用比例单位，已修正呈现，统计数据保持原值。

B_none/D_none的30个更新块全部被拒绝，新增Actor输入权重和动作敏感性为零；这些部署checkpoint在数学上未采用新增反馈通道。然而其重复遮罩复测仍有小幅成功率差异。这是重要的负对照：相同reset参数计划和动作随机种子不能保证接触仿真的重复执行完全一致，残余差异可能涉及数值路径或reset历史，当前尚未定位。不能把这种差异归因为新增通道的效果，也不能据单个未经校正的遮罩区间宣称传感器收益。

本阶段主结论保持：在当前固定配方与预算下，未观察到新增环境反馈相对robot-only的可靠提升。非零输入权重或局部动作变化仅支持函数依赖；要进一步确认执行依赖，需先通过无作用遮罩负对照及fresh-process重复测试，再开展预注册重训消融。本次不追加RL训练。
'''
 report.write_text(report.read_text(encoding='utf-8')+note,encoding='utf-8')
 if any(digest(ROOT/name)!=sha for name,sha in immutable.items()):raise RuntimeError('Statistics changed')
 record=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),passed=True,statistics_unchanged=True,immutable_statistics=immutable,
  original_frozen_sources_unchanged=True,original_report_sha256=report_before,reviewed_report_sha256=digest(report),
  original_mask_figure_sha256=image_before,reviewed_mask_figure_sha256=digest(image),
  review_source_sha256=digest(ROOT/'experiments'/TAG/'final_review.py'),scope='Figure colorbar unit correction and explicit null-control qualification')
 (OUT/'final_review.json').write_text(json.dumps(record,indent=2))
 archive=ROOT/'results/phase12_review_artifacts_v2.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for name in (f'experiments/{TAG}','environment_feedback',f'results/{TAG}'):
   for p in (ROOT/name).rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or 'ipc' in p.parts or p.name.endswith('.partial.json') or p.suffix in ('.pt','.h5'):continue
    z.write(p,str(p.relative_to(ROOT)))
  z.write(report,'docs/phase12_environment_state_feedback_report.md')
 bundle=dict(path=str(archive),size=archive.stat().st_size,sha256=digest(archive),created_at=datetime.now(timezone.utc).isoformat())
 (OUT/'review_bundle_v2.json').write_text(json.dumps(bundle,indent=2));print(json.dumps(bundle),flush=True)
if __name__=='__main__':main()
