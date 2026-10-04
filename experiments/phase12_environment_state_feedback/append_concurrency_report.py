"""Disclose measured evaluation overhead, adopted concurrency and rejected batching."""
import hashlib,json
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT

def main():
 amendment=json.loads((OUT/'throughput_amendment.json').read_text())
 changes=[p for p,h in amendment['source_manifest'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
 if changes:raise RuntimeError(f'Performance amendment source changed: {changes}')
 diagnosis=json.loads((OUT/'throughput_diagnosis.json').read_text())
 gate=amendment['passed_preflight']
 batch=json.loads((OUT/'throughput/batched_preflight.json').read_text())
 reference=json.loads((OUT/'throughput/serial_reference.json').read_text())
 interactions=sum(json.loads(p.read_text())['metrics']['eval_env_steps'] for p in (OUT/'throughput').glob('batch_*.json'))
 interactions+=gate['physical_interactions']+reference['physical_interactions']+batch['batched_interactions']
 path=ROOT/'docs/phase12_environment_state_feedback_report.md'
 lines=['','## 9. 吞吐诊断与调度调整','',
  f"前24次训练的评测时间占比中位数{100*diagnosis['median_eval_fraction']:.2f}%；训练7.2M交互时额外评测{diagnosis['sum_eval_steps']:,}交互。",
  'GPU utilization表示采样期间有kernel运行的时间比例，不能解释为SM/Tensor Core算力占用比例。小环境batch和大量评测/同步构成当前主要开销。',
  '最后两个训练改为立即运行，最多4训练作业/GPU。训练32环境、256batch、4updates、学习率、随机种子、模型、任务、reward、BC/Anchor/std/guard和评测次数均保持原设置。',
  f"独立复测最多6进程/GPU，每个进程仍执行冻结的heldout.py及原32环境evaluator。4个独立32环境进程的测量：{json.dumps(gate,ensure_ascii=False)}。并发背景训练变化与Kit启动开销使该测量不能作为总实验加速保证。",
  '256环境微测试吞吐约为32环境的6.4倍，但合并不同测试组未通过结果兼容性预检，未用于正式复测；原始32环境布局、RNG流、动作采样和episode预算保留。',
  f"拒绝的合并预检：{json.dumps(batch['checks'],ensure_ascii=False)}。这说明增加环境数可能影响接触仿真的数值路径，不能只依据速度替换评测。",
  f"已记录吞吐与兼容性测试额外物理交互{interactions:,}，全部排除于训练buffer、方法排名和训练交互预算。另两次失败的实验性预检初始化在第一tick之前终止，未产生完整计数；保留日志，未把它们计为零成本。",
  '原冻结源码hash无变更。追加调度实现另存hash清单。throughput目录保留完整原始测量；不同环境batch的success仅作运行检查，不作为方法收益。','']
 path.write_text(path.read_text(encoding='utf-8')+'\n'.join(lines),encoding='utf-8')
 (OUT/'throughput_amendment_audit.json').write_text(json.dumps(dict(passed=True,changed=[],recorded_benchmark_interactions=interactions,
  rejected_batched_evaluation=True,failed_initializations_have_incomplete_accounting=True),indent=2))
if __name__=='__main__':main()
