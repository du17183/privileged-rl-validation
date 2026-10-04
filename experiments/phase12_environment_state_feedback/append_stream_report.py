import hashlib,json
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT
def main():
 record=json.loads((OUT/'streaming_evaluation_amendment.json').read_text())
 changes=[p for p,h in record['source_manifest'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
 if changes:raise RuntimeError(f'Streaming scheduler source changed: {changes}')
 path=ROOT/'docs/phase12_environment_state_feedback_report.md'
 text='\n### 9.1 完成任务提前独立复测\n\n20:32发现4张GPU空闲、39次训练已完成后，允许完成的单个run提前独立复测。只根据completed标记及checkpoint/HDF5是否已落盘调度，不根据success或其他结果选择先后。GPU上存在任何其他compute进程（包括未完成训练）时不准入。冻结heldout.py、metrics.py、测试seed、32环境物理布局、Best/Final、mask、episode次数和采样协议均不变，训练作业未被迁移或重启。统计分析仍等待所有50次训练与50次完整复测结束，并执行原数据/旧结果/源码审计。该调整仅消除末批训练造成的空闲等待。\n'
 path.write_text(path.read_text(encoding='utf-8')+text,encoding='utf-8')
 (OUT/'streaming_evaluation_audit.json').write_text(json.dumps(dict(passed=True,changed=[]),indent=2))
if __name__=='__main__':main()
