"""Complete the saved-data report with explicit calendar wall-clock scope."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from experiments.phase16_r_residual_sac.common import ROOT,RES,save,sha,utc
from experiments.phase16_r_residual_sac.report_supplement_v3 import main as prior
from experiments.phase16_r_residual_sac.timing_completion_v1 import main as timing

def main():
    prior();timing()
    data=json.loads((RES/'calendar_timing.json').read_text())
    doc=ROOT/'docs/phase16_r_report.md'
    folder=RES/'report_revisions'/('calendar_'+utc().replace(':',''));folder.mkdir(parents=True,exist_ok=False)
    (folder/doc.name).write_bytes(doc.read_bytes())
    lines=['','## 11. 实验日历耗时','',
           '下列跨度由保存的调度与完成UTC记录计算，包含排队、其他项目负载和独立评测等待。它们有重叠，不可相加；不是GPU或CPU活动计算时间。','',
           '| scope | 起始UTC | 结束UTC | 日历wall s |','|---|---|---|---|']
    for row in data['rows']:lines.append(f"| {row['scope']} | {row['start_utc']} | {row['end_utc']} | {row['calendar_wall_s']:.0f} |")
    lines+=['','历史核验之前的人工工作耗时未记录，不能由文件mtime反推。训练活动wall、update/physics inclusive计时及eval pause另见training_throughput.csv；这里也不将并发试验的wall求和冒充总耗时。','']
    doc.write_text(doc.read_text()+'\n'.join(lines),encoding='utf-8')
    status=json.loads((RES/'report_status.json').read_text());status.update(report_sha256=sha(doc),audit_pending=True,supplement_source_sha256=sha(Path(__file__)));save(RES/'report_status.json',status)
    print(json.dumps(dict(report_sha256=sha(doc),physical_interactions_added=0)))
if __name__=='__main__':main()
