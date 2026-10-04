import hashlib,json
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT
def main():
 a=json.loads((OUT/'throughput_amendment.json').read_text());changes=[p for p,h in a['source_manifest'].items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
 if changes:raise RuntimeError(f'Performance amendment source changed: {changes}')
 diagnosis=json.loads((OUT/'throughput_diagnosis.json').read_text());gate=a['passed_preflight']
 interactions=sum(json.loads(p.read_text())['metrics']['eval_env_steps'] for p in (OUT/'throughput').glob('batch_*.json'))+gate['serial_interactions']+gate['batched_interactions']
 path=ROOT/'docs/phase12_environment_state_feedback_report.md'
 text=path.read_text(encoding='utf-8')
 lines=['','## 9. 吞吐调整及额外测量成本','',
  f"已完成24次训练的实测中，评测时间占比中位数{100*diagnosis['median_eval_fraction']:.2f}%；训练7.2M交互时已额外评测{diagnosis['sum_eval_steps']:,}交互。",
  '用户要求提高吞吐后，仅调整作业调度和独立评测并发。原训练32环境、256batch、4updates、种子、模型、reward、BC/Anchor/std/guard均不变。原冻结源码hash无变更，追加性能实现另有hash清单。',
  '最后两个训练从等待改为立即运行。独立评测把8个原32-clone测试组同时执行，总256环境/进程；每组仍以自己的32个原克隆RNG流、相同episode次数、相同reset分布完成。跨条件共享的CUDA高斯噪声仍按32×7逐tick生成并重复，未降低随机策略std。',
  f"预检结果：{json.dumps(gate['checks'],ensure_ascii=False)}。同一物理初态参数计划逐条匹配；克隆世界坐标排列和浮点执行路径存在数值差别，预检按≤2/64 episode成功差容差判断，不承诺bitwise轨迹一致。",
  '每GPU最多6个独立评测进程。总体episode预算不变；完成较早的组与padding继续仿真的交互也计入physical总成本，不能把有效样本数当作全部物理交互。',
  f"吞吐测试与兼容性预检额外物理交互{interactions:,}，不进入训练buffer，不参与方法排名。throughput/*.json保留原始测量。",
  '吞吐微测试的不同batch使用不同克隆集合，其success只用于运行检查，不能比较为策略收益。性能修改发生在独立heldout开始之前，所有50组使用同一实现；训练AUC仍由原有validation流程取得。','']
 path.write_text(text+chr(10).join(lines),encoding='utf-8')
 (OUT/'throughput_amendment_audit.json').write_text(json.dumps(dict(passed=True,changed=[],benchmark_and_preflight_interactions=interactions),indent=2))
if __name__=='__main__':main()
