"""Rebuild the final reports and postprocessing provenance, without training."""
import json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    for module in ['analyze','review','finalize']:
        subprocess.run([sys.executable,'-m','experiments.phase14_1_recovery_generation.'+module,'--run-id','formal_v1'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','experiments.phase14_1_recovery_generation.runtime_metadata'],cwd=ROOT,check=True)
    result=ROOT/'results/phase14_1_recovery_generation/formal_v1'
    natural=json.loads((result/'natural_crosscheck.json').read_text())
    failures=[x for x in natural['failures'] if x['case']=='natural_ee_offset']
    distances=[x['eef_distance_start'] for x in failures]
    margin=[x['evidence']['min_joint_margin'] for x in failures]
    extra=['','## 自然末端偏离的进一步证据','',
           f'5个自然末端负例接管时EEF到把手距离为{min(distances):.3f}–{max(distances):.3f}m，均剩余431个控制步（约7.18秒），均停在CLEAR阶段且没有进入OPEN。记录的URDF硬限位最小余量为{min(margin):.3f}–{max(margin):.3f}rad。',
           '', '人工末端扰动是在距把手4.5–11.5cm的接近阶段注入±5/10/20mm；这是不同于上述自然偏离的状态分布。当前证据不能把自然失败笼统归因于接管太晚、硬限位或GT无效；需要检查局部路径、姿态约束与真实接触阻塞。',
           '', '## 运行版本','',
           'Python3.11.15、PyTorch2.7.0+cu128、IsaacLab0.47.2、IsaacSim5.1.0.0、NumPy1.26.0、h5py3.16.0、SciPy1.15.3、Pinocchio2.7.0；8×NVIDIA B300，driver580.82.07。完整版本见runtime_environment.json。',
           '', '正式主动扰动671104次仿真交互，补充自然复测124736次；均为数据生成/专家验证，BC和RL优化步数为0。各case初始化之外的最长运行约567秒，四case并行；开发预检与正式预算分别记录。']
    report=ROOT/'docs/phase14_1_recovery_generation_report.md'
    report.write_text(report.read_text()+'\n'.join(extra)+'\n');(result/'report.md').write_text(report.read_text())
    paths=list((ROOT/'experiments/phase14_1_recovery_generation').glob('*.py'))
    paths+=list((ROOT/'recovery_expert/phase14_1').glob('*.py'))
    paths+=[ROOT/'recovery_expert'/name for name in ['collect_recovery.py','perturbation_generator.py','failure_classifier.py']]
    (result/'delivery_source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__=='__main__':main()
