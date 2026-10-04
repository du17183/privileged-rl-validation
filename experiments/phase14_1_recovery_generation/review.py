"""Final preservation checks and scientific figures, without further rollouts."""
import argparse,hashlib,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    result=ROOT/'results/phase14_1_recovery_generation'/a.run_id
    base=json.loads((ROOT/'results/phase14_recovery_bc/baseline_inventory.json').read_text())
    changed=[]
    for name,digest in base['sources'].items():
        file=ROOT/name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:changed.append(name)
    for category in ['artifacts','phase13_artifacts']:
        for name,original in base[category].items():
            file=ROOT/name
            if not file.exists():changed.append(name);continue
            stat=file.stat()
            size=original['size'] if isinstance(original,dict) else original[0]
            ns=original.get('mtime_ns') if isinstance(original,dict) else original[1]
            if stat.st_size!=size or (ns is not None and stat.st_mtime_ns!=ns):changed.append(name)
    old=json.loads((ROOT/'results/phase14_recovery_bc/data_manifest.json').read_text())
    for name,metadata in old['dataset_files'].items():
        file=ROOT/'datasets/recovery_expert/validation_v1'/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=metadata['sha256']:changed.append(str(file.relative_to(ROOT)))
    for metadata in [old['source_expert_dataset'],old['prefix_checkpoint']]:
        if hashlib.sha256((ROOT/metadata['path']).read_bytes()).hexdigest()!=metadata['sha256']:changed.append(metadata['path'])
    preservation=dict(checked_sources=len(base['sources']),checked_artifacts=len(base['artifacts']),
                      checked_phase13_artifacts=len(base['phase13_artifacts']),changed=changed)
    (result/'preservation_audit.json').write_text(json.dumps(preservation,indent=2))
    figdir=result/'figures';figdir.mkdir(exist_ok=True)
    names=['ee_offset','contact_loss','door_regression','stagnation']
    summaries=[json.loads((ROOT/'datasets/recovery_expert/phase14_1'/a.run_id/n/'summary.json').read_text()) for n in names]
    y=np.array([s['success_rate'] for s in summaries]);interval=np.array([s['wilson95'] for s in summaries])
    fig,ax=plt.subplots(figsize=(8,4.5));ax.bar(names,y*100,color=['#2563eb','#16a34a','#d97706','#7c3aed'])
    ax.errorbar(names,y*100,yerr=np.array([y-interval[:,0],interval[:,1]-y])*100,fmt='none',color='black',capsize=4)
    for i,s in enumerate(summaries):ax.text(i,min(104,s['success_rate']*100+4),f'{s["success_count"]}/{s["valid_attempts"]}',ha='center')
    ax.axhline(90,color='#dc2626',linestyle='--',label='90% point-estimate gate')
    ax.set_ylim(0,111);ax.set_ylabel('Recovery success (%)');ax.set_title('Phase14.1: actual injected disturbances')
    ax.legend(loc='lower right');fig.tight_layout();fig.savefig(figdir/'expert_recovery.png',dpi=180);fig.savefig(figdir/'expert_recovery.pdf');plt.close(fig)
    buckets=summaries[0]['buckets'];labels=list(buckets);n=np.array([buckets[k]['n'] for k in labels]);success=np.array([buckets[k]['success'] for k in labels])
    fig,ax=plt.subplots(figsize=(12,4));ax.bar(labels,100*success/np.maximum(1,n),color='#2563eb');ax.axhline(90,color='#dc2626',linestyle='--')
    ax.tick_params(axis='x',rotation=60);ax.set_ylim(0,105);ax.set_ylabel('Recovery success (%)');ax.set_title('EEF axis/sign/amplitude coverage')
    fig.tight_layout();fig.savefig(figdir/'eef_buckets.png',dpi=180);fig.savefig(figdir/'eef_buckets.pdf');plt.close(fig)
    report=ROOT/'docs/phase14_1_recovery_generation_report.md'
    protocol=json.loads((result/'protocol.json').read_text())
    addition=['','## 复核与资源','',
              f'历史源文件{preservation["checked_sources"]}个、此前结果{preservation["checked_artifacts"]}个、Phase13产物{preservation["checked_phase13_artifacts"]}个；检测到变化{len(changed)}。',
              '', f'每进程32环境，至多{protocol.get("workers",2)}进程并发，各4 CPU线程，nice10，与其他项目共享GPU，无抢占。全部计入的是仿真数据生成交互；BC/RL优化步数均为0。',
              '', f'![恢复成功率](../results/phase14_1_recovery_generation/{a.run_id}/figures/expert_recovery.png)',
              '', f'![末端偏离分桶](../results/phase14_1_recovery_generation/{a.run_id}/figures/eef_buckets.png)']
    report.write_text(report.read_text()+'\n'.join(addition)+'\n')
    print(json.dumps(preservation))


if __name__=='__main__':main()
