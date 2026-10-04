"""MSE/success mismatch is diagnostic; it never changes formal selection."""
import json
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase14_2_recovery_bc.model import load
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/phase14_3_pi05_bc'
def main():
    torch.set_num_threads(4);rng=np.random.default_rng(143980);ds={}
    for src in ['base','recovery']:
        with np.load(ROOT/f'datasets/phase14_2/prepared_v1/{src}_validation.npz') as h:
            ids=rng.choice(len(h['robot']),2048,replace=False)
            ds[src]=(torch.tensor(np.concatenate((h['robot'][ids],h['environment'][ids]),-1)),torch.tensor(h['action'][ids]))
    steps=[1000,5000,10000,15000,20000];out={};fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,arm in zip(axes,['A','B']):
        out[arm]={};curves=[]
        for seed in range(5):
            losses=[];success=[];rows=[]
            for step in steps:
                net,mean,std,_=load(ROOT/f'checkpoints/phase14_3_pi05_bc/{arm}_seed{seed}_step{step}.pt','cpu')
                errors={}
                for src,(x,y) in ds.items():
                    with torch.no_grad():errors[src]=float(((net((x-mean)/std)-y)**2).mean())
                loss=errors['base'] if arm=='A' else (errors['base']+errors['recovery'])/2
                closed=json.loads((R/'validation'/f'{arm}_seed{seed}'/f'step{step}'/'random.json').read_text())['success']
                losses.append(loss);success.append(closed);rows.append(dict(update=step,validation_action_mse=errors,model_validation_mse=loss,closed_loop_random_success=closed))
            mse_step=int(np.argmin(losses));loop_step=int(np.argmax(success))
            corr=spearmanr(losses,success).statistic
            out[arm][str(seed)]=dict(checkpoints=rows,mse_best_update=steps[mse_step],random_best_update=steps[loop_step],
                success_gap_if_mse_selected=success[loop_step]-success[mse_step],spearman_mse_vs_success=float(corr) if np.isfinite(corr) else None)
            curves.append(success);ax.plot(steps,np.array(success)*100,alpha=.45,marker='o',label=f'seed{seed}')
        ax.set_title(arm+' MSE BC: closed-loop validation');ax.set_xlabel('Training update');ax.set_ylabel('Success (%)');ax.set_ylim(-5,105);ax.legend(fontsize=7)
    fig.tight_layout();dest=R/'figures';dest.mkdir(exist_ok=True);fig.savefig(dest/'mse_checkpoint_curve.png',dpi=180);plt.close(fig)
    (R/'mse_checkpoint_diagnosis.json').write_text(json.dumps(out,indent=2));print('MSE CHECKPOINT DIAGNOSIS COMPLETE',flush=True)
if __name__=='__main__':main()
