import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.phase12_environment_state_feedback.protocol import OUT
def main():
 s=json.loads((OUT/'summary.json').read_text());curves=json.loads((OUT/'curves.json').read_text())
 folder=OUT/'figures';folder.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
 for ax,keys,title in [(axes[0],['A_strong','B_strong','C_strong','D_strong'],'State feedback (anchor=1)'),
    (axes[1],[f'B_{v}' for v in ('strong','medium','weak','none')],'B: anchor strength'),
    (axes[2],[f'D_{v}' for v in ('strong','medium','weak','none')],'D: anchor strength')]:
  for key in keys:
   data=list(curves[key].values());x=np.array([float(r['env_steps']) for r in data[0]])/1000
   y=np.array([[float(r['success']) for r in rows] for rows in data])
   ax.plot(x,y.mean(0),label=key);ax.fill_between(x,np.clip(y.mean(0)-y.std(0,ddof=1),0,1),np.clip(y.mean(0)+y.std(0,ddof=1),0,1),alpha=.12)
  ax.set(title=title,xlabel='Additional training interactions (k)',ylim=(0,1));ax.legend(fontsize=8);ax.grid(alpha=.2)
 axes[0].set_ylabel('Stochastic success (mean, +/-1 seed SD)');fig.tight_layout();fig.savefig(folder/'learning_curves.png');plt.close(fig)
 keys=list(s['groups']);fig,axes=plt.subplots(1,2,figsize=(14,5))
 for ax,metric,title in [(axes[0],'final_success','Independent random final success'),(axes[1],'rollback_fraction','Rejected update blocks')]:
  for i,key in enumerate(keys):
   m=s['groups'][key][metric];ci=m['ci95'];ax.errorbar(i,m['mean'],yerr=[[m['mean']-ci[0]],[ci[1]-m['mean']]],fmt='o',capsize=3)
   ax.scatter([i]*5,m['values'],s=12,alpha=.45)
  ax.set_xticks(range(len(keys)),keys,rotation=45,ha='right');ax.set_title(title);ax.set_ylabel('Fraction (seed t95 CI)');ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(folder/'endpoints_and_rollback.png');plt.close(fig)
 conditions=['nominal','level2','angle_zero','angle_low','angle_high','position_zero','position_halfcm','position_onecm','position_shell_halfcm','position_shell_onecm']
 array=np.array([[s['generalization'][k]['final'][c]['policy']['mean'] for c in conditions] for k in keys])
 fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(array,vmin=0,vmax=1,cmap='viridis',aspect='auto')
 ax.set_xticks(range(len(conditions)),conditions,rotation=40,ha='right');ax.set_yticks(range(len(keys)),keys)
 for i in range(len(keys)):
  for j in range(len(conditions)):ax.text(j,i,f'{100*array[i,j]:.1f}',ha='center',va='center',color='white' if array[i,j]<.6 else 'black',fontsize=8)
 ax.set_title('Independent final success (%): conditional buckets, nominal physics');fig.colorbar(im,ax=ax,label='Success fraction');fig.tight_layout();fig.savefig(folder/'generalization_buckets.png');plt.close(fig)
 columns=['door_angle','door_angular_velocity','progress','remaining_angle','angle_family','contact_state','handle_position','all_feedback']
 array=np.array([[s['input_masking'][k]['final'].get(c,{}).get('drop',{}).get('mean',np.nan) for c in columns] for k in keys])
 fig,ax=plt.subplots(figsize=(11,6));im=ax.imshow(array,vmin=-.3,vmax=.3,cmap='coolwarm',aspect='auto')
 ax.set_xticks(range(len(columns)),columns,rotation=40,ha='right');ax.set_yticks(range(len(keys)),keys)
 for i in range(len(keys)):
  for j in range(len(columns)):
   if np.isfinite(array[i,j]):ax.text(j,i,f'{100*array[i,j]:+.1f}',ha='center',va='center',fontsize=8)
 ax.set_title('Input masking: intact minus masked success (percentage points)');fig.colorbar(im,ax=ax,label='Success drop');fig.tight_layout();fig.savefig(folder/'input_masking.png');plt.close(fig)
if __name__=='__main__':main()
