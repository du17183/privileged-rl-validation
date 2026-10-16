"""Static scientific figures from audited retrospective tables."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path.cwd();O=R/'phase15_3/results';F=O/'figures';F.mkdir(exist_ok=True)
a=list(csv.DictReader((O/'retrospective/stage_aggregate.csv').open()))
stages=['retreat_candidate','safe_pose','pregrasp_plan','pregrasp_arrival','recontact','valid_grasp','handoff','door_progress','final_success']
labels=['Retreat candidate','Actual safe pose','Pre-grasp plan','Pre-grasp arrival','Bilateral contact','Valid grasp','Normal handoff','Door progress','Final opening']
fig,ax=plt.subplots(figsize=(10,5.8));values=[int(next(x for x in a if x['method']=='E' and x['population']=='natural' and x['distance_bin']=='-1' and x['mode']=='raw' and x['stage']==s)['coverage_n']) for s in stages]
bars=ax.barh(np.arange(len(stages)),values,color='#356f9f');ax.invert_yaxis();ax.set_yticks(range(len(stages)),labels);ax.set_xlim(0,530);ax.set_xlabel('Observed episodes (source N=500; 5 model seeds)');ax.set_title('Phase15.2 E natural recovery: raw milestone coverage')
for b,v in zip(bars,values):ax.text(v+4,b.get_y()+b.get_height()/2,str(v),va='center')
fig.text(.02,.01,'Across attempts and bypasses; bars are not a single causal chain. Independent coverage, not conditional counts.',fontsize=9)
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(F/'natural_milestones.png',dpi=160);fig.savefig(F/'natural_milestones.svg');plt.close(fig)
rows=list(csv.DictReader((R/'phase15_2/results/metrics/funnel_v2_episode.csv').open()));fig,ax=plt.subplots(figsize=(9,5.4))
for s,label in [('safe_pose','Safe pose'),('pregrasp_arrival','Pre-grasp arrival'),('valid_grasp','Valid grasp'),('handoff','Handoff'),('final_success','Final opening')]:
 values=[]
 for bucket in range(4):
  rr=[x for x in rows if x['method']=='E' and x['population']=='graded' and int(x['distance_bin'])==bucket];assert len(rr)==500
  values.append(100*sum(x[s]=='True' for x in rr)/len(rr))
 ax.plot(range(4),values,'o-',label=label)
ax.set_xticks(range(4),['[0,5)','[5,10)','[10,15)','[15,20)']);ax.set_ylim(-2,104);ax.set_xlabel('Actual source distance (cm)');ax.set_ylabel('Raw milestone coverage (%)');ax.set_title('E artificial graded recovery (N=500 per bin; 5 seeds)');ax.legend();ax.grid(alpha=.2)
fig.text(.02,.01,'Natural states occupy only10-24cm. Artificial graded bins do not supply low-distance natural evidence.',fontsize=9)
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(F/'graded_milestones.png',dpi=160);fig.savefig(F/'graded_milestones.svg');plt.close(fig)
probe=R/'phase15_3/experiments/frozen600_seed1_first8_attempt003/motion_steps.jsonl'
if probe.exists():
 rr=[json.loads(v) for v in probe.read_text().splitlines()];fig,ax=plt.subplots(figsize=(9.5,5.5))
 for i in sorted(set(x['clone'] for x in rr)):
  xx=[x for x in rr if x['clone']==i];ax.plot([x['tick'] for x in xx],[x['door_post'] for x in xx],label='state '+str(i))
 ax.axhline(1,color='black',ls='--',label='Opening criterion >1rad');ax.set_xlabel('Original episode tick');ax.set_ylabel('Measured post-action door angle (rad)');ax.set_title('Phase15.3 first8 validation probe: frozen E, one model seed');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
 fig.text(.02,.01,'Exploratory development replay. All8 failed within600 total ticks; this is not the formal500-state result.',fontsize=9)
 fig.tight_layout(rect=(0,.035,1,1));fig.savefig(F/'probe_door_progress.png',dpi=160);fig.savefig(F/'probe_door_progress.svg');plt.close(fig)
print('Saved figures',str(F))
time_root=R/'phase15_3/experiments/time600_900_1200_seed1_first8_v1'
if (time_root/'budget900/complete.json').exists():
 budgets=[600,900]+([1200] if (time_root/'budget1200/complete.json').exists() else [])
 fig,axes=plt.subplots(1,len(budgets),figsize=(6*len(budgets),5.4),sharey=True,sharex=True)
 for ax,budget in zip(axes,budgets):
  rr=[json.loads(x) for x in (time_root/f'budget{budget}/motion_steps.jsonl').read_text().splitlines()]
  result=json.loads((time_root/f'budget{budget}/results/natural_severe.json').read_text())
  for i in range(8):
   xx=[x for x in rr if x['clone']==i]
   ax.plot([x['tick']+1 for x in xx],[x['door_post'] for x in xx],color=plt.cm.tab10(i),label=f'State {i}')
  ax.axhline(1,color='black',ls='--');ax.axvline(600,color='grey',ls=':');ax.set_title(f'{budget} ticks: {round(result["success"]*8)}/8 opening')
  ax.set_xlabel('Post-action original episode tick');ax.grid(alpha=.2)
 axes[0].set_ylabel('Measured door angle (rad)');axes[-1].legend(ncol=2,fontsize=8)
 fig.suptitle('Same-state total-budget diagnostic: one model seed, reused validation')
 fig.text(.02,.015,'Same measured initial states and trajectories before 600 ticks. 7 states opened at 618-717 ticks; not the formal 500-state rate.',fontsize=9)
 fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(F/'time_budget_counterfactual.png',dpi=160);fig.savefig(F/'time_budget_counterfactual.svg');plt.close(fig)
