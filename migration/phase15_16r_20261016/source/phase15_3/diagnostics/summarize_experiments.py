"""Typed experimental summary and scoped scientific figures from closed cases."""
from pathlib import Path
import csv,json,hashlib,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path.cwd();O=R/'phase15_3/results';A=O/'controlled_analysis';F=O/'figures';F.mkdir(exist_ok=True)
s=json.loads((A/'analysis_scope.json').read_text());stage=list(csv.DictReader((A/'stage_metrics.csv').open()))
h=json.loads((O/'handoff_audit/scope.json').read_text())['summaries'];rows=[]
for case,info in s['closed_cases'].items():
    rr={x['stage']:x for x in stage if x['case']==case and x['mode']=='raw' and x['distance_bin']=='-1'}
    v=h[case];final=rr['final_success']
    rows.append(dict(case=case,total_budget_ticks=info['budget'],episodes=info['episodes'],model_seeds=1,
        source_split='reused historical validation' if case=='probe600' or case.startswith('time') else 'unseen new confirmation' if case.startswith('confirm_') else 'new development',
        final_successes=info['successes'],final_rate=info['successes']/info['episodes'],
        exploratory_state_wilson95_low=float(final['exploratory_state_wilson95_low']),exploratory_state_wilson95_high=float(final['exploratory_state_wilson95_high']),
        safe_pose=int(rr['safe_pose']['coverage_n']),pregrasp_arrival=int(rr['pregrasp_arrival']['coverage_n']),valid_grasp=int(rr['valid_grasp']['coverage_n']),
        normal_handoff=int(rr['handoff']['coverage_n']),door_progress=int(rr['door_progress']['coverage_n']),
        observed_strict_first_handoff=v['observed_strict_first_handoff'],direct_planner_transition_normal=v['direct_planner_transition_normal'],
        recovery_bridge_episodes=v['observed_recovery_bridge_episodes'],mean_active_sim_ticks=v['mean_active_sim_ticks'],mean_original_terminal_ticks=v['mean_original_terminal_ticks'],
        total_simulated_clone_tick_interactions=v['total_simulated_clone_tick_interactions'],per_episode_wall_seconds=None,
        five_model_seed_variance=None,five_model_seed_95ci=None,formal_generalization_claim=False,policy_training=False))
with (O/'experiment_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for prefix,name in [('', 'development_stage_intervals'),('confirm_', 'confirmation_stage_intervals')]:
    wanted=[prefix+x for x in ['frozen_E','GT_direct','GT_projected_retreat','GT_pose_pregrasp']]
    cases=[c for c in wanted if c in s['closed_cases']]
    if not cases:continue
    endpoints=['safe_pose','pregrasp_arrival','valid_grasp','handoff','final_success']
    labels=['Safe pose','Pre-grasp arrival','Valid grasp','Normal handoff','Final opening']
    fig,axes=plt.subplots(1,len(endpoints),figsize=(15,3.1+len(cases)*.35),sharex=True,sharey=True)
    for ax,endpoint,label in zip(axes,endpoints,labels):
        for i,case in enumerate(cases):
            x=next(x for x in stage if x['case']==case and x['distance_bin']=='-1' and x['mode']=='raw' and x['stage']==endpoint)
            p=float(x['cumulative_rate']);low=float(x['exploratory_state_wilson95_low']);high=float(x['exploratory_state_wilson95_high'])
            ax.errorbar(p*100,i,xerr=[[max(0,p-low)*100],[max(0,high-p)*100]],fmt='o',capsize=4,color=plt.cm.tab10(i))
            ax.annotate(x['coverage_n']+'/'+x['source_n'],(p*100,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(label);ax.set_xlim(-5,105);ax.set_xticks([0,50,100]);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Coverage (%)')
    axes[0].set_yticks(range(len(cases)),[c.replace(prefix,'').replace('GT_','GT ').replace('_',' ') for c in cases]);axes[0].set_ylim(-.5,len(cases)-.5);axes[0].invert_yaxis()
    fig.suptitle(('Unseen confirmation' if prefix else 'New development')+': same states, 600 ticks, one model seed')
    fig.text(.02,.015,'Intervals: Wilson across 16 states, not five-model-seed CIs. Raw milestones may span attempts; final opening is a separate endpoint.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(F/(name+'.png'),dpi=160);fig.savefig(F/(name+'.svg'));plt.close(fig)
(O/'experiment_summary_scope.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),closed_cases=list(s['closed_cases']),
    input_analysis_sha256=hashlib.sha256((A/'analysis_scope.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    variance_scope='one model seed; no model-seed variance estimated',wall_time_scope='no per-episode wall measured; batch32 shared control wall is not divided by32',
    figures='16-state Wilson exploratory intervals; no formal significance claim',training=False),indent=2))
print(json.dumps(rows,indent=2))
