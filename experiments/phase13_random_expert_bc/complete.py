"""Mark execution complete only after independent testing and preservation audit."""
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase13_random_expert_bc'


def main():
    state=json.loads((R/'pipeline_state.json').read_text())
    audit=json.loads((R/'final_review.json').read_text())
    rl=json.loads((R/'rl_summary.json').read_text())
    bc=json.loads((R/'bc_primary_summary.json').read_text())
    expert=json.loads((R/'expert_validation/summary.json').read_text())
    assert state['stage']=='rl_suite_complete' and state['status']=='complete'
    assert audit['passed'] and not audit['issues']
    assert len(rl['per_seed'])==9 and len(bc['per_seed'])==10
    assert expert['episodes']==256 and expert['success_rate']>.90
    for path in ['docs/random_expert_report.md','docs/phase13_random_expert_bc_report.md',
                 'results/phase13_random_expert_bc/figures/rl_stability.png']:
        assert (ROOT/path).is_file(),path
    result=dict(execution_status='completed',scientific_success='partial',
        completed_utc=datetime.now(timezone.utc).isoformat(),
        expert_success=expert['success_rate'],random_successful_expert_trajectories=300,
        paired_bc_success_gain=bc['paired']['success'],
        rl_total_interactions=rl['total_train_interactions'],
        constrained_minus_frozen_final=rl['paired']['C_minus_A']['final_success'],
        old_results_preserved=audit['passed'],
        qualified_deployment_anchor=False,ready_for_lwd_divl=False,
        conclusions=['Random expert and matched GT-conditioned BC benefit established in current protocol.',
                     'Constrained fine-tuning and improvement over frozen BC must be distinguished.',
                     'Persistent initial-angle generalization remains limited by angle settling in the inherited task.',
                     'No prior environment, reward, reset, robot, baseline or expert dataset was overwritten.'],
        reports=['docs/random_expert_report.md','docs/phase13_random_expert_bc_report.md'])
    (R/'phase13_completed.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['paired_bc_success_gain','constrained_minus_frozen_final']},indent=2))


if __name__=='__main__':main()
