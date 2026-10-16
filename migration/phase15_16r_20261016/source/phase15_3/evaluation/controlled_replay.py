"""Four controlled development arms in one physics/inference pair, budget600."""
import ast,hashlib,json,os,sys
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R));src=R/'phase15_3/evaluation/replay.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='389cb2f66d7ade77910d2f22e3955463192908bf69fb1b5661937571ffa7c152'
tree=ast.parse(src.read_text());at=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
hook='''
import phase15_2.recovery_chain as candidate_chain
from phase15_3.reacquire.planner_candidates import install
frozen_action=candidate_chain.Reacquire.action
frozen_retreat=candidate_chain.retreat_command
original_set_job=Policy.set_job
def set_job(self,job):
    candidate_chain.Reacquire.action=frozen_action
    candidate_chain.retreat_command=frozen_retreat
    if 'phase15_3_arm' not in job:
        self.diagnostic_arm='initialization'
        return original_set_job(self,job)
    self.diagnostic_arm=job['phase15_3_arm']
    kind=job.get('phase15_3_candidate')
    if kind:
        destination=R/job['output'];destination.mkdir(parents=True,exist_ok=True);install(kind,destination)
    return original_set_job(self,job)
Policy.set_job=set_job
'''
tree.body[at:at]=ast.parse(hook).body
records=0
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dict' and any(k.arg=='q_pre' for k in n.keywords):
        n.keywords.append(ast.keyword(arg='arm',value=ast.parse('self.diagnostic_arm',mode='eval').body));records+=1
    if isinstance(n,ast.keyword) and n.arg=='source_split':n.value=ast.Constant('new Phase15.3 development')
assert records==1;ast.fix_missing_locations(tree)
exec(compile(tree,str(src)+'[controlled]', 'exec'),dict(__name__='phase15_3_controlled_replay',__file__=str(src)))
