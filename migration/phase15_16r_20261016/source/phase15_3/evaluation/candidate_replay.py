"""Run a single predeclared GT candidate in a fresh physics process."""
import ast,hashlib,os,sys
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R))
source=R/'phase15_3/evaluation/replay.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='389cb2f66d7ade77910d2f22e3955463192908bf69fb1b5661937571ffa7c152'
kind=os.environ['PHASE15_3_CANDIDATE'];assert kind in ['projected_retreat','pose_pregrasp']
tree=ast.parse(source.read_text());at=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
tree.body[at:at]=ast.parse('from phase15_3.reacquire.planner_candidates import install\ninstall(os.environ["PHASE15_3_CANDIDATE"],B)').body
changed=0
for n in ast.walk(tree):
    if isinstance(n,ast.keyword) and n.arg=='algorithm_changed':assert isinstance(n.value,ast.Constant) and n.value.value is False;n.value=ast.Constant(True);changed+=1
    if isinstance(n,ast.keyword) and n.arg=='source_split':n.value=ast.Constant('new Phase15.3 development; candidate GT intervention')
assert changed==1;ast.fix_missing_locations(tree)
exec(compile(tree,str(source)+'[candidate]', 'exec'),dict(__name__='phase15_3_candidate_replay',__file__=str(source)))
