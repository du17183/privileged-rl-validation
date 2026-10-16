"""600/900/1200 total-budget extension, isolated from all formal results."""
import ast,hashlib,os,sys
from pathlib import Path
R=Path.cwd();sys.path.insert(0,str(R));src=R/'phase15_3/evaluation/replay.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='389cb2f66d7ade77910d2f22e3955463192908bf69fb1b5661937571ffa7c152'
tree=ast.parse(src.read_text());at=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets))
tree.body[at:at]=ast.parse('from phase15_3.evaluation.time_protocol import install\ninstall(ns,int(os.environ["PHASE15_3_TOTAL_BUDGET"]),B)').body
ast.fix_missing_locations(tree);exec(compile(tree,str(src)+'[new_time_protocol]','exec'),dict(__name__='phase15_3_time_replay',__file__=str(src)))
