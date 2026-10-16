"""Construct both interventions, check retained guards. Not physical validation."""
import ast,sys,types,json,time
from pathlib import Path
import numpy as np,torch
R=Path.cwd();sys.path.insert(0,str(R))
class Dummy:pass
def module(name):
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m;return m
p=module('phase15_2');c=module('phase15_2.recovery_chain');p.recovery_chain=c
c.Reacquire=Dummy;c.retreat_command=lambda *a:None;c.torch=torch;c.np=np;c.pin=None;c.time=time;c.project=None
safe=module('phase15_2.safe_retreat');p.safe_retreat=safe;r=module('phase15_2.safe_retreat.retreat_v2');safe.retreat_v2=r
exec('def retreat_command(*a):pass',vars(r))
ex=module('recovery_expert');ik=module('recovery_expert.ik_feasibility');ex.ik_feasibility=ik;ik.project=lambda *a:None
from phase15_3.reacquire import planner_candidates as m
captured={}
def build(path,name,scope,transform):
    root=ast.parse(Path(path).read_text());n=next(x for x in ast.walk(root) if isinstance(x,ast.FunctionDef) and x.name==name);n=transform(n);ast.fix_missing_locations(n)
    captured[name]=ast.unparse(n);s=dict(scope);exec(compile(ast.Module(body=[n],type_ignores=[]),str(path),'exec'),s);return s[name]
m._function=build
out=R/'phase15_3/results/candidate_contract_gate';out.mkdir(exist_ok=True)
for kind in ['projected_retreat','pose_pregrasp']:
    d=out/kind;d.mkdir(exist_ok=True);m.install(kind,d)
    text=captured['retreat_command' if kind=='projected_retreat' else 'action']
    (d/'transformed_function.py.txt').write_text(text)
    if kind=='projected_retreat':
        assert text.index('if np.max(np.abs(delta)) > 0.06:')<text.index('u = projected(planner, i, u)')<text.index('planner.escape_edge')
        assert 'best = proposed_u' in text and '100 * margin.sum() + 30 * sum(penetration.values())' in text
    else:
        assert 'path_command' not in text and 'goal = self.planner.fk(desired)' in text
        assert 'action[i, 3:6].clamp_(-0.3, 0.3)' in text and 'action = project(env, action,' in text
        assert 'self.elapsed[i] > min(330, 600 - meta' in text
        assert 'self.planner.escape_edge' in text and 'self.planner.edge' in text and 'valid_grasp(' in text
print('Both constructed; unchanged safety/330/600 guards present. No physics success or equivalence claimed.')
(out/'gate.json').write_text(json.dumps(dict(passed=True,scope='source construction and retained-guard contract only',physical_validation=False,algorithms_equivalent=False,training=False),indent=2))
