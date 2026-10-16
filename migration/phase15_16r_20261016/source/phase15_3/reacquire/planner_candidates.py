"""Isolated candidate modules. Frozen guards/action interfaces are retained.

These are algorithm interventions for development validation, not proven fixes.
Never install into an existing process. Do not write historical source files.
"""
import ast,hashlib,json
from pathlib import Path
def _function(path,name,globals_dict,transform):
    source=Path(path).read_text();tree=ast.parse(source)
    node=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    node=transform(node);ast.fix_missing_locations(node)
    ns=dict(globals_dict);exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+'[phase15_3_candidate]','exec'),ns)
    return ns[name]
def install(kind,output):
    import phase15_2.recovery_chain as chain
    import phase15_2.safe_retreat.retreat_v2 as retreat
    from recovery_expert.ik_feasibility import project
    import torch,numpy as np
    R=Path.cwd();a=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text())
    for name in ['phase15_2/recovery_chain.py','phase15_2/safe_retreat/retreat_v2.py']:
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==a['sha256'][name]
    changes=[]
    if kind=='projected_retreat':
        def projected(planner,i,u):
            env=planner.env;x=torch.zeros((32,7),device=env.device);x[:,6]=1;x[i,:6]=torch.tensor(u,device=env.device,dtype=torch.float32)
            mask=torch.zeros(32,device=env.device,dtype=torch.bool);mask[i]=True
            return project(env,x,mask)[i,:6].detach().cpu().numpy().astype(float)
        def edit(fn):
            loop=next(x for x in fn.body if isinstance(x,ast.For) and ast.unparse(x.target)=='u')
            # Original .06 joint delta limit is checked before projection.
            at=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.If) and '.06' in ast.unparse(x.test))
            insert=ast.parse('proposed_u=u.copy()\nu=projected(planner,i,u)\ndelta=mapping@u\nif np.max(np.abs(delta))>.06:continue').body
            loop.body[at+1:at+1]=insert
            replacements=0
            for x in ast.walk(loop):
                if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='best' for t in x.targets):
                    assert ast.unparse(x.value)=='u';x.value=ast.Name(id='proposed_u',ctx=ast.Load());replacements+=1
            assert replacements==1;changes.extend(['same candidate set and score; evaluate original projection before escape/score','return original selected proposal; outer projection still executes exactly once'])
            return fn
        original=retreat.retreat_command
        chain.retreat_command=_function(R/'phase15_2/safe_retreat/retreat_v2.py','retreat_command',dict(original.__globals__,projected=projected),edit)
    elif kind=='pose_pregrasp':
        def edit(fn):
            pre=next(x for x in ast.walk(fn) if isinstance(x,ast.If) and ast.unparse(x.test)=="phase == 'pregrasp'" and any(isinstance(y,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='path' for t in y.targets) for y in x.body))
            at=next(i for i,x in enumerate(pre.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='delta' for t in x.targets))
            pre.body[at:]=ast.parse('goal=self.planner.fk(desired)').body
            removed=0
            for x in ast.walk(fn):
                if hasattr(x,'body') and isinstance(x.body,list):
                    kept=[]
                    for y in x.body:
                        if isinstance(y,ast.If) and ast.unparse(y.test)=="phase == 'pregrasp'" and 'torch.tensor(path_command' in ast.unparse(y):removed+=1
                        else:kept.append(y)
                    x.body=kept
            assert removed==1;changes.extend(['track existing checked waypoint FK pose through original six-dimensional DLS interface','remove weighted seven-joint command fit; preserve rotation clamp/project/tracking collision checks'])
            return fn
        chain.Reacquire.action=_function(R/'phase15_2/recovery_chain.py','action',vars(chain),edit)
    else:raise ValueError(kind)
    f=Path(output)/'candidate_manifest.json';f.write_text(json.dumps(dict(kind=kind,changes=changes,safety_thresholds_changed=False,models_changed=False,action_interface_changed=False,phase15_2_sources_changed=False,validated_closed_loop=False),indent=2))
