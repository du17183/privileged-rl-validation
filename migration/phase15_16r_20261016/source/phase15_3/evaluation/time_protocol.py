"""Total-budget diagnostic. Local330 cap and learned inputs remain frozen.

Control sees actual elapsed/600; model RPC receives time clipped to1.
All other state/action/safety fields are unchanged. No physical state writes.
"""
import ast,hashlib,json,sys
from pathlib import Path
def function(path,name,scope,transform):
    t=ast.parse(Path(path).read_text());n=next(x for x in ast.walk(t) if isinstance(x,ast.FunctionDef) and x.name==name)
    n=transform(n);t=ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[]));exec(compile(t,str(path)+'[time_protocol]', 'exec'),scope);return scope[name]
def install(ns,budget,output):
    assert budget in [600,900,1200];R=Path.cwd();changes={}
    if budget==600:
        (Path(output)/'time_protocol.json').write_text(json.dumps(dict(total_budget=600,local_reacquire_cap=330,control_unchanged=True,model_clock='original clipped /600',training=False),indent=2));return
    import torch
    import door_env.door as door
    original_observation=door.robot_observation
    def observation(env):
        x=original_observation(env);x[:,25]=env.episode_length_buf.float()/600.;return x
    aliases=[]
    for name,module in list(sys.modules.items()):
        if module is not None and getattr(module,'robot_observation',None) is original_observation:
            setattr(module,'robot_observation',observation);aliases.append(name)
    original_create=ns['create']
    def create(*a,**kw):
        env=original_create(*a,**kw);assert env.max_episode_length==600
        env.cfg.episode_length_s=budget*env.step_dt;assert env.max_episode_length==budget;return env
    ns['create']=create
    Policy=ns['ChainPolicy'];original_rpc=Policy.rpc
    def rpc(self,d):
        if d.get('op')=='infer':
            d=dict(d);v=d['raw'].copy();v[:,25]=v[:,25].clip(0,1);d['raw']=v
        return original_rpc(self,d)
    Policy.rpc=rpc
    # Every pressure600 literal is a deadline or deadline metadata, not a feature scale.
    import phase15_1.evaluation.pressure_eval as pe
    class Pressure(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==600:changes['pressure600']=changes.get('pressure600',0)+1;return ast.copy_location(ast.Constant(budget),n)
            if n.value==605:changes['pressure605']=changes.get('pressure605',0)+1;return ast.copy_location(ast.Constant(budget+5),n)
            return n
    ns['pressure']=function(R/'phase15_1/evaluation/pressure_eval.py','pressure',dict(vars(pe)),lambda n:Pressure().visit(n))
    assert changes['pressure600']==4 and changes['pressure605']==1,changes
    import phase15_2.recovery_chain as chain
    class Local(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==600:changes['chain600']=changes.get('chain600',0)+1;return ast.copy_location(ast.Constant(budget),n)
            return n
    chain.Reacquire.action=function(R/'phase15_2/recovery_chain.py','action',dict(vars(chain)),lambda n:Local().visit(n));assert changes['chain600']==1
    import phase15_2.pregrasp.pregrasp_geometry as geom
    class Remaining(ast.NodeTransformer):
        def __init__(self,key):self.key=key
        def visit_BinOp(self,n):
            self.generic_visit(n)
            if isinstance(n.op,ast.Sub) and isinstance(n.left,ast.Constant) and n.left.value==600:
                n.left=ast.Constant(budget);changes[self.key]=changes.get(self.key,0)+1
            return n
    geom.NormalProfile.entry=function(R/'phase15_2/pregrasp/pregrasp_geometry.py','entry',dict(vars(geom)),lambda n:Remaining('entry_deadline').visit(n));assert changes['entry_deadline']==2
    import phase15_2.evaluation.stage_eval as stage
    stage.StageWriter.observe=function(R/'phase15_2/evaluation/stage_eval.py','observe',dict(vars(stage)),lambda n:Remaining('trace_remaining').visit(n));assert changes['trace_remaining']==1
    class Result(ast.NodeTransformer):
        def visit_keyword(self,n):
            self.generic_visit(n)
            if n.arg=='episode_budget_ticks':assert n.value.value==600;n.value=ast.Constant(budget);changes['result_budget']=changes.get('result_budget',0)+1
            return n
    ns['main']=function(R/'phase15_2/evaluation/full_loop_eval.py','main',ns,lambda n:Result().visit(n));assert changes['result_budget']==1
    p=Path(output)/'time_protocol.json';p.write_text(json.dumps(dict(total_budget=budget,local_reacquire_cap=330,changes=changes,clock_aliases=aliases,
        control_clock='actual elapsed ticks/600, unbounded for controller clocks',model_clock='original /600 clipped0..1 at RPC; other model fields unchanged',
        initial_600_semantics='same apart from total-time termination at600',entry_remaining_budget=budget,historical_budget=600,
        causal_scope='total budget extension only; local330 not enlarged; later physical states outside original episode support',models_frozen=True,training=False),indent=2))
