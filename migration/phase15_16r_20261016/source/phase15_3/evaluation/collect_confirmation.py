"""Fresh unseen confirmation state collection with frozen historical source controller."""
import ast,hashlib,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
R=Path.cwd();A=json.loads((R/'phase15_3/results/phase15_2_baseline_audit.json').read_text());assert A['passed']
src=R/'phase15_2/evaluation/natural_severe_eval.py';assert hashlib.sha256(src.read_bytes()).hexdigest()==A['sha256'][str(src.relative_to(R))]
tree=ast.parse(src.read_text());tree.body=[x for x in tree.body if not(isinstance(x,ast.If) and '__name__' in ast.unparse(x.test))]
ns=dict(__name__='phase15_3_new_confirmation_source',__file__=str(src));exec(compile(tree,str(src),'exec'),ns)
import phase15_2.common as common
common.RESULTS=R/'phase15_3';common.redirect_calibration()
ns['RESULTS']=common.RESULTS
for role,p in common.frozen(ns['a'].model_seed).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==A['sha256'][str(Path(p).relative_to(R))]
code=0;error=None
try:ns['main']()
except BaseException:
    code=1;error=traceback.format_exc();print(error,flush=True)
finally:
    out=Path(ns['a'].output)
    (out/'exit.json').write_text(json.dumps(dict(exit_code=code,error=error,phase15_3=True,training_labels=False),indent=2))
    if code==0:
        import h5py
        f=out/'natural_severe.h5'
        with h5py.File(f,'r+') as h:h.attrs.update(phase15_3=True,scope='unseen confirmation states, frozen source; evaluation_only; not expert labels',source_code_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
        s=out/'natural_severe.json';data=json.loads(s.read_text());data.update(phase15_3=True,source_split='unseen confirmation',cohort_sha256=hashlib.sha256(f.read_bytes()).hexdigest());s.write_text(json.dumps(data,indent=2))
    ns['app'].close()
sys.exit(code)
