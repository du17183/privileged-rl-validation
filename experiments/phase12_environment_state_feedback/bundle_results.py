import hashlib,json,zipfile
from datetime import datetime,timezone
from experiments.phase12_environment_state_feedback.protocol import ROOT,OUT,TAG
def main():
 path=ROOT/'results/phase12_review_artifacts.zip'
 with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
  for name in (f'experiments/{TAG}','environment_feedback',f'results/{TAG}'):
   for p in (ROOT/name).rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or 'ipc' in p.parts or p.name.endswith('.partial.json') or p.suffix in ('.pt','.h5'):continue
    z.write(p,str(p.relative_to(ROOT)))
  z.write(ROOT/'docs/phase12_environment_state_feedback_report.md','docs/phase12_environment_state_feedback_report.md')
 (OUT/'review_bundle.json').write_text(json.dumps(dict(path=str(path),size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),created_at=datetime.now(timezone.utc).isoformat()),indent=2))
 print(path,flush=True)
if __name__=='__main__':main()
