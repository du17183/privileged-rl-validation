"""Small review artifacts only; large training checkpoints stay on the server."""
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase9_safe_online'

def main():
    if not (OUT/'phase9_completed.json').exists():
        raise RuntimeError('Final completion marker missing')
    archive=OUT/'review_artifacts.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        bundle.write(ROOT/'docs/phase9_safe_online_report.md','docs/phase9_safe_online_report.md')
        comparison=ROOT/'docs/method_comparison.md'
        if comparison.exists():
            bundle.write(comparison,'docs/method_comparison.md')
        for path in OUT.rglob('*'):
            if not path.is_file() or path==archive or 'ipc' in path.parts or path.suffix in ('.pt','.zip','.lock'):
                continue
            if path.suffix in ('.json','.csv','.md','.py','.png','.pdf','.txt'):
                bundle.write(path,str(path.relative_to(ROOT)))
        for folder in (ROOT/'safe_online',ROOT/'experiments/phase9_safe_online'):
            for path in folder.glob('*.py'):
                bundle.write(path,str(path.relative_to(ROOT)))
    print(archive,archive.stat().st_size)

if __name__=='__main__':main()
