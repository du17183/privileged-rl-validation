"""Retain preliminary pressure results with a superseded initialization."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/phase14_2_recovery_bc'

def main():
    count=0
    for f in list((R/'evaluation').glob('*/*.json')):
        if f.stem in ['fixed','random']:continue
        if json.loads(f.read_text()).get('restore_snapshot_after_warmup'):continue
        dest=R/'evaluation_initialization_v1'/f.parent.name;dest.mkdir(parents=True,exist_ok=True)
        for suffix in ['.json','.csv','.initial.npz']:
            source=f.with_suffix(suffix)
            if source.exists():
                target=dest/source.name
                if target.exists():raise FileExistsError(target)
                source.rename(target);count+=1
    print('Archived preliminary pressure artifacts',count,flush=True)

if __name__=='__main__':main()
