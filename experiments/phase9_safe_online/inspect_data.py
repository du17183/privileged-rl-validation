import h5py
from pathlib import Path
for path in Path('datasets/phase9_safe_online').glob('*.h5'):
    with h5py.File(path) as f:
        print(path.name, len(f), dict(f.attrs), sum(int(g.attrs['success']) for g in f.values()),
              sum(len(g['action']) for g in f.values()), flush=True)
