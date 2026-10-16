"""Preserve validated snapshots when changing evaluation scheduling."""
from pathlib import Path
import hashlib
import h5py


def append_previous(destination,source_path,case,target):
    source_path=Path(source_path)
    if not source_path.exists():return dict(imported=0)
    digest=hashlib.sha256(source_path.read_bytes()).hexdigest()
    with h5py.File(source_path,'r') as source:
        selected=sorted(source)[:target]
        for number,key in enumerate(selected):
            original=source[key]
            if case=='natural_severe' and not bool(original.attrs.get('natural_source_episode_failed',False)):
                raise RuntimeError('Resume contains a natural snapshot without a failed source episode')
            for required in ['prefix_actions','reset_parameters','handoff_robot','handoff_state']:
                if required not in original:raise RuntimeError('Incomplete original snapshot: '+key)
            source.copy(key,destination,name=f'state_{number:04d}')
            copied=destination[f'state_{number:04d}']
            copied.attrs['source_archive_path']=str(source_path)
            copied.attrs['source_archive_sha256']=digest
            if 'reset_seed' not in copied.attrs:copied.attrs['reset_seed']=source.attrs['seed']
    destination.flush()
    return dict(imported=len(selected),source=str(source_path),sha256=digest,
                selection='first qualifying snapshots in original recorded order',continued_with_independent_reset_seed=True)
