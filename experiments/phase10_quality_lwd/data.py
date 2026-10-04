"""Read-only expert trajectory metadata and compact newly collected dataset."""
import h5py
import numpy as np
from trajectory_quality.quality_model import trajectory_score


def expert_records(path):
    records, cursor = [], 0
    with h5py.File(path, 'r') as h5:
        for key in sorted(k for k in h5 if k.startswith('traj_')):
            group = h5[key]
            length = len(group['action'])
            score = trajectory_score(group['state'][:], group['next_state'][:], group.attrs['success'])
            records.append(dict(start=cursor, length=length, stride=1, source='expert', **score))
            cursor += length
    if len(records) != 1000 or not all(r['success'] for r in records):
        raise ValueError('Expert set changed')
    return records


def save_online(path, online, records, metadata):
    # Flat storage avoids thousands of small HDF5 groups. Index triples recover
    # complete trajectories; pending transitions are retained but never scored.
    with h5py.File(path, 'x') as h5:
        for k, v in metadata.items():
            h5.attrs[k] = v
        flat = h5.create_group('transitions')
        for key, tensor in online.data.items():
            flat.create_dataset(key, data=tensor[:len(online)].cpu().numpy(), compression='lzf', shuffle=True)
        trajectories = h5.create_group('trajectories')
        for key in ('start','length','stride','success','quality_score','contact_stability',
                    'start_angle','final_angle','return_value','estimated_value','initial_q','discounted_return','env_steps'):
            trajectories.create_dataset(key, data=np.asarray([r[key] for r in records]))
        h5.attrs['complete_trajectories'] = len(records)
        h5.attrs['partial_transitions'] = len(online)-sum(r['length'] for r in records)
        h5.attrs['trajectory_schema'] = 'state=robot plus privileged; action,reward,next_state,return,value,success,quality_score; index=start+stride*arange(length)'
