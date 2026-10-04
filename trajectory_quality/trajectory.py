"""Portable completed-trajectory object for future replay selectors.

Fixture state is an explicitly named training-only field. A deployable policy
must use state['robot']; this object is never fed directly to the actor.
"""
import h5py
import numpy as np


def read_completed(path, trajectory_id):
    with h5py.File(path,'r') as h5:
        metadata=h5['trajectories']
        if not 0 <= trajectory_id < int(h5.attrs['complete_trajectories']):
            raise IndexError('Only completed, scored trajectories are addressable')
        start=int(metadata['start'][trajectory_id]);stride=int(metadata['stride'][trajectory_id])
        length=int(metadata['length'][trajectory_id]);ids=start+stride*np.arange(length)
        flat=h5['transitions']
        result=dict(state=dict(robot=flat['robot'][ids],privileged=flat['privileged'][ids]),
                    action=flat['action'][ids],reward=flat['reward'][ids],
                    next_state=dict(robot=flat['next_robot'][ids],privileged=flat['next_privileged'][ids]),
                    done=flat['done'][ids],return_value=float(metadata['return_value'][trajectory_id]),
                    discounted_return=float(metadata['discounted_return'][trajectory_id]),
                    value=float(metadata['initial_q'][trajectory_id]),success=bool(metadata['success'][trajectory_id]),
                    quality_score=float(metadata['quality_score'][trajectory_id]),
                    source=h5.attrs['source'],seed=int(h5.attrs['seed']),arm=h5.attrs['arm'])
        result['return']=result['return_value']
        return result
