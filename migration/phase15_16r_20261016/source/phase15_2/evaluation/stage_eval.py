"""Stream sparse physical-step diagnostics; unknown physical collisions stay unknown."""
import h5py
import numpy as np
from phase15_2.router.state_machine_v2 import NAMES
import pinocchio as pin


def rotation(q):
    q=np.asarray(q,float)
    return pin.Quaternion(q[0],q[1],q[2],q[3]).normalized().matrix()

FIELDS=[('clone','i2'),('batch','i2'),('episode','i4'),('tick','i2'),('router','i1'),('remaining_ticks','i2'),('q','f4',(9,)),('qd','f4',(9,)),('joint_margin','f4',(9,)),('eef_pose','f4',(7,)),('handle_pose','f4',(7,)),('relative_position','f4',(3,)),('relative_rotvec','f4',(3,)),('door_angle','f4'),('door_velocity','f4'),('contact','f4',(2,)),('model_collision_assessed','?'),('model_collision_rejected','?'),('actual_arm_collision_observed','?'),('actual_arm_collision_known','?'),('ik_solution_found','i1'),('planner_target','f4',(7,)),('target_position_error','f4'),('target_orientation_error','f4'),('action','f4',(7,)),('stage_seen','i2',(9,)),('terminal','?')]


def active_target(reacquire,i):
    """Return the currently executed target, not an old grasp target."""
    phase=reacquire.phase[i];meta=reacquire.current[i] or {}
    if phase=='retreat':
        checks=meta.get('retreat_checks',[])
        if checks and 'target_position' in checks[-1]:
            return np.asarray(checks[-1]['target_position']),None
    if phase=='pregrasp' and reacquire.path[i] is not None:
        path=reacquire.path[i];cursor=int(reacquire.cursor[i])
        if cursor<len(path):
            pose=reacquire.planner.fk(path[cursor]);return pose.translation.copy(),pose.rotation.copy()
    if phase in ['align','grasp','probe','ready']:
        pose=reacquire.grasp_target[i]
        if pose is not None:return pose.translation.copy(),pose.rotation.copy()
    return None,None


class StageWriter:
    def __init__(self,path):
        self.h=h5py.File(path,'x');self.h.attrs.update(schema='phase15_2_stage_v2',sampling='every10ticks or stage/state change or terminal',actual_arm_collision_instrumented=False,state_timing='pre-action q/pose/door/contact; post-action terminal/stage flags',planner_target_frame='robot root; retreat position has no orientation target',rl_updates=0)
        self.ds=self.h.create_dataset('steps',(0,),maxshape=(None,),dtype=np.dtype(FIELDS),chunks=(512,),compression='gzip')
        self.last=np.full((32,10),-1000,int);self.buffer=[]

    def observe(self,client,robot,gt,done):
        r=robot.cpu().numpy();g=gt.cpu().numpy();d=done.cpu().numpy()
        before=client.before_action;bounds=client.env.scene['robot'].data.joint_pos_limits.cpu().numpy()
        for i in np.flatnonzero(client.active):
            raw=before[i];age=round(raw[25]*600);stages=client.stage_seen[i]
            fingerprint=np.r_[client.router.state[i],stages]
            if age%10 and not d[i] and np.array_equal(fingerprint,self.last[i]):continue
            self.last[i]=fingerprint
            record=np.zeros((),dtype=self.ds.dtype)
            record['clone']=i;record['batch']=client.batch;record['episode']=client.episode[i];record['tick']=age
            record['router']=client.router.state[i];record['remaining_ticks']=600-age
            record['q']=raw[:9];record['qd']=raw[9:18];record['joint_margin']=np.minimum(raw[:9]-bounds[i,:,0],bounds[i,:,1]-raw[:9])
            record['eef_pose']=raw[18:25];record['handle_pose']=np.r_[raw[32:35],raw[35:39]]
            hr=rotation(raw[35:39]);er=rotation(raw[21:25])
            record['relative_position']=hr.T@(raw[18:21]-raw[32:35]);record['relative_rotvec']=pin.log3(hr.T@er)
            record['door_angle']=client.before_tool[i,0];record['door_velocity']=client.before_tool[i,1];record['contact']=raw[30:32]
            record['ik_solution_found']=-1;record['planner_target']=np.nan
            record['target_position_error']=np.nan;record['target_orientation_error']=np.nan
            if client.planner is not None:
                planner=client.planner.planner;meta=client.planner.current[i]
                if meta:
                    if not meta.get('planning_not_attempted',False):record['ik_solution_found']=int(meta['ik_success'])
                    position,orientation=active_target(client.planner,i)
                    if position is not None:
                        actual=planner.fk(raw[:9]);record['planner_target'][:3]=position
                        record['target_position_error']=np.linalg.norm(actual.translation-position)
                        if orientation is not None:
                            quaternion=pin.Quaternion(orientation).coeffs();record['planner_target'][3:]=np.r_[quaternion[3],quaternion[:3]]
                            record['target_orientation_error']=np.linalg.norm(pin.log3(actual.rotation@orientation.T))
                if age%30==0:
                    record['model_collision_assessed']=True
                    record['model_collision_rejected']=planner.collision(raw[:9],planner.world(int(i)),allow_handle=bool(np.all(raw[30:32]>.5)))
            record['actual_arm_collision_known']=False
            record['action']=client.last_action[i];record['stage_seen']=stages;record['terminal']=d[i]
            self.buffer.append(record)
        if len(self.buffer)>=512:self.flush()

    def flush(self):
        if not self.buffer:return
        n=len(self.ds);self.ds.resize((n+len(self.buffer),));self.ds[n:]=np.array(self.buffer,dtype=self.ds.dtype);self.buffer=[];self.h.flush()

    def close(self):
        self.flush();self.h.close()
