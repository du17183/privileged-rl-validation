"""Evidence-based recovery outcome taxonomy, with uncertainty recorded."""
import numpy as np

LABELS={1:'无法重新接近handle',2:'IK未达/限位风险',3:'姿态错误',4:'碰撞',
        5:'无法重新建立contact',6:'重新抓取后无法继续开门',7:'时间不足'}


def classify(trace,success):
    if success:return dict(code=0,label='success',confidence='observed')
    if not trace.get('action'):return dict(code=7,label=LABELS[7],confidence='observed',reason='No recovery budget')
    phase=np.asarray(trace['planner_phase']).reshape(-1)
    p=np.asarray(trace['position_error']).reshape(-1);r=np.asarray(trace['rotation_error']).reshape(-1)
    margin=np.asarray(trace['joint_margin']).reshape(-1)
    contact=np.asarray(trace['contact_state']).all(-1)
    age=int(np.asarray(trace['episode_tick'])[0,0]);length=len(phase)
    angles=np.asarray(trace['next_state'])[:,0]
    recent_progress=float(angles[-1]-angles[max(0,len(angles)-60)])
    evidence=dict(takeover_tick=age,remaining_ticks=600-age,recovery_ticks=length,
                  final_phase=int(phase[-1]),min_joint_margin=float(margin.min()),
                  final_position_error=float(p[-1]),final_rotation_error=float(r[-1]),
                  contact_ticks=int(contact.sum()),open_ticks=int((phase==5).sum()),
                  progress_last_60_ticks=recent_progress)
    if age>480:code=7
    elif phase[-1]==5 and recent_progress>.06:code=7
    elif (phase==5).any():code=6
    elif margin.min()<.015 and np.median(p[-30:])>.03:code=2
    elif phase[-1] in (2,3) and np.median(r[-30:])>.3:code=3
    elif phase[-1]==4 or (phase==4).sum()>20:code=5
    else:code=1
    return dict(code=code,label=LABELS[code],confidence='diagnostic',evidence=evidence,
                limitation='Local IK failure is not proof of global unreachability. Contact does not prove grasp. Collision is only assigned with measured collision evidence; none is available in the inherited sensors.')
