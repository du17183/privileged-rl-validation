from dataclasses import asdict, dataclass
import numpy as np

NORMAL, RECOVERY, REACQUIRE = range(3)
NAMES = ['normal', 'recovery', 'reacquire']


@dataclass(frozen=True)
class Thresholds:
    approach_grace_ticks: int = 120
    severe_distance_m: float = .12
    mild_distance_m: float = .08
    near_distance_m: float = .06
    contact_loss_ticks: int = 8
    stall_ticks: int = 60
    regression_rad: float = .05
    positive_progress_rad: float = .03
    recovery_min_ticks: int = 30
    recovery_timeout_ticks: int = 90
    switch_cooldown_ticks: int = 15


class Router:
    def __init__(self, n=32, thresholds=None):
        self.n = n
        self.cfg = thresholds or Thresholds()
        self.state = np.zeros(n, int)
        self.entry = np.zeros(n, int)
        self.last_switch = np.full(n, -1000, int)
        self.peak = np.zeros(n, np.float32)
        self.progress_mark = self.peak.copy()
        self.progress_tick = np.zeros(n, int)
        self.previous_contact = np.zeros(n, bool)
        self.contact_lost = np.zeros(n, int)
        self.grasp_run = np.zeros(n, int)
        self.switches = np.zeros(n, int)
        self.recovery_entries = np.zeros(n, int)
        self.planner_entries = np.zeros(n, int)
        self.entry_distance = np.zeros(n, np.float32)
        self.entry_angle = np.zeros(n, np.float32)

    def reset(self, ids, raw):
        self.state[ids] = NORMAL
        self.entry[ids] = 0
        self.last_switch[ids] = -1000
        self.peak[ids] = raw[ids, 26]
        self.progress_mark[ids] = raw[ids, 26]
        self.progress_tick[ids] = 0
        self.previous_contact[ids] = False
        self.contact_lost[ids] = 0
        self.grasp_run[ids] = 0
        self.switches[ids] = 0
        self.recovery_entries[ids] = 0
        self.planner_entries[ids] = 0

    def update(self, raw, active, modular, planner_enabled, planner_ready=None):
        c = self.cfg
        age = np.rint(raw[:, 25] * 600).astype(int)
        angle = raw[:, 26]
        distance = np.linalg.norm(raw[:, 32:35] - raw[:, 18:21], axis=-1)
        contact = np.all(raw[:, 30:32] > .5, axis=-1)
        width = raw[:, 7:9].sum(-1)
        grasp = contact & (distance < c.near_distance_m) & (width > .005) & (width < .079)
        self.grasp_run = np.where(active & grasp, self.grasp_run + 1, 0)
        first_contact = active & contact & ~self.previous_contact
        self.previous_contact |= active & contact
        self.contact_lost = np.where(active & ~contact & self.previous_contact, self.contact_lost + 1, 0)
        moved = active & ((angle > self.progress_mark + .015) | first_contact)
        self.progress_tick[moved] = age[moved]
        self.progress_mark[moved] = angle[moved]
        self.peak[active] = np.maximum(self.peak[active], angle[active])
        armed = active & ((age >= c.approach_grace_ticks) | self.previous_contact)
        reasons = np.full(self.n, '', dtype=object)
        opening_stall = self.previous_contact & (age - self.progress_tick >= c.stall_ticks)
        approach_stall = (age >= 200) & (distance > .04) & (angle < .05)
        reasons[armed & (opening_stall | approach_stall)] = 'stall'
        reasons[armed & (distance > c.mild_distance_m)] = 'eef_deviation'
        reasons[armed & (self.peak > .05) & (self.contact_lost >= c.contact_loss_ticks)] = 'contact_loss'
        reasons[armed & (self.peak - angle > c.regression_rad)] = 'angle_regression'
        severe = armed & (distance > c.severe_distance_m)
        reasons[severe] = 'severe_deviation'
        before = self.state.copy()
        can_switch = active & (age - self.last_switch >= c.switch_cooldown_ticks)
        if modular:
            from_normal = can_switch & (before == NORMAL) & (reasons != '')
            self.state[from_normal] = RECOVERY
            self.state[from_normal & severe & planner_enabled] = REACQUIRE
            recovery = can_switch & (before == RECOVERY)
            no_improvement = (age - self.entry >= c.recovery_timeout_ticks) & (
                distance > self.entry_distance - .02) & (angle < self.entry_angle + c.positive_progress_rad)
            if planner_enabled:
                self.state[recovery & (severe | no_improvement)] = REACQUIRE
            recovered = recovery & (age - self.entry >= c.recovery_min_ticks) & (
                self.grasp_run >= 3) & (angle > self.entry_angle + c.positive_progress_rad)
            self.state[recovered] = NORMAL
            if planner_ready is not None:
                self.state[can_switch & (before == REACQUIRE) & planner_ready] = NORMAL
        changed = self.state != before
        self.switches[changed] += 1
        self.entry[changed] = age[changed]
        self.last_switch[changed] = age[changed]
        self.entry_distance[changed] = distance[changed]
        self.entry_angle[changed] = angle[changed]
        self.recovery_entries[changed & (self.state == RECOVERY)] += 1
        self.planner_entries[changed & (self.state == REACQUIRE)] += 1
        # Reset stall/reference hysteresis when handing back; no physical reset.
        returned = changed & (self.state == NORMAL)
        self.progress_tick[returned] = age[returned]
        self.progress_mark[returned] = angle[returned]
        self.peak[returned] = angle[returned]
        self.contact_lost[returned] = 0
        return before, changed, reasons

    def specification(self):
        return asdict(self.cfg)
