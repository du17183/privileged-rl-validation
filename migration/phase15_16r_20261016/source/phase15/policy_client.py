"""Unchanged action execution; clear both caches at every module switch."""
import time
from multiprocessing.connection import Client
import h5py
import numpy as np
import torch
from phase15.common import AUTH
from phase15.recovery_policy.state import expand
from phase15.router.recovery_state_machine import Router, NORMAL, RECOVERY, REACQUIRE, NAMES


class ModularPolicy(torch.nn.Module):
    def __init__(self, socket, env, seed=143801):
        super().__init__()
        deadline = time.monotonic() + 900
        while True:
            try:
                self.conn = Client(str(socket), family='AF_UNIX', authkey=AUTH)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(.5)
        self.env = env
        self.seed = seed
        self.planner = None
        self.arm = 'A'
        self.configure()

    def rpc(self, d):
        self.conn.send(d)
        result = self.conn.recv()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def load(self, checkpoints, arm):
        self.arm = arm
        self.metas = self.rpc(dict(op='load', checkpoints=checkpoints))['meta']
        self.mean = np.asarray(self.metas['normal']['mean'], np.float32)
        self.std = np.asarray(self.metas['normal']['std'], np.float32)
        if self.mean.shape != (39,):
            raise RuntimeError('Normal interface changed')
        if arm == 'D' and self.planner is None:
            from phase15.reacquire.gt_reacquire import GTReacquire
            self.planner = GTReacquire(self.env)
        return self.metas

    def configure(self, cohort=None):
        self.router = Router()
        self.cache = np.zeros((32, 10, 7), np.float32)
        self.replan = np.full(32, -1000, int)
        self.last = np.full(32, -1, int)
        self.episode = np.zeros(32, int)
        self.thresholds = []
        self.cohort_ids = []
        self.batch = -1
        self.pressure = cohort is not None
        if cohort is not None:
            with h5py.File(cohort, 'r') as h:
                names = sorted(h)
                values = [len(h[k]['prefix_actions']) + 1 for k in names]
            self.thresholds = [np.asarray(values[i:i + 32], int) for i in range(0, len(values), 32)]
            self.cohort_ids = names
        self.active = np.zeros(32, bool)
        self.finished = np.zeros(32, bool)
        self.handed = np.zeros(32, bool)
        self.handoff_records = []
        self.events = [[] for _ in range(32)]
        self.event = [None] * 32
        self.costs = [{name: 0. for name in ['normal', 'recovery', 'router', 'planner']} for _ in range(32)]
        self.calls = [{name: 0 for name in ['normal', 'recovery']} for _ in range(32)]
        self.telemetry = []
        self.requests = []
        self.reset_clears = 0
        self.switch_clears = 0
        self.handoff_replans = 0
        self.current_raw = None
        self.last_action = np.zeros((32, 7), np.float32)
        if self.planner is not None:
            self.planner.reset(np.arange(32))

    def _reset(self, ids, raw):
        self.router.reset(ids, raw)
        for i in ids:
            self.events[i] = []
            self.event[i] = None
            self.costs[i] = {name: 0. for name in self.costs[i]}
            self.calls[i] = {name: 0 for name in self.calls[i]}
        if self.planner is not None:
            self.planner.reset(ids)

    def forward(self, x):
        raw = x.detach().cpu().numpy() * self.std + self.mean
        age = np.rint(raw[:, 25] * 600).astype(int)
        if self.pressure and np.all(age == 0) and (self.batch < 0 or np.any(self.last > 0)):
            self.batch += 1
            if self.batch >= len(self.thresholds):
                raise RuntimeError('Unexpected pressure batch reset')
            self.finished[:] = False
            self.handed[:] = False
        reset = (age < self.last) | (self.last < 0)
        self.episode += (age < self.last)
        self.reset_clears += int(reset.sum())
        self.replan[reset] = -1000
        self._reset(np.flatnonzero(reset), raw)
        active = np.ones(32, bool)
        handed = np.zeros(32, bool)
        thresholds = np.zeros(32, int)
        if self.pressure:
            thresholds[:] = 10000
            if self.batch >= 0:
                thresholds[:len(self.thresholds[self.batch])] = self.thresholds[self.batch]
            # The historical pressure evaluator keeps stepping auto-reset
            # clones while other samples finish. They are no longer scored.
            # Do not route, infer or count a second takeover for those clones.
            active = (age >= thresholds) & ~self.finished
            # Never switch/infer on prerecorded actions which are overwritten.
            handed = active & ~self.handed
            self.handed[handed] = True
            self.handoff_replans += int(handed.sum())
            self.replan[handed] = -1000
            self.cache[handed] = 0
            history = [v.copy() for v in [self.router.previous_contact, self.router.peak,
                       self.router.progress_mark, self.router.progress_tick]]
            self._reset(np.flatnonzero(handed), raw)
            for array, old in zip([self.router.previous_contact, self.router.peak,
                                  self.router.progress_mark, self.router.progress_tick], history):
                array[handed] = old[handed]
            prefix = ~active & ~self.finished & (thresholds < 10000)
            contact = np.all(raw[:, 30:32] > .5, axis=-1) & (raw[:, 26] > .05)
            self.router.previous_contact |= prefix & contact
            self.router.peak[prefix] = np.maximum(self.router.peak[prefix], raw[prefix, 26])
            advanced = prefix & (raw[:, 26] > self.router.progress_mark + .015)
            self.router.progress_mark[advanced] = raw[advanced, 26]
            self.router.progress_tick[advanced] = age[advanced]
        tick = time.perf_counter()
        ready = self.planner.ready.copy() if self.planner is not None else None
        before, changed, reasons = self.router.update(raw, active, self.arm in ['C', 'D'],
                                                       self.arm == 'D', ready)
        router_dt = time.perf_counter() - tick
        for i in np.flatnonzero(active):
            self.costs[i]['router'] += router_dt
            new_failure = changed[i] and self.router.state[i] != NORMAL and (
                self.event[i] is None or self.event[i].get('returned_normal', False))
            if reasons[i] and (self.event[i] is None or new_failure):
                event = dict(tick=int(age[i]), trigger=str(reasons[i]),
                    detection=True, switch_correct=bool(self.arm in ['C', 'D']),
                    route=NAMES[int(self.router.state[i])], initial_angle=float(raw[i, 26]),
                    initial_distance=float(np.linalg.norm(raw[i, 32:35] - raw[i, 18:21])),
                    flags=[False] * 4, runs=[0] * 4, returned_normal=False)
                self.events[i].append(event)
                self.event[i] = event
        self.replan[changed] = -1000
        self.cache[changed] = 0
        self.switch_clears += int(changed.sum())
        for i in np.flatnonzero(changed & (self.router.state == NORMAL)):
            if self.event[i] is not None:
                self.event[i]['returned_normal'] = True
                self.event[i]['return_tick'] = int(age[i])
            # Keep stage tracking until episode terminal; do not invent another
            # failure event from the same resolved contact-loss interval.
        enters = np.flatnonzero(changed & (self.router.state == REACQUIRE))
        if len(enters):
            self.planner.start(enters)
        action = np.zeros((32, 7), np.float32)
        planning = active & (self.router.state == REACQUIRE)
        if np.any(planning):
            planner_action, timing = self.planner.action(planning)
            action[planning] = planner_action[planning]
            for i in np.flatnonzero(planning):
                self.costs[i]['planner'] += timing
        for state, module in [(NORMAL, 'normal'), (RECOVERY, 'recovery')]:
            eligible = active & (self.router.state == state)
            need = eligible & ((age - self.replan >= 5) | reset | changed)
            if np.any(need):
                ids = np.flatnonzero(need)
                seeds = self.seed + ids * 1000003 + self.episode[ids] * 1009 + age[ids] * 13
                values = raw[ids] if module == 'normal' else expand(raw[ids])
                wall = time.perf_counter()
                result = self.rpc(dict(op='infer', module=module, raw=values, noise_seeds=seeds.tolist()))
                wall_dt = time.perf_counter() - wall
                self.cache[ids] = result['actions']
                self.replan[ids] = age[ids]
                self.requests.append(dict(module=module, batch=len(ids), latency_s=result['latency_s'],
                    service_s=result['service_s'], rpc_s=wall_dt, clip_fraction=result['clip_fraction']))
                for i in ids:
                    self.costs[i][module] += wall_dt
                    self.calls[i][module] += 1
            ids = np.flatnonzero(eligible)
            cursor = np.clip(age[ids] - self.replan[ids], 0, 4)
            action[ids] = self.cache[ids, cursor]
        for i in np.flatnonzero(handed):
            index = self.batch * 32 + int(i)
            planner = self.router.state[i] == REACQUIRE
            refreshed = planner or self.replan[i] == age[i]
            self.handoff_records.append(dict(cohort_id=self.cohort_ids[index],
                tick=int(age[i]), cache_cleared=True, refreshed=bool(refreshed),
                module=NAMES[int(self.router.state[i])]))
            if not refreshed:
                raise RuntimeError('Action cache was not refreshed at physical takeover')
        self.active = active
        self.current_raw = raw
        self.last_action = action
        self.last = age
        return torch.as_tensor(action, dtype=torch.float32, device=x.device)

    def observe(self, robot, gt, done):
        # Called after physics using the unchanged terminal-state interface.
        r = robot.detach().cpu().numpy()
        g = gt.detach().cpu().numpy()
        d = done.detach().cpu().numpy()
        distance = np.linalg.norm(r[:, 18:21] - g[:, 2:5], axis=-1)
        contact = np.all(g[:, 9:11] > .5, axis=-1)
        width = r[:, 7:9].sum(-1)
        for i in np.flatnonzero(self.active):
            e = self.event[i]
            if e is not None:
                conditions = [distance[i] < .06, bool(contact[i]), bool(contact[i] and distance[i] < .06 and
                    self.last_action[i, 6] < 0 and .005 < width[i] < .079), g[i, 0] > e['initial_angle'] + .05]
                for k, (ok, duration) in enumerate(zip(conditions, [6, 3, 3, 30])):
                    e['runs'][k] = e['runs'][k] + 1 if ok else 0
                    e['flags'][k] |= e['runs'][k] >= duration
            if d[i]:
                for event in self.events[i]:
                    event['final_success'] = bool(g[i, 0] > 1.)
                if self.planner is not None:
                    for plan in self.planner.history[i]:
                        plan['final_task_success'] = bool(g[i, 0] > 1.)
                row = dict(clone=int(i), episode=int(self.episode[i]), success=int(g[i, 0] > 1),
                    final_angle_rad=float(g[i, 0]), events=self.events[i],
                    recovery_triggers=int(self.router.recovery_entries[i]),
                    reacquire_triggers=int(self.router.planner_entries[i]),
                    switches=int(self.router.switches[i]), costs_s=self.costs[i].copy(),
                    pi_calls=self.calls[i].copy(), original_episode_ticks=int(round(r[i, 25] * 600)),
                    planner=(self.planner.records(i) if self.planner is not None else []))
                if self.pressure:
                    index = self.batch * 32 + int(i)
                    if index < len(self.cohort_ids):
                        row['cohort_id'] = self.cohort_ids[index]
                self.telemetry.append(row)
                if self.pressure:
                    self.finished[i] = True
                # Auto-reset may reach age0 == last0 in extremely short tests.
                self.replan[i] = -1000

    def stats(self):
        summary = dict(chunk_length=10, execute_steps=5, simulated_replan_hz=12.,
            reset_cache_clears=self.reset_clears, switch_cache_clears=self.switch_clears,
            handoff_replans=self.handoff_replans, router=self.router.specification(),
            handoff_records=self.handoff_records,
            requests=len(self.requests), terminal_episodes=len(self.telemetry),
            episode_cost_definition='sum full batch request wall latency for each participating episode; shared batch cost not divided',
            no_physical_state_writes_after_test_handoff=True, rl_updates=0)
        for module in ['normal', 'recovery']:
            values = [v['latency_s'] for v in self.requests if v['module'] == module]
            summary[module] = dict(calls=len(values), mean_s=float(np.mean(values)) if values else None,
                p95_s=float(np.quantile(values, .95)) if values else None,
                batch1_calls=sum(v['batch'] == 1 for v in self.requests if v['module'] == module))
        return summary

    def benchmark(self):
        result = {}
        raw = self.current_raw[:1]
        for module in self.metas:
            values = raw if module == 'normal' else expand(raw)
            times = []
            for i in range(8):
                tick = time.perf_counter()
                response = self.rpc(dict(op='infer', module=module, raw=values, noise_seeds=[150999 + i]))
                wall = time.perf_counter() - tick
                if i >= 3:
                    times.append(dict(model_s=response['latency_s'], service_s=response['service_s'], rpc_s=wall))
            result[module] = dict(batch=1, warmup=3, measured=5, samples=times,
                mean_model_s=float(np.mean([v['model_s'] for v in times])),
                mean_rpc_s=float(np.mean([v['rpc_s'] for v in times])),
                replan_budget_s=5 / 60., budget_met=all(v['rpc_s'] <= 5 / 60. for v in times))
        return result

    def stop(self):
        self.rpc(dict(op='stop'))
        self.conn.close()
