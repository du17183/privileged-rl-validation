"""Parallelize independent clone checks; execute state updates in clone order."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from phase15_2 import recovery_chain as serial
from phase15_2.common import ROOT, RESULTS
from phase15_2.parallel_geometry import (GeometryPool, config_from_planner,
                                        exactly_equal, pose_to_wire, world_to_wire)


class Reacquire(serial.Reacquire):
    def __init__(self, env, variant='B', retreat='R2', geometry=False, strict=False, workers=4):
        super().__init__(env, variant, retreat, geometry, strict)
        self.pool_record = RESULTS / 'code_versions/parallel_geometry_v2' / f'pool_{os.getpid()}.json'
        config = config_from_planner(self.planner, ROOT)
        baseline = json.loads((self.pool_record.parent / 'baseline_manifest.json').read_text())['files']
        if any(config['hashes'][name] != baseline[name] for name in config['hashes']):
            raise RuntimeError('Parallel geometry kernels differ from archived serial baseline')
        self.parallel_pool = GeometryPool(config, workers=workers, record_path=self.pool_record)
        self.parallel_geometry_workers = workers
        self.shadow_batches = 0
        self.shadow_checks = 0
        self.candidate_shadow_batches = 0
        self.shadow_failed = False
        self.dispatch_times = []
        self.operation_times = {key:[] for key in ['retreat','tracking','candidates']}
        self.shadow_s = 0.
        print('CPU_GEOMETRY_READY ' + json.dumps(dict(parent_pid=os.getpid(), workers=workers,
              pool_record=str(self.pool_record), kernel_hashes=config['hashes'], simulation=False)), flush=True)

    def close(self):
        self.parallel_pool.close()
        record = json.loads(self.pool_record.read_text())
        record.update(shadow_batches=self.shadow_batches, shadow_checks=self.shadow_checks,
                      candidate_shadow_batches=self.candidate_shadow_batches,
                      exact_shadow_passed=self.shadow_checks > 0 and not self.shadow_failed,
                      dispatch_calls=len(self.dispatch_times),
                      dispatch_total_s=sum(self.dispatch_times),shadow_s=self.shadow_s,
                      operation_timing={key:dict(calls=len(values),total_s=sum(values),
                        quantiles_s={f'p{p}':float(np.quantile(values,p/100)) if values else None for p in [50,95,99]})
                        for key,values in self.operation_times.items()},
                      timing_scope='CPU geometry dispatch for shared batch32 simulation; startup/shadow separate, not exclusive batch1 deployment')
        self.pool_record.write_text(json.dumps(record, indent=2) + '\n')

    def _retreat_batch(self, active, joint, mapping, scale):
        # Real native replay exposed a last-bit candidate difference across CPU
        # contexts. Preserve original goal calls, mutable data and clone order.
        # No parallel retreat proposal is accepted, and no tolerance is relaxed.
        return {}

    def _consume_retreat(self, results, i, joint, mapping, scale):
        tick = time.perf_counter()
        answer = serial.retreat_command(self.planner, int(i), joint, mapping, scale)
        self.operation_times['retreat'].append(time.perf_counter() - tick)
        return answer

    def _tracking_batch(self, active, joint, mapping, action):
        ids = [int(i) for i in np.flatnonzero(active) if self.phase[i] not in ['failed', 'ready', 'release']]
        jobs = []
        for i in ids:
            trial = joint[i].copy()
            trial[:7] += mapping[i] @ action[i, :6].cpu().numpy()
            jobs.append(('tracking', dict(joint=joint[i].copy(), trial=trial,
                         world=world_to_wire(self.planner.world(i)), escape=self.phase[i] == 'retreat',
                         allow_handle=self.phase[i] in ['align', 'grasp', 'probe'])))
        started = time.perf_counter()
        accepted = self.parallel_pool.run_many(jobs)
        self.dispatch_times.append(time.perf_counter() - started)
        if jobs:self.operation_times['tracking'].append(self.dispatch_times[-1])
        if self.shadow_batches < 12:
            tick=time.perf_counter()
            for i, (_, payload), answer in zip(ids, jobs, accepted):
                world = self.planner.world(i)
                reference = (self.planner.escape_edge(payload['joint'], payload['trial'], world)
                             if payload['escape'] else self.planner.edge(payload['joint'], payload['trial'], world,
                                                                        allow_handle=payload['allow_handle']))
                if bool(reference) != bool(answer):
                    self.shadow_failed = True
                    raise RuntimeError('Parallel tracking check shadow mismatch: clone ' + str(i))
                self.shadow_checks += 1
            self.shadow_s+=time.perf_counter()-tick
        if jobs:
            self.shadow_batches += 1
        if jobs and self.shadow_batches == 12:
            print('CPU_GEOMETRY_SHADOW_PASSED ' + json.dumps(dict(batches=12, exact_checks=self.shadow_checks)), flush=True)
        return ids, accepted

    def safe_candidates(self, raw, ids, actions, mapping):
        jobs = [('candidates', dict(joint=raw[i, :9].copy(), world=world_to_wire(self.planner.world(int(i))),
                 mapping=mapping[i].copy(order='K'), actions=actions[j].copy())) for j, i in enumerate(ids)]
        started=time.perf_counter()
        answers = self.parallel_pool.run_many(jobs)
        if jobs:self.operation_times['candidates'].append(time.perf_counter()-started)
        if self.candidate_shadow_batches < 12:
            tick=time.perf_counter()
            for i, (_, payload), answer in zip(ids, jobs, answers):
                q = raw[i, :9].copy()
                world = self.planner.world(int(i))
                reference = True
                for action in payload['actions'][:3]:
                    if action[6] >= 0:
                        reference = False
                        break
                    trial = q.copy()
                    trial[:7] += mapping[i] @ action[:6]
                    if not self.planner.edge(q, trial, world, allow_handle=True):
                        reference = False
                        break
                    q = trial
                if bool(reference) != bool(answer):
                    self.shadow_failed = True
                    raise RuntimeError('Parallel Normal candidate shadow mismatch: clone ' + str(i))
                self.shadow_checks += 1
            self.shadow_s+=time.perf_counter()-tick
            if jobs:
                self.candidate_shadow_batches += 1
                if self.candidate_shadow_batches == 12:
                    print('CPU_GEOMETRY_CANDIDATE_SHADOW_PASSED ' + json.dumps(dict(batches=12, exact_checks=self.shadow_checks)), flush=True)
        checked = np.zeros(32, bool)
        for i, answer in zip(ids, answers):
            checked[i] = answer
        return checked


def install_action():
    source_path = ROOT / 'phase15_2/recovery_chain.py'
    baseline = json.loads((RESULTS / 'code_versions/parallel_geometry_v1/baseline_manifest.json').read_text())['files']
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != baseline['phase15_2/recovery_chain.py']:
        raise RuntimeError('Serial recovery action changed after baseline archive')
    source = source_path.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Reacquire')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'action')
    function = ast.get_source_segment(source, node)
    old = 'for i in np.flatnonzero(active):\n            self.elapsed[i]+=1'
    new = 'retreat_results=self._retreat_batch(active,joint,mapping,scale)\n        for i in np.flatnonzero(active):\n            self.elapsed[i]+=1'
    if function.count(old) != 1:
        raise RuntimeError('Unexpected serial action structure')
    function = function.replace(old, new)
    old = 'u,safe,check=retreat_command(self.planner,int(i),joint[i],mapping[i],sc)'
    if function.count(old) != 1:
        raise RuntimeError('Unexpected retreat call structure')
    function = function.replace(old, 'u,safe,check=self._consume_retreat(retreat_results,i,joint[i],mapping[i],sc)')
    begin = function.index('        for i in np.flatnonzero(active):\n            if self.phase[i]')
    end = function.index('        return action.cpu().numpy()', begin)
    function = function[:begin] + '''        ids,accepted=self._tracking_batch(active,joint,mapping,action)
        for i,allowed in zip(ids,accepted):
            if not allowed:
                action[i,:6]=0;self.current[i]['tracking_collision_rejections']=self.current[i].get('tracking_collision_rejections',0)+1
''' + function[end:]
    namespace = dict(serial.__dict__)
    exec(compile(function, str(source_path) + ':parallel-dispatch', 'exec'), namespace)
    Reacquire.action = torch.no_grad()(namespace['action'])


install_action()
