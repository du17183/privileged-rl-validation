"""Independent CPU geometry processes. No simulator, actor, or task imports.

Each process owns Pinocchio/FCL mutable data. The authoritative functions are
compiled unchanged from SHA-pinned source, and jobs retain environment order.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pinocchio as pin
import hppfcl as fcl
from scipy.optimize import linprog


FUNCTIONS = {
    'phase15/reacquire/pregrasp_planner.py': ('PregraspPlanner', ['pack', 'fk', 'collision', 'edge']),
    'phase15_1/severe_reacquire/pregrasp_planner.py': ('PregraspPlanner', ['profile', 'escape_edge']),
    'phase15_1/severe_reacquire/safe_retreat.py': (None, ['retreat_command']),
    'phase15_2/safe_retreat/retreat_v1.py': (None, ['normal_target']),
    'phase15_2/safe_retreat/retreat_v2.py': (None, ['retreat_command']),
}


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_functions(root, hashes):
    namespace = dict(np=np, pin=pin, fcl=fcl, linprog=linprog)
    methods = {}
    for relative, (classname, names) in FUNCTIONS.items():
        path = Path(root) / relative
        if source_hash(path) != hashes[relative]:
            raise RuntimeError('Geometry source SHA mismatch: ' + relative)
        tree = ast.parse(path.read_text())
        nodes = tree.body
        if classname:
            nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == classname).body
        for name in names:
            node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            if classname:
                methods[name] = namespace[name]
            elif relative.endswith('severe_reacquire/safe_retreat.py'):
                namespace['reference'] = namespace[name]
    return methods, namespace['retreat_command']


def world_to_wire(world):
    result = []
    for shape, transform in world:
        if not isinstance(shape['geometry'], fcl.Box):
            raise TypeError('World geometry is not an original Box; serial path required')
        metadata = {k: v for k, v in shape.items() if k != 'geometry'}
        result.append((metadata, (np.asarray(shape['geometry'].halfSide) * 2).copy(),
                       np.asarray(transform.getRotation()).copy(), np.asarray(transform.getTranslation()).copy()))
    return result


def wire_to_world(world):
    return [(dict(metadata, geometry=fcl.Box(*size)), fcl.Transform3f(rotation, position))
            for metadata, size, rotation, position in world]


def pose_to_wire(pose):
    return np.asarray(pose.rotation).copy(), np.asarray(pose.translation).copy()


def config_from_planner(planner, root):
    root = Path(root)
    urdf = Path(sys.prefix) / 'lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf'
    hashes = {relative: source_hash(root / relative) for relative in FUNCTIONS}
    return dict(root=str(root), hashes=hashes, urdf=str(urdf), urdf_sha256=source_hash(urdf),
                indices=list(planner.indices), collision_pairs=[(int(x.first), int(x.second)) for x in planner.geom.collisionPairs],
                geometry_names=[x.name for x in planner.geom.geometryObjects],
                lower=planner.lower.copy(), upper=planner.upper.copy(), clear=planner.clear.copy())


class Geometry:
    def __init__(self, config):
        methods, self.retreat_function = load_functions(config['root'], config['hashes'])
        for name, method in methods.items():
            setattr(self, name, method.__get__(self, type(self)))
        urdf = Path(config['urdf'])
        if source_hash(urdf) != config['urdf_sha256']:
            raise RuntimeError('Robot URDF changed')
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.frame = self.model.getFrameId('panda_hand')
        self.offset = pin.SE3(np.eye(3), np.array([0., 0., .107]))
        self.indices = list(config['indices'])
        self.geom = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.COLLISION,
                                       [str(urdf.parent.parent.parent), str(urdf.parent.parent)])
        if [x.name for x in self.geom.geometryObjects] != config['geometry_names']:
            raise RuntimeError('Robot geometry order changed')
        for first, second in config['collision_pairs']:
            self.geom.addCollisionPair(pin.CollisionPair(first, second))
        self.gdata = pin.GeometryData(self.geom)
        self.request = fcl.CollisionRequest()
        self.profile_request = fcl.DistanceRequest()
        self.profile_request.enable_nearest_points = False
        self.lower = config['lower'].copy()
        self.upper = config['upper'].copy()
        self.clear = config['clear'].copy()
        self.retreat_origins = {}

    def world(self, i):
        return self._world

    def goal(self, i, pregrasp=True):
        rotation, position = self._payload['pregrasp' if pregrasp else 'grasp']
        return pin.SE3(rotation, position)

    def door_frame(self, i):
        return self._payload['door_center'], self._payload['door_normal']

    def run(self, job):
        op, payload = job
        self._payload = payload
        self._world = wire_to_world(payload['world'])
        if op == 'retreat':
            self.retreat_mode = payload['retreat_mode']
            self.retreat_origins = {payload['i']: payload['origin'].copy()} if payload['origin'] is not None else {}
            answer = self.retreat_function(self, payload['i'], payload['joint'], payload['mapping'], payload['scale'])
            return answer, self.retreat_origins.get(payload['i'])
        if op == 'tracking':
            if payload['escape']:
                return self.escape_edge(payload['joint'], payload['trial'], self._world)
            return self.edge(payload['joint'], payload['trial'], self._world, allow_handle=payload['allow_handle'])
        if op == 'candidates':
            q = payload['joint'].copy()
            for action in payload['actions'][:3]:
                if action[6] >= 0:
                    return False
                trial = q.copy()
                trial[:7] += payload['mapping'] @ action[:6]
                if not self.edge(q, trial, self._world, allow_handle=True):
                    return False
                q = trial
            return True
        if op == 'profile':
            return self.profile(payload['joint'], self._world)
        raise ValueError('Unknown geometry operation: ' + op)


def _read(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError('Geometry process stream closed')
        chunks.append(chunk)
        size -= len(chunk)
    return b''.join(chunks)


def receive(stream):
    size = struct.unpack('!Q', _read(stream, 8))[0]
    if size > 64 * 1024 * 1024:
        raise RuntimeError('Unexpected geometry message size')
    return pickle.loads(_read(stream, size))


def send(stream, value):
    data = pickle.dumps(value, protocol=5)
    stream.write(struct.pack('!Q', len(data)))
    stream.write(data)
    stream.flush()


def worker():
    geometry = Geometry(receive(sys.stdin.buffer))
    send(sys.stdout.buffer, dict(ready=True, pid=os.getpid()))
    while True:
        try:
            job = receive(sys.stdin.buffer)
        except EOFError:
            return
        if job is None:
            return
        try:
            send(sys.stdout.buffer, (True, geometry.run(job)))
        except BaseException:
            import traceback
            send(sys.stdout.buffer, (False, traceback.format_exc()))


class GeometryPool:
    """Persistent CPU subprocesses; no fork of the simulator or CUDA context."""
    def __init__(self, config, workers=4, record_path=None):
        if not 1 <= workers <= 4:
            raise ValueError('CPU geometry workers must be 1..4')
        self.processes = []
        self.commands = []
        self.locks = []
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.record_path = Path(record_path) if record_path else None
        self.identity = dict(kernel_sha256=config['hashes'],urdf_sha256=config['urdf_sha256'],
                             geometry_names=config['geometry_names'],collision_pairs=config['collision_pairs'],
                             numerical_threads={key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
        self.jobs = 0
        self.closed = False
        try:
            for index in range(workers):
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--parent-pid', str(os.getpid())]
                # Preserve the parent's numerical runtime configuration. Changing
                # BLAS thread settings can change last bits of candidate solves.
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
                error = open(str(self.record_path) + f'.worker{index}.stderr.log', 'ab') if self.record_path else subprocess.DEVNULL
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error, env=environment)
                if error != subprocess.DEVNULL:
                    error.close()
                self.processes.append(process)
                self.commands.append(command)
                self.locks.append(threading.Lock())
                send(process.stdin, config)
                if not receive(process.stdout).get('ready'):
                    raise RuntimeError('Geometry process did not become ready')
            self._record()
        except BaseException:
            self.close()
            raise

    def _record(self):
        if self.record_path:
            value = dict(parent_pid=os.getpid(), processes=[dict(pid=p.pid, command=c) for p, c in zip(self.processes, self.commands)],
                         closed=self.closed, submitted_jobs=self.jobs, simulator=False, gpu=False, training=False,
                         identity=self.identity)
            temporary = self.record_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(self.record_path)

    def _run(self, index, job):
        with self.locks[index]:
            process = self.processes[index]
            send(process.stdin, job)
            okay, result = receive(process.stdout)
            if not okay:
                raise RuntimeError('CPU geometry process failed:\n' + result)
            return result

    def run_many(self, jobs):
        if self.closed:
            raise RuntimeError('Geometry pool is closed')
        futures = [self.executor.submit(self._run, index % len(self.processes), job) for index, job in enumerate(jobs)]
        self.jobs += len(jobs)
        # Retrieve in original clone order, not completion order.
        return [future.result() for future in futures]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True)
        for process, command in zip(self.processes, self.commands):
            try:
                send(process.stdin, None)
            except (OSError, EOFError):
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                path = Path('/proc') / str(process.pid) / 'cmdline'
                actual = [s.decode() for s in path.read_bytes().split(b'\0') if s] if path.exists() else []
                if actual != command:
                    raise RuntimeError('Refusing cleanup: geometry PID identity mismatch')
                process.terminate()
                process.wait(timeout=10)
            for stream in [process.stdin, process.stdout]:
                if stream:
                    stream.close()
        self._record()


def exactly_equal(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc')
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exactly_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(exactly_equal(a, b) for a, b in zip(left, right))
    return left == right


if __name__ == '__main__':
    if '--worker' not in sys.argv:
        raise SystemExit('This file is a CPU geometry worker entry point')
    worker()
