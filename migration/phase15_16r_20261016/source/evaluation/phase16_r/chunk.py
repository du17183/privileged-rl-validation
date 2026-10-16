"""Peek next base action exactly once; committing never re-infers the same tick."""
import time
import numpy as np
from multiprocessing.connection import Client
from experiments.phase16_r_residual_sac.common import AUTH

class ChunkClient:
    def __init__(self, socket, n, seed):
        deadline = time.monotonic()+900
        while True:
            try:
                self.conn = Client(str(socket), family='AF_UNIX', authkey=AUTH)
                break
            except (FileNotFoundError,ConnectionRefusedError):
                if time.monotonic()>deadline: raise
                time.sleep(.5)
        self.n, self.seed = n, seed
        self.calls = []
        self.configure()

    def rpc(self, request):
        self.conn.send(request)
        r = self.conn.recv()
        if 'error' in r: raise RuntimeError(r['error'])
        return r

    def load(self, path):
        self.meta = self.rpc(dict(op='load',checkpoints={'normal':str(path)}))['meta']['normal']
        self.mean = np.asarray(self.meta['mean'],np.float32)
        self.std = np.asarray(self.meta['std'],np.float32)
        assert self.mean.shape==(39,) and np.all(self.std>0)
        self.configure()

    def configure(self):
        self.cache = np.zeros((self.n,10,7),np.float32)
        self.origin = np.full(self.n,-1000,int)
        self.episode = np.zeros(self.n,int)
        self.last_commit = np.full(self.n,-1,int)
        self.ready_tick = np.full(self.n,-1,int)
        self.resets = 0

    def reset_ids(self, ids):
        self.origin[ids]=-1000
        self.episode[ids]+=1
        self.last_commit[ids]=-1
        self.ready_tick[ids]=-1
        self.cache[ids]=0
        self.resets+=len(ids)

    def peek(self, raw, age):
        need = (age-self.origin>=5)
        if np.any(need):
            ids=np.flatnonzero(need)
            seeds=self.seed+ids*1000003+self.episode[ids]*1009+age[ids]*13
            start=time.perf_counter()
            r=self.rpc(dict(op='infer',module='normal',raw=raw[ids],noise_seeds=seeds.tolist(),horizon=10))
            if r['actions'].shape!=(len(ids),10,7): raise RuntimeError('Wrong base chunk')
            self.cache[ids]=r['actions']; self.origin[ids]=age[ids]
            self.calls.append(dict(batch=len(ids),latency_s=r['latency_s'],service_s=r['service_s'],rpc_s=time.perf_counter()-start,clip_fraction=r['clip_fraction']))
        cursor=age-self.origin
        if np.any((cursor<0)|(cursor>4)): raise RuntimeError('Chunk cursor skipped/repeated')
        self.ready_tick=age.copy()
        base=self.cache[np.arange(self.n),cursor].copy()
        z=2/np.pi*np.arctan((raw-self.mean)/self.std)
        ctx=np.concatenate((z,base,self.cache.reshape(self.n,-1),np.eye(5,dtype=np.float32)[cursor]),axis=-1).astype(np.float32)
        return ctx,base,cursor

    def commit(self, age):
        if not np.array_equal(age,self.ready_tick) or np.any(age!=self.last_commit+1):
            raise RuntimeError('Physical tick and base chunk commit misaligned')
        self.last_commit=age.copy()

    def stop(self):
        self.rpc(dict(op='stop'));self.conn.close()

    def disconnect(self):
        self.rpc(dict(op='disconnect'));self.conn.close()
