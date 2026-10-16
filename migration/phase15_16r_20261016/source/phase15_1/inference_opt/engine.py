import dataclasses
import math
import os
import time
import types
from collections import OrderedDict
import numpy as np
import torch
from pi05.model import observation
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding


def gpu_prefix(net,images,imasks,tokens,masks):
    if images:raise RuntimeError('This experiment is state-only')
    embedding=net.paligemma_with_expert.embed_language_tokens(tokens)
    embedding=embedding*math.sqrt(embedding.shape[-1])
    return embedding,masks,torch.zeros_like(masks)


def gpu_suffix(net,state,noise,timestep):
    if not net.pi05:raise RuntimeError('Only frozen pi05 supported')
    t=create_sinusoidal_pos_embedding(timestep,net.action_in_proj.out_features,min_period=4e-3,
        max_period=4.0,device=timestep.device).type(timestep.dtype)
    t=torch.nn.functional.silu(net.time_mlp_in(t));t=torch.nn.functional.silu(net.time_mlp_out(t))
    embedding=net.action_in_proj(noise)
    masks=torch.ones(embedding.shape[:2],dtype=torch.bool,device=noise.device)
    attention=torch.zeros(embedding.shape[:2],dtype=embedding.dtype,device=noise.device);attention[:,0]=1
    return embedding,masks,attention,t


class ExactStateCache:
    """Cache complete tokenizations only for identical standardized state bins.

    Prefix KV is recomputed every invocation: its attention depends on state.
    Splitting SentencePiece tokens at a text boundary is not assumed equivalent.
    """
    def __init__(self, adapter, capacity=2048):
        self.adapter, self.capacity = adapter, capacity
        self.cache = OrderedDict(); self.hits = self.misses = 0

    def encode(self, raw):
        z = (raw-self.adapter.mean)/self.adapter.std
        bins = np.digitize(2/np.pi*np.arctan(z), np.linspace(-1,1,257)[:-1])-1
        ids=[]; masks=[]
        for row, value in zip(raw, bins):
            key=value.astype(np.int16).tobytes()
            if key in self.cache:
                self.hits+=1; self.cache.move_to_end(key)
            else:
                self.misses+=1
                token,mask=self.adapter.encode(row[None])
                self.cache[key]=(token[0].copy(),mask[0].copy())
                if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            token,mask=self.cache[key];ids.append(token);masks.append(mask)
        return np.stack(ids),np.stack(masks)


def unrolled(net, obs, noise, steps=10):
    images,imasks,tokens,lmasks,state=net._preprocess_observation(obs,train=False)
    embs,pads,att=net.embed_prefix(images,imasks,tokens,lmasks)
    attention=net._prepare_attention_masks_4d(make_att_2d_masks(pads,att))
    positions=torch.cumsum(pads,dim=1)-1
    net.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
    _,kv=net.paligemma_with_expert.forward(attention_mask=attention,position_ids=positions,
        past_key_values=None,inputs_embeds=[embs,None],use_cache=True)
    dt=torch.full((),-1.0/steps,dtype=torch.float32,device=noise.device)
    clock=torch.ones((),dtype=torch.float32,device=noise.device)
    value=noise
    # Original float32 time accumulation and ten Euler updates are preserved.
    for _ in range(steps):
        velocity=net.denoise_step(state,pads,kv,value,clock.expand(len(noise)))
        value=value+dt*velocity
        clock=clock+dt
    return value


class Engine:
    def __init__(self, net, adapter, backend='unroll',verify_count=2):
        self.net=net;self.adapter=ExactStateCache(adapter)
        self.backend=backend;self.graphs={};self.validated={};self.failures=[]
        self.graph_memory_budget=int(float(os.environ.get('PHASE15_1_GRAPH_MEMORY_GIB','48'))*(1<<30))
        self.graph_fallbacks=0
        self.verify_count=verify_count
        if not hasattr(net,'_phase15_1_original_denoise'):
            net._phase15_1_original_denoise=net.denoise_step
        self.original_denoise=net._phase15_1_original_denoise
        net.denoise_step=self.original_denoise
        if not hasattr(net,'_phase15_1_original_prefix'):
            net._phase15_1_original_prefix=net.embed_prefix;net._phase15_1_original_suffix=net.embed_suffix
        net.embed_prefix=net._phase15_1_original_prefix;net.embed_suffix=net._phase15_1_original_suffix
        if backend=='graph':
            net.embed_prefix=types.MethodType(gpu_prefix,net);net.embed_suffix=types.MethodType(gpu_suffix,net)
        if backend in ['compile','compile_approx']:
            net.denoise_step=torch.compile(net.denoise_step,mode='reduce-overhead',fullgraph=False,dynamic=True)

    def horizon(self,n):
        self.net.config=dataclasses.replace(self.net.config,action_horizon=n)

    def _call(self,obs,noise):
        if self.backend=='cached':
            return self.net.sample_actions('cuda:0',obs,noise=noise,num_steps=10)
        return unrolled(self.net,obs,noise)

    def _graph(self,obs,noise,key):
        if key not in self.graphs:
            if len(noise)!=1 and torch.cuda.memory_reserved()>self.graph_memory_budget:
                self.graph_fallbacks+=1
                return unrolled(self.net,obs,noise)
            static=types.SimpleNamespace(images={},image_masks={},
                state=obs.state.clone(),tokenized_prompt=obs.tokenized_prompt.clone(),
                tokenized_prompt_mask=obs.tokenized_prompt_mask.clone())
            static_noise=noise.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):out=unrolled(self.net,static,static_noise)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=unrolled(self.net,static,static_noise)
            self.graphs[key]=(graph,static,static_noise,out)
        graph,static,static_noise,out=self.graphs[key]
        static.state.copy_(obs.state);static.tokenized_prompt.copy_(obs.tokenized_prompt)
        static.tokenized_prompt_mask.copy_(obs.tokenized_prompt_mask);static_noise.copy_(noise)
        graph.replay()
        return out.clone()

    @torch.no_grad()
    def run(self,raw,seeds,horizon=10):
        self.horizon(horizon)
        tokens,mask=self.adapter.encode(raw)
        obs=observation(raw,tokens,mask,'cuda:0')
        noise=torch.as_tensor(np.stack([np.random.default_rng(int(s)).normal(size=(horizon,32)).astype(np.float32)
                                       for s in seeds]),device='cuda:0')
        key=(len(raw),horizon)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=self.backend!='graph'):
            if self.validated.get(key,{}).get('checked_changed_inputs',0)<self.verify_count:
                # Same raw/noise baseline. Different subsequent input catches stale graph buffers.
                accelerated=self.net.denoise_step
                prefix,suffix=self.net.embed_prefix,self.net.embed_suffix
                self.net.denoise_step=self.original_denoise
                self.net.embed_prefix=self.net._phase15_1_original_prefix;self.net.embed_suffix=self.net._phase15_1_original_suffix
                ref=self.net.sample_actions('cuda:0',obs,noise=noise,num_steps=10).clone()
                self.net.denoise_step=accelerated
                self.net.embed_prefix=prefix;self.net.embed_suffix=suffix
                candidate=self._graph(obs,noise,key) if self.backend=='graph' else self._call(obs,noise)
                torch.cuda.synchronize()
                error=float((ref-candidate).abs().max())
                active_error=float((ref[:,:,:7].clamp(-1,1)-candidate[:,:,:7].clamp(-1,1)).abs().max())
                # Approximate fused BF16 execution is a distinct experimental backend.
                # <=0.01 action units means <=0.5mm translation / 3mrad orientation per tick.
                # It must also pass independent closed-loop success-retention validation.
                matched=(active_error<=.01) if self.backend=='compile_approx' else torch.allclose(ref,candidate,rtol=2e-3,atol=2e-3)
                if not matched:
                    raise RuntimeError(f'Frozen inference equivalence failure: {key} {error}')
                previous=self.validated.get(key,{})
                self.validated[key]=dict(max_abs_error=max(error,previous.get('max_abs_error',0)),
                    output_shape=list(candidate.shape),checked_changed_inputs=previous.get('checked_changed_inputs',0)+1,
                    compared_to_uncompiled_frozen_reference=True,active_action_max_error=max(active_error,previous.get('active_action_max_error',0)),
                    approximate_backend=self.backend=='compile_approx',closed_loop_retention_required=self.backend=='compile_approx')
            torch.cuda.synchronize();tick=time.perf_counter()
            output=self._graph(obs,noise,key) if self.backend=='graph' else self._call(obs,noise)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        if output.shape!=(len(raw),horizon,32):raise RuntimeError('Unpredicted or padded action chunk')
        return output,elapsed

    def status(self):
        return dict(backend=self.backend,validated={str(k):v for k,v in self.validated.items()},
            cache_hits=self.adapter.hits,cache_misses=self.adapter.misses,
            gpu_memory_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_memory_reserved_bytes=torch.cuda.memory_reserved(),
            graph_memory_budget_bytes=self.graph_memory_budget,graph_memory_fallback_calls=self.graph_fallbacks,
            trained_horizon=10,prefix_kv_reused_across_states=False)
