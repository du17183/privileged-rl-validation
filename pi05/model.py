"""Official OpenPI pi05 pretrained flow model; state-only modality ablation.

Read-only dependency environment. No peg policies or task-specific weights.
Only image preprocessing is bypassed; prefix/suffix, adaRMS, flow loss and
Euler sampler are the official pi05 implementation.
"""
import sys,types
from pathlib import Path
from runtime_paths import OPENPI_SOURCE, PI05_BASE
sys.path.insert(0,str(OPENPI_SOURCE / 'src'))
import torch
from torch import nn
from safetensors import safe_open
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from .action_adapter import HORIZON

BASE=PI05_BASE

class StateOnlyPi05(PI0Pytorch):
    def _preprocess_observation(self,observation,*,train=True):
        assert not observation.images and observation.state.shape[-1]==39
        return [],[],observation.tokenized_prompt,observation.tokenized_prompt_mask,observation.state

class LoRALinear(nn.Module):
    def __init__(self,base,rank=16):
        super().__init__();self.base=base;self.rank=rank
        self.in_features=base.in_features;self.out_features=base.out_features
        self.A=nn.Parameter(torch.randn(rank,base.in_features,dtype=torch.float32)*.01)
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32))
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,x):
        return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)

def build(device='cuda:0',seed=0):
    torch.manual_seed(seed)
    config=Pi0Config(pi05=True,action_horizon=HORIZON,action_dim=32,max_token_len=200,pytorch_compile_mode=None)
    net=StateOnlyPi05(config)
    expected=net.state_dict();mapped={}
    with safe_open(str(BASE),framework='pt',device='cpu') as f:
        for key in f.keys():
            alias=key.replace('paligemma.model.language_model.','paligemma.language_model.').replace('paligemma.model.vision_tower.','paligemma.vision_tower.').replace('paligemma.model.multi_modal_projector.','paligemma.multi_modal_projector.')
            dest=key if key in expected else alias
            if dest in expected:
                assert tuple(f.get_slice(key).get_shape())==tuple(expected[dest].shape),(key,dest)
                mapped[dest]=f.get_tensor(key)
    # Some safetensors exports omit tied LM head/embed weights.
    head='paligemma_with_expert.paligemma.lm_head.weight';emb='paligemma_with_expert.paligemma.language_model.embed_tokens.weight'
    if head in expected and head not in mapped and emb in mapped:mapped[head]=mapped[emb]
    for key in expected:
        if key.endswith('language_model.embed_tokens.weight') and key not in mapped and head in mapped:mapped[key]=mapped[head]
    result=net.load_state_dict(mapped,strict=False)
    if result.missing_keys or result.unexpected_keys:raise RuntimeError({'missing_count':len(result.missing_keys),'missing':result.missing_keys[:12],'unexpected':result.unexpected_keys[:12]})
    del mapped,expected
    net.requires_grad_(False)
    count=0
    for name,module in list(net.named_modules()):
        if isinstance(module,nn.Linear) and name.split('.')[-1] in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] and 'vision_tower' not in name:
            parent,leaf=name.rsplit('.',1);net.get_submodule(parent)._modules[leaf]=LoRALinear(module);count+=1
    for name,p in net.named_parameters():
        if name.startswith(('action_in_proj.','action_out_proj.','time_mlp_in.','time_mlp_out.')):
            p.data=p.data.float();p.requires_grad_(True)
    net=net.to(device);net.lora_modules=count
    return net

def observation(raw,tokens,mask,device):
    return types.SimpleNamespace(images={},image_masks={},state=torch.as_tensor(raw,device=device,dtype=torch.float32),
          tokenized_prompt=torch.as_tensor(tokens,device=device,dtype=torch.long),tokenized_prompt_mask=torch.as_tensor(mask,device=device,dtype=torch.bool))

def learned_state(net):return {k:p.detach().cpu().clone() for k,p in net.named_parameters() if p.requires_grad}

def restore(net,path):
    d=torch.load(path,map_location='cpu',weights_only=False)
    p=dict(net.named_parameters())
    if set(d['learned'])!={k for k,v in p.items() if v.requires_grad}:raise RuntimeError('Trainable parameter schema mismatch')
    with torch.no_grad():
        for k,v in d['learned'].items():p[k].copy_(v.to(p[k].device))
    return d
