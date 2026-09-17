from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable
import torch
from torch import nn

class ParamKind(str,Enum):
    MATRIX='matrix'; CONV='conv'; DEPTHWISE='depthwise'; STEM='stem'; VECTOR='vector'; TABLE='table'
    @property
    def is_spectral(self): return self in (ParamKind.MATRIX,ParamKind.CONV)

@dataclass(frozen=True)
class ParamSpec:
    name:str; kind:ParamKind; shape:tuple[int,...]; reason:str
    @property
    def is_spectral(self): return self.kind.is_spectral

def classify_parameter(name:str,param:torch.Tensor,*,min_dim:int=8,stem_in_chans:int=4,table_patterns:Iterable[str]=('embed','classifier','logit','lm_head'))->ParamSpec:
    shape=tuple(param.shape); lowered=name.lower()
    if param.ndim<=1: return ParamSpec(name,ParamKind.VECTOR,shape,'ndim<=1')
    if any(p in lowered for p in table_patterns): return ParamSpec(name,ParamKind.TABLE,shape,'embedding/head name')
    if param.ndim==4:
        out_ch,in_ch=shape[:2]
        if in_ch==1 and out_ch>1: return ParamSpec(name,ParamKind.DEPTHWISE,shape,'depthwise')
        if in_ch<=stem_in_chans: return ParamSpec(name,ParamKind.STEM,shape,'stem')
        return ParamSpec(name,ParamKind.CONV,shape,'dense convolution')
    if param.ndim==2:
        if min(shape)<min_dim: return ParamSpec(name,ParamKind.VECTOR,shape,'too thin')
        return ParamSpec(name,ParamKind.MATRIX,shape,'dense operator')
    folded=(shape[0],int(torch.tensor(shape[1:]).prod()))
    if shape[1]==1: return ParamSpec(name,ParamKind.DEPTHWISE,shape,'grouped/depthwise')
    if min(folded)<min_dim: return ParamSpec(name,ParamKind.VECTOR,shape,'too thin')
    return ParamSpec(name,ParamKind.CONV,shape,'folded operator')

def classify_module(module:nn.Module,*,detect_head:bool=True,**kwargs:object)->dict[str,ParamSpec]:
    specs={n:classify_parameter(n,p,**kwargs) for n,p in module.named_parameters() if p.requires_grad}
    name_of={id(p):n for n,p in module.named_parameters()}
    for child in module.modules():
        if isinstance(child,nn.Embedding) and id(child.weight) in name_of:
            n=name_of[id(child.weight)]; specs[n]=ParamSpec(n,ParamKind.TABLE,tuple(child.weight.shape),'nn.Embedding')
    if detect_head:
        linears=[c for c in module.modules() if isinstance(c,nn.Linear)]
        if len(specs)>1 and linears and id(linears[-1].weight) in name_of:
            n=name_of[id(linears[-1].weight)]; specs[n]=ParamSpec(n,ParamKind.TABLE,tuple(linears[-1].weight.shape),'last Linear classifier head')
    return specs

def matrix_view(tensor:torch.Tensor)->torch.Tensor:
    return tensor.reshape(tensor.shape[0],-1) if tensor.ndim>2 else tensor

FUSED_PATTERNS=('c_attn','qkv','in_proj','query_key_value')
def fused_block_sizes(name:str,tensor:torch.Tensor,*,detect_by_shape:bool=True)->tuple[int,...]:
    if tensor.ndim<2: return (tensor.shape[0],) if tensor.ndim else ()
    rows=tensor.shape[0]; cols=int(tensor.reshape(rows,-1).shape[1])
    if not (any(p in name.lower() for p in FUSED_PATTERNS) or detect_by_shape): return (rows,)
    excess=rows-cols
    if excess>0 and excess%2==0:
        kv=excess//2
        if kv>0 and cols%kv==0: return (cols,kv,kv)
    return (rows,)

def fused_block_count(name:str,tensor:torch.Tensor,**kwargs:object)->int:
    return len(fused_block_sizes(name,tensor,**kwargs))
