from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import torch

MUON_QUINTIC: tuple[float,float,float]=(3.4445,-4.7750,2.0315)
@dataclass(frozen=True)
class SpectralFilter:
    steps:tuple[tuple[float,float,float],...]; name:str='muon'
    def response(self,sigma:torch.Tensor)->torch.Tensor:
        s=sigma.clone().to(torch.float64)
        for a,b,c in self.steps:
            s2=s*s; s=a*s+b*s*s2+c*s*s2*s2
        return s
    def __call__(self,matrix:torch.Tensor,eps:float=1e-7)->torch.Tensor:
        if matrix.ndim<2: raise ValueError(f'expected matrix, got {tuple(matrix.shape)}')
        x=matrix.to(torch.float32); tr=x.size(-2)>x.size(-1)
        if tr:x=x.mT
        x=x/(x.norm(dim=(-2,-1),keepdim=True)+eps)
        for a,b,c in self.steps:
            g=x@x.mT; x=a*x+(b*g+c*(g@g))@x
        if tr:x=x.mT
        return x.to(matrix.dtype)

def muon_filter(steps:int=5)->SpectralFilter:return SpectralFilter(tuple([MUON_QUINTIC]*steps),f'muon{steps}')

def power_iteration(matrix:torch.Tensor,iters:int=3)->torch.Tensor:
    x=matrix.to(torch.float32); v=x.sum(dim=-2,keepdim=True).mT
    if bool((v.abs().amax(dim=-2,keepdim=True)<1e-12).any()):v=torch.ones_like(v)
    v=v/(v.norm(dim=-2,keepdim=True)+1e-12)
    sigma=None
    for _ in range(iters):
        u=x@v;u=u/(u.norm(dim=-2,keepdim=True)+1e-12);v=x.mT@u;sigma=v.norm(dim=-2,keepdim=True);v=v/(sigma+1e-12)
    return sigma.squeeze(-1).squeeze(-1)

_POLAR_SOLVED_1E3=((5.741408,-17.016317,12.623472),(4.240444,-6.859093,2.787935),(4.186216,-6.613335,2.669455),(3.958440,-5.645446,2.206946),(2.621392,-2.503740,.833594),(1.889525,-1.266059,.376621),(1.777582,-1.055164,.277581))
_POLAR_CACHE={(s,1e-3):_POLAR_SOLVED_1E3[:s] for s in range(1,8)}
def polar_filter(steps:int=5,lo:float=1e-3)->SpectralFilter:
    if (steps,lo) not in _POLAR_CACHE:raise KeyError(f'no solved coefficients for steps={steps}, lo={lo}')
    return SpectralFilter(_POLAR_CACHE[(steps,lo)],f'polar{steps}@{lo:g}')

_DEADZONE_CACHE={}
def _register(tau:float,steps:int,coeffs:Sequence[Sequence[float]])->None:_DEADZONE_CACHE[(round(float(tau),4),int(steps))]=tuple((float(a),float(b),float(c)) for a,b,c in coeffs)
_register(.1,10,[(1.569795,-.641675,-.660823),(.945208,2.475732,-5.815943),(.697653,9.661669,-23.437894),(2.482474,-5.161337,3.950530),(.8732,-.459821,-.243791),(-.064389,6.807862,-4.566212),(1.128763,3.550339,-22.976123),(1.118623,-.25404,-3.52384),(-.000039,7.459515,8.326961),(1.891771,.498864,-3.189934)])
def deadzone_filter(tau:float,steps:int=7)->SpectralFilter:
    key=(round(float(tau),4),int(steps))
    if key not in _DEADZONE_CACHE:raise KeyError(f'no cached dead-zone filter for tau={tau}, steps={steps}')
    return SpectralFilter(_DEADZONE_CACHE[key],f'deadzone{tau}x{steps}')

def solve_deadzone_filter(*args,**kwargs):
    raise NotImplementedError('Offline coefficient fitting is intentionally not run during training; use the source research implementation.')
def solve_polar_filter(*args,**kwargs):
    raise NotImplementedError('Offline coefficient fitting is intentionally not run during training; use the source research implementation.')
