"""Reference implementations of the optimizers ASTRO is measured against.

Kept in-repo rather than pulled from pip so that every optimizer in a benchmark
table runs on the same PyTorch version, the same routing policy and the same
tuning budget. Version skew between an installed baseline and the proposed method
is a silent way to manufacture a win.
"""

from astro.baselines.adaptive import AdEMAMix, CautiousAdamW
from astro.baselines.hyperball import Hyperball
from astro.baselines.muon import Muon, NorMuon
from astro.baselines.soap import SOAP

__all__ = ["SOAP", "AdEMAMix", "CautiousAdamW", "Hyperball", "Muon", "NorMuon"]
