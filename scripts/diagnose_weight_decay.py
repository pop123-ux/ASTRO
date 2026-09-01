"""Is ASTRO receiving less weight decay than Muon at the same nominal setting?

Muon:  param *= (1 - lr*wd)                       -- every coordinate
ASTRO: cautious weight decay, only where update*param > 0

If the masked fraction is well below 1, then at one nominal `weight_decay` the
two optimizers are running at materially different *effective* regularisation,
and the difference compounds with step count -- which is the shape of the
widening gap we measured at 300/900/2700 steps.

Measures the fraction directly on a real transformer, and the resulting weight
norms, on CPU.
"""
import sys

import torch

sys.path.insert(0, "/home/user/knsa-knee-normality-kaggle/astro/scripts")
sys.path.insert(0, "/home/user/knsa-knee-normality-kaggle/astro/src")

import astro_lab
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
CONFIG = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64, vocab_size=512)
STEPS = 60
DRAW = {"lr": 0.0144, "weight_decay": 0.02, "scalar_lr_mult": 0.1, "beta2": 0.95}


def spectral_norm_total(model):
    total = 0.0
    for name, param in model.named_parameters():
        if param.ndim == 2 and "wte" not in name and "wpe" not in name:
            total += float(param.detach().pow(2).sum())
    return total ** 0.5


def run(name, patch_fraction=None):
    torch.manual_seed(0)
    model = GPT2LMHeadModel(CONFIG)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    optimizer = astro_lab.build_optimizer(name, model, DRAW)
    generator = torch.Generator().manual_seed(7)
    fractions = []

    original = astro_lab.apply_weight_decay

    def spy(param, update, rate, *, cautious):
        if cautious and rate:
            mask = (update * param > 0)
            fractions.append(float(mask.sum()) / mask.numel())
        return original(param, update, rate, cautious=cautious)

    astro_lab.apply_weight_decay = spy
    try:
        for _ in range(STEPS):
            ids = torch.randint(0, 512, (4, 64), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            model(ids, labels=ids).loss.backward()
            optimizer.step()
    finally:
        astro_lab.apply_weight_decay = original
    return spectral_norm_total(model), fractions


muon_norm, _ = run("muon")
astro_norm, fractions = run("astro")
plain_norm, _ = run("astro_plain_wd")

print(f"steps = {STEPS}, lr = {DRAW['lr']}, weight_decay = {DRAW['weight_decay']}\n")
if fractions:
    mean = sum(fractions) / len(fractions)
    print(f"cautious mask keeps {mean:.4f} of coordinates on average "
          f"(min {min(fractions):.4f}, max {max(fractions):.4f})")
    print(f"  -> ASTRO receives about {mean:.2f}x Muon's weight decay\n")

print("total Frobenius norm of the spectral weights after training:")
print(f"  muon                      {muon_norm:.4f}")
print(f"  astro (cautious wd)       {astro_norm:.4f}   "
      f"({astro_norm / muon_norm:.4f}x muon)")
print(f"  astro (plain wd)          {plain_norm:.4f}   "
      f"({plain_norm / muon_norm:.4f}x muon)")
print("\nIf the cautious row sits above muon and the plain row sits on it, the")
print("comparison has been running the two at different effective regularisation,")
print("and the discrepancy grows with step count.")
