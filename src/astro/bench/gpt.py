"""The GPT-2 architecture, as implemented by nanoGPT.

Vendored from Andrej Karpathy's nanoGPT (``model.py``, MIT licence) with the
training/generation/HuggingFace-loading machinery removed and nothing else
changed. Keeping the architecture byte-for-byte faithful matters: the point of
this benchmark is to compare optimizers on *the* reference decoder-only
transformer, not on a lookalike whose differences could explain a result.

    https://github.com/karpathy/nanoGPT

What is preserved
-----------------
Pre-norm blocks; fused QKV projection (``c_attn``, ``n_embd -> 3*n_embd``);
4x-expansion MLP with GELU; weight tying between ``wte`` and ``lm_head``;
learned position embeddings; the scaled initialisation ``N(0, 0.02/sqrt(2L))``
on residual projections; LayerNorm with optional bias.

What is configured differently from GPT-2 (124M)
------------------------------------------------
Only ``n_layer``, ``n_head``, ``n_embd``, ``block_size`` and ``vocab_size``,
which are reduced so a run finishes on four CPU cores. Every structural choice
is nanoGPT's. See :mod:`astro.bench.corpora` for why the vocabulary is
character level.

Why weight tying matters here
-----------------------------
``wte.weight`` and ``lm_head.weight`` are the *same tensor*, so
``named_parameters()`` reports it once, under the ``wte`` name. A router that
looks for a parameter literally named ``lm_head.weight`` will not find it, and a
router that takes "the last 2-D parameter" as the head will pick the final
block's MLP projection -- a genuine hidden operator -- and wrongly exclude it
from the matrix path. :func:`astro.routing.classify_module` resolves parameters
by identity for exactly this reason, and ``tests/test_routing.py`` pins the
behaviour against this model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["GPTConfig", "GPT"]


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch's does not support ``bias=False``."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fused QKV projection."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        assert config.n_head % self.n_kv_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        # Under GQA the fused projection is (n_embd + 2*kv_dim, n_embd) rather
        # than (3*n_embd, n_embd): Q keeps every head, K and V share fewer.
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.kv_dim, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.size()
        q, k, v = self.c_attn(x).split([self.n_embd, self.kv_dim, self.kv_dim], dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with 4x expansion."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    """nanoGPT's configuration. Defaults here are the CPU benchmark size.

    GPT-2 (124M) is ``block_size=1024, vocab_size=50257, n_layer=12, n_head=12,
    n_embd=768``; the defaults below are the same architecture scaled down.
    """

    block_size: int = 64
    vocab_size: int = 96
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.0
    bias: bool = True
    #: Key/value heads. Fewer than ``n_head`` gives grouped-query attention, as
    #: used by LLaMA-2/3, Mistral, Qwen and Gemma. ``None`` means multi-head
    #: attention (GPT-2's setting), i.e. ``n_kv_head == n_head``.
    n_kv_head: int | None = None


class GPT(nn.Module):
    """Decoder-only transformer, nanoGPT's ``GPT`` minus the training utilities."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying, as in GPT-2. This makes wte.weight and lm_head.weight one
        # tensor; see the module docstring for what that means for routing.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init on residual projections, per the GPT-2 paper (section 2.3).
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        _, time = idx.size()
        assert time <= self.config.block_size, (
            f"sequence of length {time} exceeds block size {self.config.block_size}"
        )
        pos = torch.arange(0, time, dtype=torch.long, device=device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self, *, non_embedding: bool = True) -> int:
        """Parameter count, excluding position embeddings by default (nanoGPT's convention)."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.transformer.wpe.weight.numel()
        return total
