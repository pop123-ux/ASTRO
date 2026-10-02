from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from orbit import Orbit, OrbitGPT, OrbitGPTConfig
from orbit.model import RotaryAttention
from orbit.optimizer import inverse_metric_power

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import paper_campaign as paper_campaign  # noqa: E402


def tiny_config(**overrides):
    values = dict(
        vocab_size=67,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
    )
    values.update(overrides)
    return OrbitGPTConfig(**values)


def test_rope_score_and_metric_use_same_causal_sign():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    q = torch.zeros(1, 1, 9, 2)
    k = torch.zeros(1, 1, 9, 2)
    q_vec = torch.tensor([0.8, -1.1])
    k_vec = torch.tensor([1.7, 0.4])
    q[0, 0, 8] = q_vec
    k[0, 0, 0] = k_vec

    score = attn._apply_rope(q)[0, 0, 8] @ attn._apply_rope(k)[0, 0, 0]
    r = attn._causal_relative_rotation(8)[0, 0]
    assert torch.allclose(score, q_vec @ (r @ k_vec), atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        k_cov = torch.tensor([[4.0, 1.3], [1.3, 0.7]])
        q_cov = torch.tensor([[1.1, -0.6], [-0.6, 3.2]])
        attn.orbit_k_second_moment.copy_(k_cov.view(1, 1, 2, 2))
        attn.orbit_q_second_moment.copy_(q_cov.view(1, 1, 2, 2))

    mq, mk = attn.orbit_metrics((8,), rotate=True, eps=0.0)
    assert torch.allclose(mq[0, 0], r @ k_cov @ r.T, atol=1e-6, rtol=1e-6)
    assert torch.allclose(mk[0, 0], r.T @ q_cov @ r, atol=1e-6, rtol=1e-6)



def test_autograd_query_score_gradient_matches_causal_rotation():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    q = torch.zeros(1, 1, 9, 2, requires_grad=True)
    k = torch.zeros(1, 1, 9, 2)
    with torch.no_grad():
        q[0, 0, 8] = torch.tensor([0.8, -1.1])
        k[0, 0, 0] = torch.tensor([1.7, 0.4])

    score = attn._apply_rope(q)[0, 0, 8] @ attn._apply_rope(k)[0, 0, 0]
    (grad_q,) = torch.autograd.grad(score, q)

    r = attn._causal_relative_rotation(8)[0, 0]
    expected = r @ k[0, 0, 0]
    assert torch.allclose(grad_q[0, 0, 8], expected, atol=1e-6, rtol=1e-6)

def test_negative_relative_distance_is_rejected():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    with pytest.raises(ValueError, match="causal distances"):
        attn.orbit_metrics((-1,), rotate=True)



@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_inverse_metric_power_matches_float64_reference_across_scales(power):
    torch.manual_seed(19)
    scales = (1e-8, 1e-4, 1.0, 1e4, 1e8)
    metrics = []
    for scale in scales:
        q, _ = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))
        values = torch.tensor([scale, scale * 1e3], dtype=torch.float64)
        metrics.append((q * values.unsqueeze(0)) @ q.T)
    metric64 = torch.stack(metrics)
    got, _ = inverse_metric_power(
        metric64.float(),
        power=power,
        eps=1e-12,
        condition_cap=1e6,
    )
    values, vectors = torch.linalg.eigh(metric64)
    reference = (
        vectors
        * values.pow(-0.5 * power).unsqueeze(-2)
    ) @ vectors.transpose(-1, -2)
    assert torch.allclose(got.double(), reference, atol=2e-3, rtol=2e-3)


def test_joint_frobenius_restore_preserves_qk_budget():
    model = OrbitGPT(tiny_config(n_head=1, n_embd=16))
    optimizer = Orbit(model, lr=0.01, adamw_lr=1e-3)
    pair = model.orbit_qk_pairs()[0]
    with torch.no_grad():
        pair["module"].orbit_q_second_moment.copy_(
            torch.tensor([[5.0, 1.4], [1.4, 0.8]])
            .view(1, 1, 2, 2)
            .repeat(1, pair["module"].n_freq, 1, 1)
        )
        pair["module"].orbit_k_second_moment.copy_(
            torch.tensor([[0.9, -0.3], [-0.3, 3.0]])
            .view(1, 1, 2, 2)
            .repeat(1, pair["module"].n_freq, 1, 1)
        )
    torch.manual_seed(23)
    q_update = torch.randn_like(pair["q"])
    k_update = torch.randn_like(pair["k"])
    before = q_update.float().square().sum() + k_update.float().square().sum()
    q_after, k_after = optimizer._precondition_pair(pair, q_update, k_update)
    after = q_after.float().square().sum() + k_after.float().square().sum()
    assert torch.allclose(after, before, atol=2e-4, rtol=2e-6)


def test_paper_identity_control_matches_paper_muon_update():
    config = tiny_config(n_head=1, n_embd=16)
    torch.manual_seed(31)
    orbit_model = OrbitGPT(config)
    muon_model = OrbitGPT(config)
    muon_model.load_state_dict(orbit_model.state_dict())

    recipe = {"lr": 0.01, "scalar_lr_mult": 0.1, "weight_decay": 0.02}
    orbit_opt = Orbit(
        orbit_model,
        variant="orbit_identity",
        lr=recipe["lr"],
        adamw_lr=recipe["lr"] * recipe["scalar_lr_mult"],
        weight_decay=recipe["weight_decay"],
    )
    muon_opt = paper_campaign.paper_build_optimizer("muon", muon_model, recipe)

    generator = torch.Generator().manual_seed(37)
    for _ in range(5):
        x = torch.randint(0, config.vocab_size, (2, 12), generator=generator)
        orbit_loss = orbit_model(x, labels=x).loss
        muon_loss = muon_model(x, labels=x).loss
        assert torch.equal(orbit_loss, muon_loss)
        orbit_loss.backward()
        muon_loss.backward()
        orbit_opt.step()
        muon_opt.step()
        orbit_opt.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)

    for orbit_param, muon_param in zip(orbit_model.parameters(), muon_model.parameters()):
        assert torch.equal(orbit_param, muon_param)

def test_checkpointing_updates_statistics_once_per_step():
    model = OrbitGPT(tiny_config(gradient_checkpointing=True))
    Orbit(model, lr=0.01, adamw_lr=1e-3)
    x = torch.randint(0, model.config.vocab_size, (2, 12))
    model(x, labels=x).loss.backward()
    assert int(model.blocks[0].attn.orbit_stats_seen) == 1


def test_orbit_statistics_survive_model_checkpoint_roundtrip():
    config = tiny_config()
    model = OrbitGPT(config)
    Orbit(model, lr=0.01, adamw_lr=1e-3)
    x = torch.randint(0, config.vocab_size, (2, 12))
    model(x, labels=x)

    state = model.state_dict()
    assert "blocks.0.attn.orbit_q_second_moment" in state
    assert "blocks.0.attn.orbit_k_second_moment" in state
    assert "blocks.0.attn.orbit_stats_seen" in state

    restored = OrbitGPT(config)
    restored.load_state_dict(state)
    assert torch.equal(restored.blocks[0].attn.orbit_q_second_moment, model.blocks[0].attn.orbit_q_second_moment)
    assert torch.equal(restored.blocks[0].attn.orbit_k_second_moment, model.blocks[0].attn.orbit_k_second_moment)
    assert torch.equal(restored.blocks[0].attn.orbit_stats_seen, model.blocks[0].attn.orbit_stats_seen)


def test_standard_scheduler_scales_auxiliary_path():
    model = OrbitGPT(tiny_config())
    optimizer = Orbit(model, lr=0.02, adamw_lr=0.004)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    before = optimizer.auxiliary_lr()
    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)
    assert optimizer.auxiliary_lr() == pytest.approx(before * 0.1)




def test_paper_lr_schedule_scales_orbit_matrix_and_auxiliary_paths_together():
    model = OrbitGPT(tiny_config())
    optimizer = Orbit(model, lr=0.02, adamw_lr=0.004)
    paper_campaign.schedule_optimizer_lrs(optimizer, 0.25)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    assert optimizer.auxiliary_lr() == pytest.approx(0.001)


def test_identity_matches_paper_muon_under_the_paper_lr_schedule():
    config = tiny_config(n_head=1, n_embd=16, bias=False)
    torch.manual_seed(41)
    orbit_model = OrbitGPT(config)
    muon_model = OrbitGPT(config)
    muon_model.load_state_dict(orbit_model.state_dict())

    recipe = {"lr": 0.01, "scalar_lr_mult": 0.1, "weight_decay": 0.02}
    orbit_opt = Orbit(
        orbit_model,
        variant="orbit_identity",
        lr=recipe["lr"],
        adamw_lr=recipe["lr"] * recipe["scalar_lr_mult"],
        weight_decay=recipe["weight_decay"],
    )
    muon_opt = paper_campaign.paper_build_optimizer("muon", muon_model, recipe)

    generator = torch.Generator().manual_seed(43)
    for step in range(5):
        factor = 0.2 + 0.15 * step
        paper_campaign.schedule_optimizer_lrs(orbit_opt, factor)
        paper_campaign.schedule_optimizer_lrs(muon_opt, factor)

        x = torch.randint(0, config.vocab_size, (2, 12), generator=generator)
        orbit_loss = orbit_model(x, labels=x).loss
        muon_loss = muon_model(x, labels=x).loss
        assert torch.equal(orbit_loss, muon_loss)

        orbit_loss.backward()
        muon_loss.backward()
        orbit_opt.step()
        muon_opt.step()
        orbit_opt.zero_grad(set_to_none=True)
        muon_opt.zero_grad(set_to_none=True)

    for orbit_param, muon_param in zip(orbit_model.parameters(), muon_model.parameters()):
        assert torch.equal(orbit_param, muon_param)

def _ddp_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        config = tiny_config()
        model = OrbitGPT(config)
        optimizer = Orbit(model, lr=0.01, adamw_lr=1e-3)

        # Reproduce DDP's relevant invariant directly: each rank sees different data,
        # gradients are averaged before the optimizer step, and ORBIT must also make
        # its forward second-moment state identical across ranks. Avoiding the DDP
        # wrapper keeps this regression focused on ORBIT and avoids backend-specific
        # DDP destructor races in short-lived CI worker processes.
        for step in range(3):
            generator = torch.Generator().manual_seed(10_000 + 100 * step + rank)
            x = torch.randint(0, config.vocab_size, (2, 12), generator=generator)
            loss = model(x, labels=x).loss
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(world_size)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        q_weight = model.blocks[0].attn.q_proj.weight.detach()
        q_moment = model.blocks[0].attn.orbit_q_second_moment.detach()
        weights = [torch.empty_like(q_weight) for _ in range(world_size)]
        moments = [torch.empty_like(q_moment) for _ in range(world_size)]
        dist.all_gather(weights, q_weight)
        dist.all_gather(moments, q_moment)

        for other in weights[1:]:
            assert torch.allclose(weights[0], other, atol=1e-7, rtol=1e-7)
        for other in moments[1:]:
            assert torch.allclose(moments[0], other, atol=1e-7, rtol=1e-7)

        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_ddp_equivalent_sync_keeps_orbit_state_and_weights_identical():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_ddp_worker, args=(2, port), nprocs=2, join=True)
