import torch

from orbit import Orbit, OrbitGPT, OrbitGPTConfig


def tiny_model():
    torch.manual_seed(7)
    return OrbitGPT(
        OrbitGPTConfig(
            vocab_size=127,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
        )
    )


def test_statistics_are_off_for_baseline_model_by_default():
    model = tiny_model()
    x = torch.randint(0, 127, (2, 16))
    model(x, labels=x)
    assert all(int(block.attn.orbit_stats_seen) == 0 for block in model.blocks)


def test_orbit_optimizer_enables_functional_statistics():
    model = tiny_model()
    Orbit(model, lr=0.01, adamw_lr=3e-4)
    x = torch.randint(0, 127, (2, 16))
    out = model(x, labels=x)
    assert out.logits.shape == (2, 16, 127)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for block in model.blocks:
        assert int(block.attn.orbit_stats_seen) > 0
        assert torch.isfinite(block.attn.orbit_q_cov).all()
        assert torch.isfinite(block.attn.orbit_k_cov).all()


def test_rope_metric_is_spd_and_position_sensitive():
    model = tiny_model()
    attn = model.blocks[0].attn
    with torch.no_grad():
        base = torch.tensor([[5.0, 1.2], [1.2, 0.8]])
        attn.orbit_k_cov.copy_(base.view(1, 1, 2, 2).repeat(attn.n_head, attn.n_freq, 1, 1))
        attn.orbit_q_cov.copy_(base.flip(0).flip(1).view(1, 1, 2, 2).repeat(attn.n_head, attn.n_freq, 1, 1))
    mq0, _ = attn.orbit_metrics((0,), rotate=True)
    mq8, _ = attn.orbit_metrics((8,), rotate=True)
    assert torch.linalg.eigvalsh(mq0).min() > 0
    assert torch.linalg.eigvalsh(mq8).min() > 0
    assert not torch.allclose(mq0, mq8)


def test_orbit_step_is_finite_and_changes_weights():
    model = tiny_model()
    optimizer = Orbit(model, lr=0.01, adamw_lr=3e-4, functional_power=0.5)
    x = torch.randint(0, 127, (2, 16))
    before = model.blocks[0].attn.q_proj.weight.detach().clone()
    loss = model(x, labels=x).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = model.blocks[0].attn.q_proj.weight.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
    assert optimizer.diagnostics()


def test_identity_variant_runs_same_model_without_functional_metric():
    model = tiny_model()
    optimizer = Orbit(model, variant="orbit_identity")
    x = torch.randint(0, 127, (1, 8))
    model(x, labels=x).loss.backward()
    optimizer.step()
    assert optimizer.diagnostics() == {}
    assert all(int(block.attn.orbit_stats_seen) > 0 for block in model.blocks)


def test_eval_does_not_mutate_stats():
    model = tiny_model()
    Orbit(model)
    model.eval()
    seen = [int(block.attn.orbit_stats_seen) for block in model.blocks]
    with torch.no_grad():
        model(torch.randint(0, 127, (1, 8)))
    assert seen == [int(block.attn.orbit_stats_seen) for block in model.blocks]
