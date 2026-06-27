"""
tests/test_awd.py
==================
Unit tests for the AWD optimizer.
Run: pytest tests/
"""

import pytest
import torch
import torch.nn as nn
from vellum import AWD


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

def make_data():
    torch.manual_seed(1)
    return torch.randn(32, 8), torch.randn(32, 1)


# ── Basic functionality ───────────────────────────────────────────────────────

def test_step_runs():
    model     = make_model()
    optimizer = AWD(model.parameters(), lr=1e-3)
    x, y      = make_data()
    loss      = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()                          # should not raise


def test_loss_decreases():
    """AWD should reduce loss on a simple regression problem."""
    torch.manual_seed(42)
    model     = make_model()
    optimizer = AWD(model.parameters(), lr=1e-3)
    x, y      = make_data()

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_rho_bounded():
    """Sign coherence ρ must stay in [0, 1] after many steps."""
    model     = make_model()
    optimizer = AWD(model.parameters(), lr=1e-3)
    x, y      = make_data()

    for _ in range(30):
        optimizer.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        optimizer.step()

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p in optimizer.state and "rho" in optimizer.state[p]:
                rho = optimizer.state[p]["rho"]
                assert rho.min().item() >= 0.0, "ρ below 0"
                assert rho.max().item() <= 1.0, "ρ above 1"


def test_adhesion_stats_populated():
    """get_adhesion_stats() should return valid keys after one step."""
    model     = make_model()
    optimizer = AWD(model.parameters(), lr=1e-3)
    x, y      = make_data()

    optimizer.zero_grad()
    nn.MSELoss()(model(x), y).backward()
    optimizer.step()

    stats = optimizer.get_adhesion_stats()
    assert "mean_rho"        in stats
    assert "shadow_fraction" in stats
    assert 0.0 <= stats["mean_rho"]        <= 1.0
    assert 0.0 <= stats["shadow_fraction"] <= 1.0


def test_shadow_field_freezes_params():
    """
    With very short patience, params should eventually enter shadow field
    and stop updating.
    """
    model = make_model()
    optimizer = AWD(
        model.parameters(),
        lr=1e-2,
        shadow_threshold=0.99,   # almost everything fails threshold
        shadow_patience=5,
    )
    x, y = make_data()

    for _ in range(20):
        optimizer.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        optimizer.step()

    stats = optimizer.get_adhesion_stats()
    # With threshold=0.99, some params should be shadowed
    assert stats["shadow_params"] > 0, "Expected some params in shadow field"


def test_release_shadow():
    """release_shadow() should clear all shadow masks."""
    model = make_model()
    optimizer = AWD(
        model.parameters(),
        lr=1e-2,
        shadow_threshold=0.99,
        shadow_patience=3,
    )
    x, y = make_data()

    for _ in range(10):
        optimizer.zero_grad()
        nn.MSELoss()(model(x), y).backward()
        optimizer.step()

    optimizer.release_shadow()
    stats = optimizer.get_adhesion_stats()
    assert stats["shadow_params"] == 0, "Shadow field should be empty after release"


def test_weight_decay():
    """Weight decay should produce smaller parameter norms than without."""
    torch.manual_seed(0)
    model_wd  = make_model()
    model_no  = make_model()
    model_no.load_state_dict(model_wd.state_dict())

    opt_wd = AWD(model_wd.parameters(), lr=1e-3, weight_decay=1e-2)
    opt_no = AWD(model_no.parameters(), lr=1e-3, weight_decay=0.0)
    x, y   = make_data()

    for _ in range(100):
        for opt, mdl in [(opt_wd, model_wd), (opt_no, model_no)]:
            opt.zero_grad()
            nn.MSELoss()(mdl(x), y).backward()
            opt.step()

    norm_wd = sum(p.norm().item() for p in model_wd.parameters())
    norm_no = sum(p.norm().item() for p in model_no.parameters())
    assert norm_wd < norm_no, "Weight decay should produce smaller norms"


def test_no_grad_required():
    """Params with requires_grad=False should be silently skipped."""
    model = make_model()
    for p in list(model.parameters())[:1]:
        p.requires_grad_(False)

    optimizer = AWD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    x, y = make_data()
    optimizer.zero_grad()
    nn.MSELoss()(model(x), y).backward()
    optimizer.step()   # should not raise


def test_sparse_grad_raises():
    """AWD should raise on sparse gradients."""
    embedding = nn.Embedding(10, 4, sparse=True)
    optimizer = AWD(embedding.parameters(), lr=1e-3)
    idx  = torch.tensor([0, 1, 2])
    loss = embedding(idx).sum()
    loss.backward()
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()


def test_print_adhesion_report_no_crash(capsys):
    """print_adhesion_report() should not raise and should produce output."""
    model     = make_model()
    optimizer = AWD(model.parameters(), lr=1e-3)
    x, y      = make_data()

    optimizer.zero_grad()
    nn.MSELoss()(model(x), y).backward()
    optimizer.step()

    optimizer.print_adhesion_report()
    captured = capsys.readouterr()
    assert "AWD Adhesion Report" in captured.out
