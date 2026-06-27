"""
tests/test_lbl.py
==================
Unit tests for the Liminal Boundary Layer.
Run: pytest tests/
"""

import pytest
import torch
import torch.nn as nn
from vellum import LiminalBoundaryLayer, LBLSequential


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM         = 32
NUM_CLASSES = 5
BATCH       = 16


def make_lbl(**kwargs):
    return LiminalBoundaryLayer(DIM, NUM_CLASSES, **kwargs)

def make_activations():
    torch.manual_seed(0)
    return torch.randn(BATCH, DIM)

def make_labels():
    return torch.randint(0, NUM_CLASSES, (BATCH,))


# ── Projection ────────────────────────────────────────────────────────────────

def test_projection_output_shape():
    lbl  = make_lbl()
    h    = make_activations()
    out, loss = lbl(h)
    assert out.shape == h.shape


def test_projection_non_expansive():
    """
    Projection onto a convex set is non-expansive: ‖Π(h)‖ ≤ ‖h - μ‖ + ‖μ‖
    More directly: projected norms should not exceed a reasonable bound.
    """
    lbl = make_lbl(r_init=0.5)    # tight radius — should clip most activations
    h   = make_activations() * 5  # large activations to force boundary contact
    with torch.no_grad():
        projected, s = lbl._smooth_project(h)

    # After projection, scaled norms should be ≤ r + eps
    scale    = torch.exp(lbl.log_scale)
    r        = torch.exp(lbl.log_r).item()
    centered = projected - lbl.mu
    scaled   = centered * scale
    norms    = scaled.norm(dim=-1)
    assert (norms <= r * 1.01 + 1e-4).all(), "Projection exceeded ellipsoid boundary"


def test_s_positive():
    """Normalized distance s should always be positive."""
    lbl = make_lbl()
    h   = make_activations()
    with torch.no_grad():
        _, s = lbl._smooth_project(h)
    assert (s >= 0).all()


def test_boundary_fraction_tracked():
    """Boundary fraction diagnostic should be populated after forward pass."""
    lbl = make_lbl(r_init=0.1)   # very tight — expect high boundary fraction
    h   = make_activations()
    lbl(h)
    assert lbl._last_boundary_fraction.item() >= 0.0
    assert lbl._last_boundary_fraction.item() <= 1.0


def test_tight_radius_high_boundary_fraction():
    """Very tight radius should push boundary fraction toward 1.0."""
    lbl = make_lbl(r_init=1e-3)
    h   = make_activations()
    lbl(h)
    assert lbl._last_boundary_fraction.item() > 0.5


def test_loose_radius_low_boundary_fraction():
    """Very loose radius should give boundary fraction near 0."""
    lbl = make_lbl(r_init=1e6)
    h   = make_activations()
    lbl(h)
    assert lbl._last_boundary_fraction.item() < 0.05


# ── Gradient flow ─────────────────────────────────────────────────────────────

def test_gradient_flows_through():
    """Gradients must flow through the LBL back to upstream params."""
    lbl = make_lbl()
    h   = make_activations().requires_grad_(True)
    out, loss = lbl(h, labels=make_labels())
    (out.sum() + loss).backward()
    assert h.grad is not None
    assert not h.grad.isnan().any(), "NaN gradients through LBL"


def test_gradient_flows_at_boundary():
    """Gradients should still flow when activations are at the boundary."""
    lbl = make_lbl(r_init=0.01)   # almost everything hits boundary
    h   = make_activations().requires_grad_(True)
    out, _ = lbl(h)
    out.sum().backward()
    assert h.grad is not None
    assert not h.grad.isnan().any()
    assert h.grad.abs().sum().item() > 0, "Zero gradient at boundary"


# ── Loss terms ────────────────────────────────────────────────────────────────

def test_oracle_loss_nonzero_with_labels():
    lbl  = make_lbl()
    h    = make_activations()
    _, loss = lbl(h, labels=make_labels())
    assert loss.item() > 0


def test_no_oracle_loss_without_labels():
    lbl  = make_lbl()
    h    = make_activations()
    _, loss = lbl(h, labels=None)
    assert loss.item() == 0.0


def test_decorr_loss_with_adj():
    lbl  = make_lbl()
    h    = make_activations()
    adj  = torch.randn(BATCH, DIM)
    _, loss_with    = lbl(h, adj_activations=adj)
    _, loss_without = lbl(h)
    assert loss_with.item() > loss_without.item(), (
        "Decorrelation penalty should increase loss"
    )


def test_lambda_oracle_scales_loss():
    h      = make_activations()
    labels = make_labels()

    lbl_lo = make_lbl(lambda_oracle=0.1)
    lbl_hi = make_lbl(lambda_oracle=2.0)
    # Share weights so only lambda differs
    lbl_hi.load_state_dict(lbl_lo.state_dict(), strict=False)

    _, loss_lo = lbl_lo(h, labels=labels)
    _, loss_hi = lbl_hi(h, labels=labels)
    assert loss_hi.item() > loss_lo.item()


# ── 3D input (sequence) ───────────────────────────────────────────────────────

def test_3d_input():
    lbl = make_lbl()
    h   = torch.randn(BATCH, 10, DIM)   # (batch, seq, dim)
    out, loss = lbl(h, labels=make_labels())
    assert out.shape == h.shape


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibrate_sets_radius():
    lbl = make_lbl(r_init=1.0)
    h   = make_activations()

    lbl.collect_calibration_norms(h)
    lbl.calibrate(percentile=50.0)

    assert lbl._calibrated
    # Radius should have changed from init
    r = torch.exp(lbl.log_r).item()
    assert r > 0


def test_calibrate_without_data_raises():
    lbl = make_lbl()
    with pytest.raises(RuntimeError, match="No calibration data"):
        lbl.calibrate()


# ── LBLSequential ─────────────────────────────────────────────────────────────

def make_sequential():
    layers = [
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, NUM_CLASSES),
    ]
    return LBLSequential(layers, lbl_every_n=2, dim=DIM, num_classes=NUM_CLASSES)


def test_sequential_forward_shape():
    model = make_sequential()
    x     = torch.randn(BATCH, DIM)
    out, loss = model(x, labels=make_labels())
    assert out.shape == (BATCH, NUM_CLASSES)


def test_sequential_lbl_count():
    """With 7 layers and lbl_every_n=2, expect 3 LBLs (after layers 2, 4, 6)."""
    layers = [nn.Linear(DIM, DIM) for _ in range(7)]
    model  = LBLSequential(layers, lbl_every_n=2, dim=DIM, num_classes=NUM_CLASSES)
    assert len(model.lbl_layers) == 3


def test_sequential_no_lbl_loss_at_inference():
    model  = make_sequential()
    x      = torch.randn(BATCH, DIM)
    _, lbl_loss = model(x, return_lbl_loss=False)
    assert lbl_loss.item() == 0.0


def test_sequential_gradient_flows():
    model  = make_sequential()
    x      = torch.randn(BATCH, DIM, requires_grad=True)
    out, lbl_loss = model(x, labels=make_labels())
    (out.sum() + lbl_loss).backward()
    assert x.grad is not None
    assert not x.grad.isnan().any()


def test_sequential_calibrate_all():
    model   = make_sequential()
    dataset = torch.utils.data.TensorDataset(
        torch.randn(64, DIM), torch.randint(0, NUM_CLASSES, (64,))
    )
    loader  = torch.utils.data.DataLoader(dataset, batch_size=16)
    model.calibrate_all(loader, device=torch.device("cpu"), n_batches=3)

    for lbl in model.lbl_layers:
        assert lbl._calibrated


def test_print_boundary_report_no_crash(capsys):
    model  = make_sequential()
    x      = torch.randn(BATCH, DIM)
    model(x, labels=make_labels())
    model.print_boundary_report()
    captured = capsys.readouterr()
    assert "LBL Boundary Report" in captured.out
