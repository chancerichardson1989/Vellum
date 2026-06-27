"""
Liminal Boundary Layer (LBL)
============================
Part of the Vellum framework.

Constrains activations to a learnable ellipsoid without severing gradient
flow. The projection is smooth and everywhere differentiable — gradients at
the boundary are deflected onto the tangent plane, not zeroed. AWD upstream
reads this deflection as low sign coherence and slows naturally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LiminalBoundaryLayer(nn.Module):
    """
    Liminal Boundary Layer (LBL).

    Projects activations onto a learnable ellipsoid C whose shape is trained
    to be predictive of the final output while being decorrelated from
    adjacent layer activations.

    Args:
        dim (int):              activation dimensionality
        num_classes (int):      output classes for oracle auxiliary head
        r_init (float):         initial ellipsoid radius (default: 1.0)
                                overwritten by calibrate()
        lambda_decorr (float):  weight on decorrelation penalty (default: 0.1)
        lambda_oracle (float):  weight on oracle auxiliary loss (default: 1.0)
        eps (float):            numerical stability (default: 1e-6)
        learn_center (bool):    whether μ is a learned parameter (default: True)
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        r_init: float = 1.0,
        lambda_decorr: float = 0.1,
        lambda_oracle: float = 1.0,
        eps: float = 1e-6,
        learn_center: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.lambda_decorr = lambda_decorr
        self.lambda_oracle = lambda_oracle
        self.eps = eps

        # Ellipsoid center
        if learn_center:
            self.mu = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("mu", torch.zeros(dim))

        # Per-dimension scale factors — diagonal approximation of Σ^{-1/2}
        # Full matrix would cost O(d²); diagonal is sufficient for most tasks
        self.log_scale = nn.Parameter(torch.zeros(dim))

        # Ellipsoid radius — scalar, calibrated from activation norms
        self.log_r = nn.Parameter(torch.tensor(math.log(r_init)))

        # Oracle: lightweight linear head trained against output labels
        # Stable proxy for mutual information with the final output
        self.oracle_head = nn.Linear(dim, num_classes)

        # Calibration buffer
        self.register_buffer("_calib_norms", torch.zeros(0))
        self._calibrated = False

        # Diagnostics
        self.register_buffer("_last_boundary_fraction", torch.tensor(0.0))
        self.register_buffer("_last_mean_s", torch.tensor(0.0))

    def _smooth_project(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Smooth ellipsoidal projection onto C.

        Π(h) = μ + (h - μ) / max(1, s)
        where s = ‖ Σ^{-1/2} (h - μ) ‖ / r

        Jacobian:
            s ≤ 1 (inside C):   identity — gradient passes through unchanged
            s > 1 (boundary):   gradient projected onto tangent plane of ellipsoid

        Returns:
            projected:  (batch, dim)
            s:          (batch,) normalized distance — s > 1 means boundary contact
        """
        scale = torch.exp(self.log_scale)           # (dim,)
        r = torch.exp(self.log_r)                   # scalar

        centered = h - self.mu                      # (batch, dim)
        scaled   = centered * scale                 # (batch, dim)
        s = scaled.norm(dim=-1, keepdim=True) / (r + self.eps)  # (batch, 1)

        denom     = torch.clamp(s, min=1.0)
        projected = self.mu + centered / denom      # (batch, dim)

        return projected, s.squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        adj_activations: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:               (batch, dim) or (batch, seq, dim) activations
            adj_activations: adjacent layer activations for decorr penalty
            labels:          class labels (Long tensor) for oracle loss

        Returns:
            projected:  constrained activations, same shape as h
            lbl_loss:   scalar — add to task loss during training
        """
        original_shape = h.shape
        if h.dim() == 3:
            batch, seq, dim = h.shape
            h_flat   = h.reshape(batch * seq, dim)
            adj_flat = adj_activations.reshape(batch * seq, -1) if adj_activations is not None else None
        else:
            h_flat   = h
            adj_flat = adj_activations

        projected_flat, s = self._smooth_project(h_flat)

        with torch.no_grad():
            self._last_boundary_fraction = (s > 1.0).float().mean()
            self._last_mean_s = s.mean()

        projected = projected_flat.reshape(original_shape)
        lbl_loss  = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        # Oracle loss — ellipsoid shape pulled toward output-predictive geometry
        if labels is not None:
            oracle_input  = projected.mean(dim=1) if h.dim() == 3 else projected_flat
            oracle_logits = self.oracle_head(oracle_input)
            lbl_loss      = lbl_loss + self.lambda_oracle * F.cross_entropy(oracle_logits, labels)

        # Decorrelation penalty — outline ignores its neighbors
        if adj_flat is not None:
            lbl_loss = lbl_loss + self.lambda_decorr * self._decorrelation_penalty(
                projected_flat, adj_flat
            )

        return projected, lbl_loss

    def _decorrelation_penalty(
        self, projected: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Squared Frobenius norm of the cross-correlation matrix.
        Penalizes correlation between LBL output and adjacent activations.
        Falls back to random projection for large dimensions.
        """
        p_norm = F.normalize(projected,      dim=-1, eps=self.eps)
        a_norm = F.normalize(adj.float(),    dim=-1, eps=self.eps)

        if p_norm.shape[-1] > 512 or a_norm.shape[-1] > 512:
            p_norm, a_norm = self._random_project(p_norm, a_norm)

        cross_corr = p_norm.T @ a_norm / p_norm.shape[0]
        return cross_corr.pow(2).sum()

    def _random_project(
        self, p: torch.Tensor, a: torch.Tensor, target_dim: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rp = F.normalize(torch.randn(p.shape[-1], target_dim, device=p.device, dtype=p.dtype), dim=0)
        ra = F.normalize(torch.randn(a.shape[-1], target_dim, device=a.device, dtype=a.dtype), dim=0)
        return p @ rp, a @ ra

    @torch.no_grad()
    def collect_calibration_norms(self, h: torch.Tensor):
        """
        Collect activation norms during a warm-up pass.
        Call calibrate() after collecting from enough batches.
        """
        scale  = torch.exp(self.log_scale)
        h_flat = h.reshape(-1, self.dim) if h.dim() == 3 else h
        norms  = ((h_flat - self.mu) * scale).norm(dim=-1).cpu()
        self._calib_norms = torch.cat([self._calib_norms, norms])

    @torch.no_grad()
    def calibrate(self, percentile: float = 95.0):
        """
        Set ellipsoid radius to the given percentile of observed activation norms.

        95th percentile is the recommended starting point:
            too tight → projection severs gradient flow
            too loose → outline never contacts activations

        Args:
            percentile: which percentile to use as radius (default: 95.0)
        """
        if self._calib_norms.numel() == 0:
            raise RuntimeError(
                "No calibration data. Run collect_calibration_norms() first, "
                "or use LBLSequential.calibrate_all()."
            )
        k     = max(1, min(int(math.ceil(percentile / 100.0 * self._calib_norms.numel())),
                           self._calib_norms.numel()))
        r_val = self._calib_norms.kthvalue(k).values.item()
        self.log_r.data.fill_(math.log(max(r_val, self.eps)))
        self._calib_norms  = torch.zeros(0)
        self._calibrated   = True
        print(f"  LBL (dim={self.dim}) calibrated: r = {r_val:.4f}")

    def get_boundary_stats(self) -> dict:
        """
        Boundary contact statistics from the last forward pass.

        boundary_fraction:
            0.00–0.02  → ellipsoid too large, outline inactive
            0.05–0.30  → healthy operating range
            0.60+      → ellipsoid too tight, recalibrate or increase r_init

        mean_s:
            < 1.0  → most activations inside C
            > 1.0  → most activations being projected
        """
        return {
            "boundary_fraction": self._last_boundary_fraction.item(),
            "mean_s":            self._last_mean_s.item(),
            "r":                 torch.exp(self.log_r).item(),
            "calibrated":        self._calibrated,
            "dim":               self.dim,
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, "
            f"r={torch.exp(self.log_r).item():.3f}, "
            f"λ_oracle={self.lambda_oracle}, λ_decorr={self.lambda_decorr}, "
            f"calibrated={self._calibrated}"
        )


class LBLSequential(nn.Module):
    """
    Wraps a flat list of layers and inserts LBLs at every lbl_every_n boundary.

    Limitation: assumes uniform activation dimension across all layers.
    For networks with varying width (ResNets, transformers), insert
    LiminalBoundaryLayer manually at cascade junctions.

    Args:
        layers (list):      nn.Module layers in forward order
        lbl_every_n (int):  insert LBL after every N layers
        dim (int):          activation dimension (must be uniform)
        num_classes (int):  output classes for oracle head
        lbl_kwargs (dict):  passed to LiminalBoundaryLayer
    """

    def __init__(
        self,
        layers: list,
        lbl_every_n: int,
        dim: int,
        num_classes: int,
        lbl_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        lbl_kwargs = lbl_kwargs or {}

        self._layer_list = nn.ModuleList()
        self.lbl_layers  = nn.ModuleList()
        self._schedule   = []   # list of ('layer', idx) | ('lbl', idx)

        lbl_count = 0
        for i, layer in enumerate(layers):
            self._layer_list.append(layer)
            self._schedule.append(("layer", len(self._layer_list) - 1))

            if (i + 1) % lbl_every_n == 0 and i < len(layers) - 1:
                lbl = LiminalBoundaryLayer(dim, num_classes, **lbl_kwargs)
                self.lbl_layers.append(lbl)
                self._schedule.append(("lbl", len(self.lbl_layers) - 1))
                lbl_count += 1

        print(f"LBLSequential: {len(layers)} layers, {lbl_count} LBLs inserted")

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_lbl_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               input tensor
            labels:          ground truth labels (training only)
            return_lbl_loss: set False at inference to skip loss computation

        Returns:
            (output, total_lbl_loss)
        """
        total_lbl_loss   = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        prev_activation  = None
        h = x

        for kind, idx in self._schedule:
            if kind == "layer":
                prev_activation = h
                h = self._layer_list[idx](h)
            else:
                lbl = self.lbl_layers[idx]
                if return_lbl_loss:
                    h, lbl_loss = lbl(h, adj_activations=prev_activation, labels=labels)
                    total_lbl_loss = total_lbl_loss + lbl_loss
                else:
                    h, _ = lbl(h)

        return h, total_lbl_loss

    def calibrate_all(
        self,
        dataloader,
        device,
        n_batches: int = 10,
        percentile: float = 95.0,
    ):
        """
        Warm-up pass to calibrate all LBL radii before training.

        Args:
            dataloader: yields (x, y) — only x is used
            device:     torch device
            n_batches:  number of batches to collect norms from
            percentile: passed to LiminalBoundaryLayer.calibrate()
        """
        self.eval()
        hooks = []

        for lbl in self.lbl_layers:
            def make_hook(layer):
                def hook(module, input, output):
                    layer.collect_calibration_norms(input[0])
                return hook
            hooks.append(lbl.register_forward_hook(make_hook(lbl)))

        print("Calibrating LBL radii...")
        with torch.no_grad():
            for i, (x, *_) in enumerate(dataloader):
                if i >= n_batches:
                    break
                self(x.to(device), return_lbl_loss=False)

        for h in hooks:
            h.remove()

        for lbl in self.lbl_layers:
            lbl.calibrate(percentile=percentile)

        self.train()

    def print_boundary_report(self):
        """Prints boundary contact stats for all LBLs."""
        print("\n┌────────────────────────────────────────────────────────┐")
        print("│              LBL Boundary Report                       │")
        print("├────┬────────────┬──────────┬────────┬─────────────────┤")
        print("│ #  │ Bound Frac │  Mean s  │   r    │  Status         │")
        print("├────┼────────────┼──────────┼────────┼─────────────────┤")
        for i, lbl in enumerate(self.lbl_layers):
            stats = lbl.get_boundary_stats()
            bf    = stats["boundary_fraction"]
            status = "✓  healthy  " if 0.02 <= bf <= 0.6 else (
                     "⚠  too loose" if bf < 0.02 else "⚠  too tight")
            print(
                f"│ {i:<2d} │ {bf:>10.2%} │ {stats['mean_s']:>8.3f} │ "
                f"{stats['r']:>6.3f} │ {status}      │"
            )
        print("└────┴────────────┴──────────┴────────┴─────────────────┘\n")
