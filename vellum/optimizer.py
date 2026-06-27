"""
Adhesion-Weighted Descent (AWD) Optimizer
==========================================
Part of the Vellum framework.

The adhesion signal ρ = |m_hat| / (v_hat + ε) is bounded in [0, 1]
and measures sign coherence of the gradient over recent steps.
Adam computes this signal and immediately divides it away.
AWD uses it to weight the update.
"""

import torch
from torch.optim import Optimizer
import math


class AWD(Optimizer):
    """
    Adhesion-Weighted Descent.

    Drop-in replacement for Adam with adhesion-weighted updates and
    optional shadow field freezing for non-adhering parameters.

    Args:
        params:                  iterable of parameters or param groups
        lr (float):              learning rate α (default: 1e-3)
        betas (tuple):           (β₁, β₂) EMA coefficients (default: 0.9, 0.999)
        eps (float):             numerical stability term (default: 1e-8)
        eta (float):             exploratory step scale when ρ ≈ 0 (default: 0.01)
        shadow_threshold (float):ρ below which parameter is non-adhering (default: 0.2)
        shadow_patience (int):   steps below threshold before shadow freeze (default: 50)
        weight_decay (float):    L2 penalty (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        eta: float = 0.01,
        shadow_threshold: float = 0.2,
        shadow_patience: int = 50,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"Invalid eta: {eta}")
        if not 0.0 <= shadow_threshold < 1.0:
            raise ValueError(f"Invalid shadow_threshold: {shadow_threshold}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eta=eta,
            shadow_threshold=shadow_threshold,
            shadow_patience=shadow_patience,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            eta = group["eta"]
            shadow_threshold = group["shadow_threshold"]
            shadow_patience = group["shadow_patience"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AWD does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["rho"] = torch.ones_like(p, memory_format=torch.preserve_format)
                    state["shadow_counter"] = torch.zeros_like(
                        p, dtype=torch.int32, memory_format=torch.preserve_format
                    )
                    state["shadow_mask"] = torch.zeros_like(
                        p, dtype=torch.bool, memory_format=torch.preserve_format
                    )
                    state["shadow_values"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                m = state["m"]
                v = state["v"]
                shadow_mask = state["shadow_mask"]
                shadow_counter = state["shadow_counter"]
                state["step"] += 1
                t = state["step"]

                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)

                # Zero gradients for shadow parameters
                active_grad = grad.masked_fill(shadow_mask, 0.0)

                # Signed EMA (Adam m)
                m.mul_(beta1).add_(active_grad, alpha=1.0 - beta1)
                # Magnitude EMA — |grad|, not grad² (key difference from Adam)
                v.mul_(beta2).add_(active_grad.abs(), alpha=1.0 - beta2)

                # Bias correction
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2

                # Adhesion signal: ρ = |m_hat| / (v_hat + ε)
                # Hidden inside Adam all along — used here instead of discarded
                rho = (m_hat.abs() / (v_hat + eps)).clamp(0.0, 1.0)
                state["rho"] = rho

                # AWD update:
                # ρ-weighted Adam term + (1-ρ)-weighted exploratory term
                adam_term = m_hat / (v_hat + eps)
                update = rho * adam_term + (1.0 - rho) * eta * active_grad
                p.add_(update, alpha=-lr)

                # Shadow field: freeze parameters that fail to adhere
                below_threshold = (rho < shadow_threshold) & (~shadow_mask)
                shadow_counter.add_(below_threshold.to(torch.int32))
                shadow_counter.masked_fill_(~below_threshold & ~shadow_mask, 0)

                newly_shadowed = (shadow_counter >= shadow_patience) & (~shadow_mask)
                if newly_shadowed.any():
                    state["shadow_values"][newly_shadowed] = p[newly_shadowed].clone()
                    shadow_mask.logical_or_(newly_shadowed)
                    shadow_counter.masked_fill_(newly_shadowed, 0)

                if shadow_mask.any():
                    p[shadow_mask] = state["shadow_values"][shadow_mask]

        return loss

    def get_adhesion_stats(self) -> dict:
        """
        Returns adhesion statistics across all parameters.

        Keys:
            mean_rho:        average sign coherence (rises as training progresses)
            min_rho:         worst-adhering parameter group
            shadow_fraction: fraction of params frozen in shadow field
            total_params:    total parameter count
            shadow_params:   number of frozen parameters
        """
        all_rho = []
        total_params = 0
        shadow_params = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state or len(self.state[p]) == 0:
                    continue
                state = self.state[p]
                rho = state["rho"].float()
                mask = state["shadow_mask"]
                all_rho.append(rho[~mask].flatten())
                total_params += p.numel()
                shadow_params += mask.sum().item()

        if not all_rho:
            return {}

        all_rho_cat = torch.cat(all_rho)
        return {
            "mean_rho": all_rho_cat.mean().item(),
            "min_rho": all_rho_cat.min().item(),
            "shadow_fraction": shadow_params / max(total_params, 1),
            "total_params": total_params,
            "shadow_params": int(shadow_params),
        }

    def get_shadow_mask(self) -> list:
        """Returns (param, shadow_mask_tensor) pairs for all parameters."""
        return [
            (p, self.state[p]["shadow_mask"])
            for group in self.param_groups
            for p in group["params"]
            if p in self.state and "shadow_mask" in self.state[p]
        ]

    def release_shadow(self):
        """Thaws all shadow field parameters. Call after curriculum shifts."""
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state and "shadow_mask" in self.state[p]:
                    self.state[p]["shadow_mask"].fill_(False)
                    self.state[p]["shadow_counter"].fill_(0)

    def print_adhesion_report(self):
        """Prints current adhesion statistics."""
        stats = self.get_adhesion_stats()
        if not stats:
            print("AWD: No state yet — run at least one step first.")
            return

        bar_len = 30
        rho_bar    = int(stats["mean_rho"] * bar_len)
        shadow_bar = int(stats["shadow_fraction"] * bar_len)

        print("\n┌─────────────────────────────────────────┐")
        print("│         AWD Adhesion Report             │")
        print("├─────────────────────────────────────────┤")
        print(f"│  Mean ρ (sign coherence):               │")
        print(f"│  [{'█' * rho_bar}{'░' * (bar_len - rho_bar)}] {stats['mean_rho']:.4f}  │")
        print(f"│  Min ρ:  {stats['min_rho']:.4f}                          │")
        print(f"│                                         │")
        print(f"│  Shadow field:                          │")
        print(f"│  [{'█' * shadow_bar}{'░' * (bar_len - shadow_bar)}] {stats['shadow_fraction']:.2%}  │")
        print(f"│  {stats['shadow_params']:,} / {stats['total_params']:,} params frozen         │")
        print("└─────────────────────────────────────────┘\n")
