"""
examples/cifar10_benchmark.py
==============================
Trains a ConvNet on CIFAR-10 with Adam, AdamW, and AWD from identical
initialization. CIFAR-10 is a meaningful ML benchmark — 10 classes,
60,000 color images, requires real feature learning unlike MNIST.

This is the minimum bar for the ML community to take optimizer results
seriously. MNIST is too easy. CIFAR-10 has enough complexity that
optimizer behavior actually matters.

Architecture: 5-layer ConvNet with BatchNorm and residual-style skip
              (~370k parameters — large enough to show optimizer differences,
               small enough to run in ~15 minutes on CPU)

What to watch:
    - Does AWD converge faster than Adam in early epochs?
    - Does AWD's final test accuracy differ from Adam?
    - Does mean ρ rise as training progresses? (MNIST: it didn't)
    - Does shadow field fraction stay low? (healthy = <5%)

Run:
    pip install vellum-ml matplotlib torchvision
    python examples/cifar10_benchmark.py

Results saved to:
    results/cifar10_benchmark.png
    results/cifar10_results.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import copy
import time

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError("Run: pip install torchvision")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not found — will print results only, no plot saved.")
    print("Install with: pip install matplotlib\n")

from vellum import AWD


# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS      = 30
BATCH_SIZE  = 128
LR          = 1e-3
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"

print(f"\nDevice: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Estimated time: ~5 minutes")
else:
    print("Running on CPU — estimated time: 15-25 minutes")
    print("Tip: reduce EPOCHS to 15 for a faster result\n")


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN with optional residual projection."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Residual projection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CIFARNet(nn.Module):
    """
    5-block ConvNet for CIFAR-10.
    ~370k parameters. Real feature learning required.

    Architecture:
        stem:    3 → 32 channels, 3×3 conv
        block1:  32 → 32,  stride 1
        block2:  32 → 64,  stride 2  (16×16)
        block3:  64 → 128, stride 2  (8×8)
        block4:  128 → 128, stride 2 (4×4)
        head:    GAP → 128 → 10
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block1 = ConvBlock(32,  32,  stride=1)
        self.block2 = ConvBlock(32,  64,  stride=2)
        self.block3 = ConvBlock(64,  128, stride=2)
        self.block4 = ConvBlock(128, 128, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Data ──────────────────────────────────────────────────────────────────────

def make_loaders():
    """
    CIFAR-10 with standard augmentation for training.
    Augmentation is fixed — same for all three optimizers.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616),
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_ds = datasets.CIFAR10("data", train=True,  download=True,
                                 transform=train_transform)
    test_ds  = datasets.CIFAR10("data", train=False, download=True,
                                 transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss = correct = n = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        # Gradient clipping — same for all optimizers, removes confound
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = correct = n = 0

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)

    return total_loss / n, correct / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = make_loaders()

    # Base model — all experiments start from identical weights
    base_model = CIFARNet().to(DEVICE)
    base_state = copy.deepcopy(base_model.state_dict())
    print(f"Model parameters: {count_params(base_model):,}")

    experiments = {
        "Adam":  optim.Adam,
        "AdamW": optim.AdamW,
        "AWD":   AWD,
    }

    history = {
        name: {
            "train_loss": [], "train_acc": [],
            "test_loss":  [], "test_acc":  [],
            "rho": [], "shadow_frac": [],
            "epoch_time": [],
        }
        for name in experiments
    }

    results_lines = []

    for name, opt_class in experiments.items():
        print(f"\n{'═'*54}")
        print(f"  {name}")
        print(f"{'═'*54}")

        model = CIFARNet().to(DEVICE)
        model.load_state_dict(copy.deepcopy(base_state))
        optimizer = opt_class(model.parameters(), lr=LR,
                              weight_decay=1e-4 if name == "AdamW" else 0.0)

        # Cosine annealing — same schedule for all optimizers
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )

        header = (f"{'Ep':>4} {'Train L':>8} {'Train A':>8} "
                  f"{'Test L':>8} {'Test A':>8} {'ρ':>7} {'Shadow':>7} {'Time':>6}")
        print(header)
        print("─" * len(header))

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader,
                                                optimizer, scheduler)
            test_loss,  test_acc  = eval_epoch(model, test_loader)
            epoch_time = time.time() - t0

            rho_val    = None
            shadow_val = 0.0
            rho_str    = "  —    "
            shadow_str = "  —    "

            if isinstance(optimizer, AWD):
                stats      = optimizer.get_adhesion_stats()
                rho_val    = stats.get("mean_rho", 0.0)
                shadow_val = stats.get("shadow_fraction", 0.0)
                rho_str    = f"{rho_val:.4f}"
                shadow_str = f"{shadow_val:.2%}"

            history[name]["train_loss"].append(train_loss)
            history[name]["train_acc"].append(train_acc)
            history[name]["test_loss"].append(test_loss)
            history[name]["test_acc"].append(test_acc)
            history[name]["rho"].append(rho_val)
            history[name]["shadow_frac"].append(shadow_val)
            history[name]["epoch_time"].append(epoch_time)

            line = (f"{epoch:>4} {train_loss:>8.4f} {train_acc:>8.2%} "
                    f"{test_loss:>8.4f} {test_acc:>8.2%} "
                    f"{rho_str:>7} {shadow_str:>7} {epoch_time:>5.1f}s")
            print(line)
            results_lines.append(f"{name} | {line}")

        if isinstance(optimizer, AWD):
            optimizer.print_adhesion_report()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*54}")
    print("  CIFAR-10 Final Results (epoch 30)")
    print(f"{'═'*54}")
    print(f"  {'Optimizer':<10} {'Test Acc':>10} {'Test Loss':>10} {'Train Acc':>10}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    summary_lines = []
    for name in experiments:
        ta  = history[name]["test_acc"][-1]
        tl  = history[name]["test_loss"][-1]
        tra = history[name]["train_acc"][-1]
        line = f"  {name:<10} {ta:>10.2%} {tl:>10.4f} {tra:>10.2%}"
        print(line)
        summary_lines.append(line)

    # Best epoch for each
    print(f"\n  Best test accuracy per optimizer:")
    for name in experiments:
        accs    = history[name]["test_acc"]
        best    = max(accs)
        best_ep = accs.index(best) + 1
        print(f"  {name:<10} {best:.2%} at epoch {best_ep}")

    # AWD-specific
    rho_vals = [v for v in history["AWD"]["rho"] if v is not None]
    if rho_vals:
        print(f"\n  AWD ρ trajectory:")
        print(f"    Epoch  1: {rho_vals[0]:.4f}")
        print(f"    Epoch 15: {rho_vals[14]:.4f}" if len(rho_vals) > 14 else "")
        print(f"    Epoch {EPOCHS}: {rho_vals[-1]:.4f}")
        trend = "rising ✓" if rho_vals[-1] > rho_vals[0] else "flat/falling — see theory.md"
        print(f"    Trend: {trend}")

    # Save text results
    txt_path = os.path.join(RESULTS_DIR, "cifar10_results.txt")
    with open(txt_path, "w") as f:
        f.write("CIFAR-10 Benchmark — Vellum AWD vs Adam vs AdamW\n")
        f.write("=" * 54 + "\n\n")
        f.write("\n".join(results_lines))
        f.write("\n\nSummary:\n")
        f.write("\n".join(summary_lines))
    print(f"\nFull results saved to: {txt_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not HAS_MATPLOTLIB:
        print("Install matplotlib to save plot: pip install matplotlib")
        return

    colors = {"Adam": "#6B8EAD", "AdamW": "#8EAD6B", "AWD": "#C9963A"}
    epochs = list(range(1, EPOCHS + 1))

    fig = plt.figure(figsize=(16, 10), facecolor="#F2E8D5")
    fig.suptitle(
        "Vellum: AWD vs Adam vs AdamW — CIFAR-10 ConvNet Benchmark",
        fontsize=14, fontweight="bold", color="#1A1209", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    ax_tloss = fig.add_subplot(gs[0, 0])
    ax_vloss = fig.add_subplot(gs[0, 1])
    ax_vacc  = fig.add_subplot(gs[0, 2])
    ax_tacc  = fig.add_subplot(gs[1, 0])
    ax_rho   = fig.add_subplot(gs[1, 1])
    ax_shdw  = fig.add_subplot(gs[1, 2])

    def style_ax(ax, title, ylabel, xlabel="Epoch"):
        ax.set_facecolor("#EDE3CE")
        ax.set_title(title, fontsize=10, color="#1A1209", pad=7)
        ax.set_xlabel(xlabel, fontsize=8, color="#6B4F2A")
        ax.set_ylabel(ylabel, fontsize=8, color="#6B4F2A")
        ax.tick_params(colors="#6B4F2A", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#C9963A")
            spine.set_linewidth(0.7)
        ax.grid(True, color="#D4C5A9", linewidth=0.4, linestyle="--", alpha=0.7)

    style_ax(ax_tloss, "Training Loss",    "Cross-Entropy")
    style_ax(ax_vloss, "Test Loss",        "Cross-Entropy")
    style_ax(ax_vacc,  "Test Accuracy",    "Accuracy")
    style_ax(ax_tacc,  "Training Accuracy","Accuracy")
    style_ax(ax_rho,   "AWD Mean ρ",       "Sign Coherence ρ")
    style_ax(ax_shdw,  "AWD Shadow Field", "Fraction Frozen")

    for name in experiments:
        c  = colors[name]
        lw = 2.0 if name == "AWD" else 1.4
        ax_tloss.plot(epochs, history[name]["train_loss"], label=name, color=c, lw=lw)
        ax_vloss.plot(epochs, history[name]["test_loss"],  label=name, color=c, lw=lw)
        ax_vacc.plot( epochs, history[name]["test_acc"],   label=name, color=c, lw=lw)
        ax_tacc.plot( epochs, history[name]["train_acc"],  label=name, color=c, lw=lw)

    # ρ plot
    rho_vals = [v for v in history["AWD"]["rho"] if v is not None]
    if rho_vals:
        ax_rho.plot(epochs[:len(rho_vals)], rho_vals,
                    color=colors["AWD"], lw=2.2, label="AWD ρ")
        ax_rho.axhline(0.5, color="#8B2014", lw=0.8, linestyle=":",
                       alpha=0.7, label="ρ=0.5 reference")
        ax_rho.set_ylim(0, 1)

        # Annotate first and last
        ax_rho.annotate(f"ρ={rho_vals[0]:.3f}",
                        xy=(1, rho_vals[0]),
                        xytext=(3, rho_vals[0] + 0.05),
                        fontsize=7, color="#6B4F2A")
        ax_rho.annotate(f"ρ={rho_vals[-1]:.3f}",
                        xy=(EPOCHS, rho_vals[-1]),
                        xytext=(EPOCHS - 8, rho_vals[-1] + 0.05),
                        fontsize=7, color="#6B4F2A")
        ax_rho.legend(fontsize=7, facecolor="#F2E8D5", edgecolor="#C9963A")

    # Shadow field plot
    shadow_vals = history["AWD"]["shadow_frac"]
    if any(v > 0 for v in shadow_vals):
        ax_shdw.plot(epochs, shadow_vals,
                     color=colors["AWD"], lw=2.0, label="Shadow fraction")
        ax_shdw.axhline(0.05, color="#8B2014", lw=0.8, linestyle=":",
                        alpha=0.7, label="5% warning threshold")
        ax_shdw.set_ylim(0, max(max(shadow_vals) * 1.2, 0.1))
        ax_shdw.legend(fontsize=7, facecolor="#F2E8D5", edgecolor="#C9963A")
    else:
        ax_shdw.text(0.5, 0.5, "No params frozen\n(shadow patience not reached)",
                     ha="center", va="center", transform=ax_shdw.transAxes,
                     fontsize=8, color="#6B4F2A")

    for ax in [ax_tloss, ax_vloss, ax_vacc, ax_tacc]:
        ax.legend(fontsize=7, facecolor="#F2E8D5", edgecolor="#C9963A")

    # Annotation box — key result
    best_awd  = max(history["AWD"]["test_acc"])
    best_adam = max(history["Adam"]["test_acc"])
    delta     = (best_awd - best_adam) * 100
    sign      = "+" if delta >= 0 else ""
    note = (f"Best test acc:\n"
            f"AWD   {best_awd:.2%}\n"
            f"Adam  {best_adam:.2%}\n"
            f"Δ = {sign}{delta:.2f}pp")
    fig.text(0.98, 0.02, note, ha="right", va="bottom",
             fontsize=8, color="#1A1209",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#EDE3CE", edgecolor="#C9963A", alpha=0.9))

    out_path = os.path.join(RESULTS_DIR, "cifar10_benchmark.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
