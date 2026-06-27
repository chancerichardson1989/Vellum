"""
examples/mnist_benchmark.py
============================
Trains a 4-layer MLP on MNIST with Adam, AdamW, and AWD from identical
initialization. Plots loss curves and mean ρ (AWD adhesion signal) over
training.

Run:
    pip install vellum-ml matplotlib torchvision
    python examples/mnist_benchmark.py

Results are saved to: results/mnist_benchmark.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import copy

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

from vellum import AWD


# ── Config ────────────────────────────────────────────────────────────────────

EPOCHS      = 15
BATCH_SIZE  = 256
LR          = 1e-3
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"


# ── Model ─────────────────────────────────────────────────────────────────────

def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0

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

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Base model — all experiments start from identical weights
    base_model = make_mlp().to(DEVICE)
    base_state = copy.deepcopy(base_model.state_dict())

    experiments = {
        "Adam":  optim.Adam,
        "AdamW": optim.AdamW,
        "AWD":   AWD,
    }

    history = {name: {"train_loss": [], "test_loss": [], "test_acc": [], "rho": []}
               for name in experiments}

    for name, opt_class in experiments.items():
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")

        model = make_mlp().to(DEVICE)
        model.load_state_dict(copy.deepcopy(base_state))
        optimizer = opt_class(model.parameters(), lr=LR)

        print(f"{'Ep':>4} {'Train Loss':>11} {'Test Loss':>10} {'Test Acc':>9} {'Mean ρ':>8}")
        print(f"{'─'*4} {'─'*11} {'─'*10} {'─'*9} {'─'*8}")

        for epoch in range(1, EPOCHS + 1):
            train_loss, _   = train_epoch(model, train_loader, optimizer)
            test_loss, acc  = eval_epoch(model, test_loader)

            rho_str = "  —     "
            rho_val = None
            if isinstance(optimizer, AWD):
                stats   = optimizer.get_adhesion_stats()
                rho_val = stats.get("mean_rho", 0.0)
                rho_str = f"{rho_val:.4f}"

            history[name]["train_loss"].append(train_loss)
            history[name]["test_loss"].append(test_loss)
            history[name]["test_acc"].append(acc)
            history[name]["rho"].append(rho_val)

            print(f"{epoch:>4} {train_loss:>11.5f} {test_loss:>10.5f} {acc:>9.2%} {rho_str:>8}")

        if isinstance(optimizer, AWD):
            optimizer.print_adhesion_report()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print("  Final Test Accuracy")
    print(f"{'═'*50}")
    for name in experiments:
        acc = history[name]["test_acc"][-1]
        print(f"  {name:<8}  {acc:.2%}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not HAS_MATPLOTLIB:
        print("\nInstall matplotlib to save the plot: pip install matplotlib")
        return

    colors  = {"Adam": "#6B8EAD", "AdamW": "#8EAD6B", "AWD": "#C9963A"}
    epochs  = list(range(1, EPOCHS + 1))

    fig = plt.figure(figsize=(14, 9), facecolor="#F2E8D5")
    fig.suptitle(
        "Vellum: AWD vs Adam vs AdamW — MNIST MLP Benchmark",
        fontsize=14, fontweight="bold", color="#1A1209", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_train = fig.add_subplot(gs[0, 0])
    ax_test  = fig.add_subplot(gs[0, 1])
    ax_acc   = fig.add_subplot(gs[1, 0])
    ax_rho   = fig.add_subplot(gs[1, 1])

    def style_ax(ax, title, ylabel, xlabel="Epoch"):
        ax.set_facecolor("#EDE3CE")
        ax.set_title(title, fontsize=11, color="#1A1209", pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color="#6B4F2A")
        ax.set_ylabel(ylabel, fontsize=9, color="#6B4F2A")
        ax.tick_params(colors="#6B4F2A", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#C9963A")
            spine.set_linewidth(0.8)
        ax.grid(True, color="#D4C5A9", linewidth=0.5, linestyle="--", alpha=0.7)

    style_ax(ax_train, "Training Loss",   "Cross-Entropy Loss")
    style_ax(ax_test,  "Test Loss",        "Cross-Entropy Loss")
    style_ax(ax_acc,   "Test Accuracy",    "Accuracy")
    style_ax(ax_rho,   "AWD Mean ρ (Sign Coherence)", "ρ")

    for name in experiments:
        c = colors[name]
        lw = 2.0 if name == "AWD" else 1.4
        ax_train.plot(epochs, history[name]["train_loss"], label=name, color=c, linewidth=lw)
        ax_test.plot( epochs, history[name]["test_loss"],  label=name, color=c, linewidth=lw)
        ax_acc.plot(  epochs, history[name]["test_acc"],   label=name, color=c, linewidth=lw)

    rho_vals = [v for v in history["AWD"]["rho"] if v is not None]
    if rho_vals:
        ax_rho.plot(epochs[:len(rho_vals)], rho_vals,
                    color=colors["AWD"], linewidth=2.2, label="AWD ρ")
        ax_rho.axhline(0.5, color="#8B2014", linewidth=0.8, linestyle=":", alpha=0.7,
                       label="ρ = 0.5 (threshold reference)")
        ax_rho.set_ylim(0, 1)
        ax_rho.legend(fontsize=8, facecolor="#F2E8D5", edgecolor="#C9963A")
    else:
        ax_rho.text(0.5, 0.5, "No ρ data", ha="center", va="center",
                    transform=ax_rho.transAxes, color="#6B4F2A")

    for ax in [ax_train, ax_test, ax_acc]:
        ax.legend(fontsize=8, facecolor="#F2E8D5", edgecolor="#C9963A")

    out_path = os.path.join(RESULTS_DIR, "mnist_benchmark.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
