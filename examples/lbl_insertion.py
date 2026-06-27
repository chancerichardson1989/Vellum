"""
examples/lbl_insertion.py
==========================
Minimal demonstration of LBL calibration, training, and boundary monitoring.

Run:
    pip install vellum-ml torchvision
    python examples/lbl_insertion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError("Run: pip install torchvision")

from vellum import AWD, LBLSequential

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────

transform    = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds     = datasets.MNIST("data", train=True,  download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

# ── Model ─────────────────────────────────────────────────────────────────────
# 6 layers, LBL inserted every 2 layers → 2 LBLs at cascade junctions

DIM = 256

layers = [
    nn.Flatten(),
    nn.Linear(784, DIM), nn.ReLU(),
    nn.Linear(DIM,  DIM), nn.ReLU(),
    nn.Linear(DIM,  DIM), nn.ReLU(),
    nn.Linear(DIM,  DIM), nn.ReLU(),
    nn.Linear(DIM, 10),
]

model = LBLSequential(
    layers,
    lbl_every_n=2,
    dim=DIM,
    num_classes=10,
    lbl_kwargs={
        "lambda_oracle": 0.5,
        "lambda_decorr": 0.05,
    },
).to(DEVICE)

# ── Calibrate radii from warm-up pass ─────────────────────────────────────────

model.calibrate_all(train_loader, DEVICE, n_batches=8)

# ── Train ─────────────────────────────────────────────────────────────────────

optimizer = AWD(model.parameters(), lr=1e-3)

print("\nTraining (AWD + LBL)...\n")
print(f"{'Ep':>4} {'Task Loss':>10} {'LBL Loss':>10} {'Total':>10} {'Mean ρ':>8}")
print(f"{'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

for epoch in range(1, 11):
    model.train()
    total_task = total_lbl = total_steps = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits, lbl_loss = model(x, labels=y, return_lbl_loss=True)
        task_loss        = F.cross_entropy(logits, y)
        loss             = task_loss + lbl_loss

        loss.backward()
        optimizer.step()

        total_task  += task_loss.item()
        total_lbl   += lbl_loss.item()
        total_steps += 1

    avg_task = total_task / total_steps
    avg_lbl  = total_lbl  / total_steps
    stats    = optimizer.get_adhesion_stats()
    rho      = stats.get("mean_rho", 0.0)

    print(f"{epoch:>4} {avg_task:>10.5f} {avg_lbl:>10.5f} "
          f"{avg_task + avg_lbl:>10.5f} {rho:>8.4f}")

# ── Reports ───────────────────────────────────────────────────────────────────

optimizer.print_adhesion_report()
model.print_boundary_report()
