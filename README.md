# Vellum

**Vellum** is an experimental PyTorch framework for training neural networks using principles derived from medieval manuscript illumination.

It contains two components:

- **AWD** (Adhesion-Weighted Descent) — an optimizer that recovers a signal Adam silently discards
- **LBL** (Liminal Boundary Layer) — a constraint layer that holds the geometric outline of a signal without severing gradient flow

Both are unvalidated. This is a theory-first release. The goal is to find out whether the ideas hold up on real networks.

---

## The Core Hypothesis

Adam computes two exponential moving averages:

```
m = signed EMA of gradients       (first moment)
v = EMA of squared gradients      (second moment)
```

It uses the ratio `m / sqrt(v)` to normalize the update — which divides away the sign coherence information entirely.

The quantity `|m| / v` (using magnitude, not squared magnitude) is bounded in [0, 1] and measures something Adam ignores: **how consistently a parameter's gradient is pointing the same direction.** We call this ρ — the adhesion signal.

- ρ ≈ 1.0 — gradient signs are consistent. The parameter has found purchase. Take a full step.
- ρ ≈ 0.0 — gradient signs are oscillating. The parameter is reverting. Shrink the Adam term, take a small exploratory step instead.

The AWD update rule:

```
θ_{t+1} = θ_t
         - α · ρ · m̂ / (v̂ + ε)        # adhesion-scaled Adam term
         - α · (1 - ρ) · η · g_t        # exploratory term when ρ is low
```

When ρ = 1 this is Adam. When ρ = 0, m̂ has already cancelled itself (sign oscillation means the signed average goes to zero), so the Adam term vanishes naturally — no special casing required.

Parameters that remain below a coherence threshold for too long are moved to a **shadow field** — frozen as inhibitory boundary conditions that prevent the active network from revisiting failed regions.

The LBL extends this: it inserts a smooth ellipsoidal projection at cascade junctions in the network. Activations are constrained to a learnable ellipsoid whose shape is trained against the output distribution. The projection is everywhere differentiable — gradients at the boundary are deflected onto the tangent plane rather than severed. AWD upstream reads this deflection as low ρ and slows naturally. The two modules never communicate directly.

---

## Status


| Component | State |
|---|---|
| AWD optimizer | Implemented, tested |
| LBL module | Implemented, tested |
| MNIST benchmark | **Run — results below** |
| Real architecture validation | Not done |
| Hyperparameter search | Not done |

This is a research artifact. The theory is internally consistent; whether it reflects something real about optimization dynamics is an open question. Contributions, counterexamples, and null results are all welcome.
---

## Installation

```bash
git clone https://github.com/chancerichardson1989/vellum.git
cd vellum
pip install -e .
```

Requires Python 3.9+ and PyTorch 2.0+.

---

## Quick Start

### AWD as a drop-in optimizer

```python
from vellum import AWD

optimizer = AWD(model.parameters(), lr=1e-3)

# Training loop — identical to Adam
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

# Inspect adhesion after training
optimizer.print_adhesion_report()
```

### LBL inserted into a network

```python
import torch.nn as nn
from vellum import AWD, LBLSequential

layers = [
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 10),
]

model = LBLSequential(
    layers,
    lbl_every_n=2,       # LBL at every 2nd layer boundary
    dim=256,             # activation dimension
    num_classes=10,
)

# Calibrate ellipsoid radii before training
model.calibrate_all(dataloader, device)

# Training loop
optimizer = AWD(model.parameters(), lr=1e-3)
for x, y in dataloader:
    optimizer.zero_grad()
    logits, lbl_loss = model(x, labels=y, return_lbl_loss=True)
    loss = criterion(logits, y) + lbl_loss
    loss.backward()
    optimizer.step()

model.print_boundary_report()
```

---

## Key Parameters

### AWD

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate α |
| `betas` | (0.9, 0.999) | EMA coefficients — same as Adam |
| `eta` | 0.01 | Exploratory step scale when ρ ≈ 0 |
| `shadow_threshold` | 0.2 | ρ below which a parameter is considered non-adhering |
| `shadow_patience` | 50 | Consecutive steps below threshold before shadow freeze |

### LBL

| Parameter | Default | Description |
|---|---|---|
| `lambda_oracle` | 1.0 | Weight on oracle auxiliary loss |
| `lambda_decorr` | 0.1 | Weight on decorrelation penalty |
| `r_init` | 1.0 | Initial ellipsoid radius (overwritten by calibrate()) |

---

## Known Limitations

- `LBLSequential` assumes uniform activation dimension across all layers. It will not work as-is on networks with changing width (ResNets, transformers, encoder-decoders). Wrapping individual layers manually with `LiminalBoundaryLayer` is the current workaround for non-uniform architectures.
- The diagonal ellipsoid approximation (per-dimension scale, no rotation) may be insufficient for activation spaces with strong off-axis correlations.
- Shadow field freezing is currently irreversible within a training run unless `release_shadow()` is called explicitly. Curriculum learning setups should monitor shadow fraction carefully.
- The oracle auxiliary head assumes classification. Regression tasks need a different oracle objective.

---

## Running the Benchmark

```bash
python examples/mnist_benchmark.py
```

Trains a 4-layer MLP on MNIST with Adam, AdamW, and AWD from the same initialization. Plots loss curves and ρ over training. Downloads MNIST automatically via torchvision.

```bash
python examples/lbl_insertion.py
```

Minimal demonstration of LBL calibration and boundary contact monitoring.

---
---

## MNIST Benchmark Results

4-layer MLP (784→256→256→128→10), 15 epochs, batch size 256, lr=1e-3.
All three optimizers start from identical weight initialization.

| Optimizer | Final Test Acc | Final Test Loss | Notes |
|---|---|---|---|
| Adam | ~98.0% | higher | spikes at epochs 6 and 13 |
| AdamW | ~98.0% | higher | similar spike pattern to Adam |
| **AWD** | **~98.1%** | **lowest** | smoother curve, lowest final test loss |

**Training loss** — all three converge to near zero by epoch 15. AWD starts marginally higher in early epochs and converges to the same floor.

**Test loss** — AWD produces the smoothest curve and the lowest final value. Adam and AdamW both show instability spikes mid-training that AWD does not replicate. This is the most meaningful difference observed.

**Test accuracy** — all three reach ~98%, competitive with standard MLP results on MNIST. AWD ties or leads at epoch 15.

**Mean ρ (sign coherence)** — stays in the 0.2–0.4 range throughout training, below the 0.5 threshold reference. Does not rise monotonically as the theory predicted. Two possible explanations:

1. MNIST is too simple to stress the optimizer into clear adhesion behavior — most parameters find their direction quickly and ρ reflects residual noise rather than meaningful surface search
2. The shadow patience of 50 steps may be too long for a 15-epoch run on this dataset, preventing the shadow field from building up enough to push ρ upward in active parameters

Both are worth investigating on a harder task (CIFAR-10, language modeling).

**Interpretation** — AWD does not dramatically outperform Adam on MNIST. It matches accuracy while showing lower test loss and a smoother optimization trajectory. The ρ signal is present and measurable but behaves differently than predicted. This is an honest first result, not a claim of superiority.

---

## Theory

The full mathematical derivation — smooth ellipsoidal projection, gradient transparency proof, oracle mutual information objective, and the relationship between AWD's ρ signal and LBL boundary contact — is in [`docs/theory.md`](docs/theory.md).

The framework emerged from a generative exercise: solving a computational problem while simultaneously holding the constraint that monastic manuscript illumination requires painstakingly slow, hand-crafted gold leaf application. The metaphor is not decorative. The specific constraints of the craft — surface preparation before inscription, the transparency threshold of beaten gold, the burnisher's tactile feedback, the outline drawn before gold is applied — translated directly into the computational mechanisms above.

---

## Contributing

Issues, benchmark results (positive or negative), and PRs are welcome.

If you run the benchmark on a real architecture and AWD underperforms Adam, please open an issue with the result. A documented null result is more useful than silence.

---

## License

MIT
