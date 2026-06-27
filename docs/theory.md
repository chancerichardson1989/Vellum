# Vellum Architecture — Theory

Full mathematical derivation of AWD and LBL.

---

## 1. Adhesion-Weighted Descent (AWD)

### 1.1 The Hidden Signal in Adam

Adam maintains two exponential moving averages for each parameter θ^(i):

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          [signed EMA]
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²          [squared EMA]
```

With bias correction:

```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
```

Adam's update: `θ_{t+1} = θ_t - α · m̂_t / (√v̂_t + ε)`

The division by `√v̂_t` normalizes the step to unit magnitude. This discards the ratio `|m̂_t| / √v̂_t` — the relative coherence of the signed and magnitude accumulators.

### 1.2 The Adhesion Signal ρ

Define the adhesion signal using the magnitude EMA (not squared):

```
v_t^mag = β₂ · v_{t-1}^mag + (1 - β₂) · |g_t|
```

Then:

```
ρ_t^(i) = |m̂_t^(i)| / (v̂_t^mag^(i) + ε)
```

**Claim:** ρ ∈ [0, 1] measures sign coherence of the gradient over recent history.

**Proof sketch:**
- When gradients consistently point the same direction: the signed EMA m accumulates toward the magnitude of the gradient; the magnitude EMA v̂^mag accumulates similarly. Ratio → 1.
- When gradients alternate sign each step: m cancels toward zero (positive and negative terms annihilate under exponential weighting); v̂^mag continues to accumulate. Ratio → 0.
- The ratio is bounded above by 1 because |m̂| ≤ v̂^mag by Jensen's inequality applied to the absolute value function.

### 1.3 The AWD Update Rule

```
θ_{t+1}^(i) = θ_t^(i)
             - α · ρ_t^(i) · m̂_t^(i) / (v̂_t^(i) + ε)    [adhesion-scaled Adam]
             - α · (1 - ρ_t^(i)) · η · g_t^(i)             [exploratory term]
```

where η ≪ 1 is the exploratory step scale.

**Behavior at extremes:**
- ρ = 1: reduces exactly to Adam. Full step in the direction of accumulated gradient.
- ρ = 0: m̂ ≈ 0 (sign cancellation), so the Adam term vanishes naturally. Parameter takes a small exploratory step in the current gradient direction. The network does not freeze — it searches locally for surface tooth.

**Note:** At ρ = 0, the Adam term `m̂ / (v̂ + ε)` has already collapsed near zero due to sign cancellation in m. The ρ multiplier is formally redundant but stabilizes the transition and makes the behavior explicit.

### 1.4 Shadow Field

Define the shadow counter:

```
d_t^(i) = d_{t-1}^(i) + 𝟙[ρ_t^(i) < ρ_threshold]  if not in shadow field
         = 0                                           if ρ recovered
```

When `d_t^(i) ≥ patience`:
- Record `θ^(i)` as shadow value
- Set shadow mask = True
- Parameter receives no further gradient updates
- Shadow value is restored each step (acts as inhibitory boundary)

The shadow field records the topography of failed grip — regions where the optimizer could not find surface tooth. Active parameters cannot migrate into shadow space without cost.

---

## 2. Liminal Boundary Layer (LBL)

### 2.1 The Constraint Set

The LBL constrains activations h ∈ ℝ^d to a learnable ellipsoid:

```
C_t = { z ∈ ℝ^d : ‖ Σ_t^{-1/2} (z - μ_t) ‖ ≤ r }
```

where:
- μ_t ∈ ℝ^d is the learned center
- Σ_t^{-1/2} = diag(exp(log_scale)) is a diagonal approximation of the inverse square root of the shape matrix (full Σ would cost O(d²))
- r > 0 is the learned radius (initialized via calibration)

### 2.2 Smooth Projection

The projection Π_C(h) is defined as:

```
Π_C^soft(h) = μ + (h - μ) / max(1, s)

where s = ‖ Σ^{-1/2}(h - μ) ‖ / r
```

**Differentiability:** This function is smooth everywhere. The `max(1, s)` operation is the only potential non-smoothness, but since s is a norm (smooth away from zero) divided by r, and the function is `x / max(1, x/r)` applied to each scaled coordinate, it is C¹ everywhere.

**Jacobian:**

For a point inside C (s ≤ 1): the projection is the identity. Jacobian = I.

For a point outside C (s > 1):

```
∂Π/∂h = (1/s) · ( I - (h-μ)(h-μ)ᵀ / ‖h-μ‖² )  (in scaled coordinates)
```

This is the projection onto the tangent plane of the ellipsoid at the contact point. The gradient is deflected along the surface rather than zeroed.

**Non-expansiveness:** For any h₁, h₂:
```
‖Π(h₁) - Π(h₂)‖ ≤ ‖h₁ - h₂‖
```
This follows from the projection onto a convex set being a contraction. The LBL cannot amplify signal — it can only contain it.

### 2.3 LBL Loss

The LBL is trained with two objectives:

**Oracle loss** (maximize MI with final output):

```
L_oracle = CrossEntropy(oracle_head(Π(h)), y)
```

The oracle head is a linear layer trained against ground-truth labels. This pulls the ellipsoid shape toward a geometry where projected activations are linearly separable by class — i.e., predictive of the final output. This is a stable approximation of the mutual information maximization `I(Π(h); y)` that avoids MINE's training instability.

**Decorrelation penalty** (minimize MI with adjacent layers):

```
L_decorr = ‖ Cov(Π(h), h_adj) / (‖Π(h)‖ · ‖h_adj‖) ‖_F²
```

The Frobenius norm of the normalized cross-correlation matrix. This penalizes the LBL output for being predictable from adjacent layer activations — enforcing that the boundary is drawn from output knowledge, not local context.

**Total LBL loss:**
```
L_LBL = λ_oracle · L_oracle + λ_decorr · L_decorr
```

Added to the task loss during training: `L_total = L_task + L_LBL`.

### 2.4 Interaction with AWD

The LBL and AWD do not communicate directly. Their interaction emerges from the gradient signal:

1. A parameter in a dense layer upstream of an LBL takes steps that push activations toward the ellipsoid boundary.
2. As activations press against C, the Jacobian of the projection deflects gradients tangentially.
3. The deflected gradients flowing back through the parameter oscillate in sign — the direction keeps changing as the activation probes the curved boundary surface.
4. AWD's sign coherence ρ drops for this parameter.
5. AWD reduces its step size and increases local exploration.
6. The parameter slows and searches along the boundary rather than pushing through it.

The LBL does not tell AWD there is a boundary. AWD feels it through the gradient signal.

### 2.5 Calibration

The ellipsoid radius r must be initialized to match the typical scale of activations at each LBL position. The recommended procedure:

1. Run a warm-up forward pass on representative data (no gradient computation).
2. Collect the scaled activation norms `‖Σ^{-1/2}(h - μ)‖` at each LBL.
3. Set `r = percentile(norms, 95)`.

**Why 95th percentile:**
- Setting r below the median means most activations are immediately constrained, which severs gradient flow effectively — the boundary is too tight for the optimizer to feel.
- Setting r above the 99th percentile means activations rarely contact the boundary, and the outline has no effect.
- The 95th percentile puts the boundary in contact with the tails of the activation distribution — active constraint without total domination.

### 2.6 Placement

LBLs should be placed at cascade junctions — points where a local failure would propagate through many subsequent layers. Practical rules:

- Every √L layers for a network of depth L
- Immediately before residual addition points
- Never consecutively (two adjacent LBLs collapse the signal manifold)

The intuition: a cascade junction is where the monk draws the outline before applying gold. The outline defines the shape the illumination will take. Placing LBLs at these points gives the constraint layer maximum influence over the final geometry with minimum interference with local computation.

---

## 3. Open Questions

The following are unresolved and represent the primary empirical questions for validation:

1. **Does ρ rise during training on real networks?** If sign coherence is a meaningful signal, we would expect mean ρ to increase as parameters find their equilibrium positions. If ρ stays flat or decreases, the theory may not reflect actual optimization dynamics.

2. **Does AWD outperform Adam on non-synthetic benchmarks?** The synthetic smoke test is encouraging but not informative. MNIST is a weak signal. CIFAR-10, ImageNet, or a language modeling task would be meaningful.

3. **Does shadow field freezing help or hurt?** The hypothesis is that frozen shadow parameters act as inhibitory boundaries that prevent the active network from revisiting failed regions. The alternative hypothesis is that frozen parameters simply reduce effective model capacity and hurt performance.

4. **Is the diagonal ellipsoid approximation sufficient?** A full Σ matrix would allow the ellipsoid to rotate — to align with the principal components of the activation distribution. The diagonal approximation can only scale along fixed axes. For activation spaces with strong off-diagonal correlations, this may be insufficient.

5. **Does LBL boundary contact actually produce the ρ drop in AWD that the theory predicts?** This is the key interaction claim and should be directly measurable: instrument ρ for parameters in layers adjacent to LBLs versus layers far from LBLs, and compare.
