# Mathematical Background

## The Self-Consistent Model

### State Space

We work on the period-2 torus ``\mathbb{T} = \mathbb{R}/(2\mathbb{Z})``, represented as ``[-1,1)`` with wrap-around.

### The Quadratic Map

```math
T_a(x) = a - (a+1)x^2
```

We interpret all phases in ``\mathbb{T}`` (mod 2) when combining with noise and shift.

### Periodized Gaussian Noise

Let ``\bar{\rho}_\sigma`` be the periodized Gaussian kernel on ``\mathbb{T}``. Its Fourier coefficients are:

```math
\widehat{\bar{\rho}_\sigma}(k) = \exp(-\tfrac{\pi^2}{2} \sigma^2 k^2)
```

### Noisy Transfer Operator with Shift

For each shift ``c \in \mathbb{R}``:

```math
(P_c f)(x) = \int_{\mathbb{T}} \bar{\rho}_\sigma(x - T_a(y) - c) f(y) \, dy
```

### Self-Consistency

We define:
- Observable: ``m(f) = \langle \Phi, f \rangle = \int_{\mathbb{T}} \Phi(x) f(x) \, dx``
- Coupling: ``c(f) = \delta \cdot G(m(f))``

The self-consistent operator is:

```math
\mathcal{T}(f) = P_{c(f)} f
```

**Goal:** Find fixed points ``f^*`` with ``\mathcal{T}(f^*) = f^*``, ``\int f^* = 1``, ``f^* \geq 0``.

## Fourier Discretization

### Truncated Fourier Series

Densities are represented by truncated Fourier series:

```math
f(x) \approx \sum_{k=-N}^{N} \hat{f}_k e^{2\pi i k x}
```

### Operator in Fourier Coordinates

Define:
```math
(B_a \hat{f})_k := \int_{\mathbb{T}} e^{-2\pi i k T_a(y)} f(y) \, dy
```

Then:
```math
\widehat{(P_c f)}_k = \underbrace{e^{-2\pi^2 \sigma^2 k^2}}_{\text{Gaussian}} \cdot \underbrace{e^{-2\pi i k c}}_{\text{shift phase}} \cdot (B_a \hat{f})_k
```

### Building ``B_a`` via FFT

Matrix entries:
```math
(B_a)_{k\ell} = \int_{\mathbb{T}} e^{-2\pi i k T_a(y)} e^{2\pi i \ell y} \, dy
```

FFT trick: For each ``k``, sample ``g_k(y) = e^{-2\pi i k T_a(y)}`` on a fine grid and compute FFT.

## Rigorous Verification (Newton-Kantorovich)

The package implements rigorous computer-assisted proofs based on the Newton-Kantorovich theorem.

### Key Constants

For Gaussian smoothing with parameter ``\sigma`` and analytic strip width ``\tau``:

```math
S_{\tau,\sigma} := \sup_{k \in \mathbb{Z}} \exp(2\pi\tau|k| - 2\pi^2\sigma^2 k^2)
```

```math
S^{(1)}_{0,\sigma} \leq \frac{1}{\sigma\sqrt{e}}, \quad S^{(2)}_{0,\sigma} \leq \frac{2}{\sigma^2 e}
```

### Kantorovich Condition

Given a numerical candidate ``\tilde{f}`` with residual bound ``\Delta``, inverse Jacobian bound ``M``, and Lipschitz constant ``\gamma``:

If ``h := M^2 \gamma \Delta \leq \frac{1}{2}``, then there exists a unique true fixed point ``f^*`` with:

```math
\|f^* - \tilde{f}\|_2 \leq r := \frac{1 - \sqrt{1 - 2h}}{M\gamma}
```
