# SelfConsistentLogisticNoise.jl (fork of PlateauExperiment.jl) — Implementation Document

This document is intended to be pasted into Claude.ai / ChatGPT Codex to guide implementation.

## 0) What we are building
A Julia package (Git repo, **unregistered**) that computes fixed points of a **self-consistent noisy transfer operator** associated to the logistic family.

We will base this on a **fork of github.com/orkolorko/PlateauExperiment.jl** and reuse its existing FFT/Fourier/operator utilities (most of the FFT work is already there). The new package should feel similar in spirit to github.com/orkolorko/PlateauExperiment.jl: a clean “experiment runner + results” repo, not meant to be registered.

Core experiment:
- Map: **logistic** \(T_a(x)=a\,x(1-x)\) (start with \(a=3.83\))
- Noise: **periodized Gaussian** on the torus
- Self-consistency: shift \(c(f)=\delta\,G(m(f))\) with \(m(f)=\langle\Phi,f\rangle\)
- Objective: compute **fixed points** \(f^\*\) of the self-consistent operator and study how they vary with \(\delta\) and \(a\).

---

## 1) Mathematical model

### 1.1 State space
Use the torus \(\mathbb T=\mathbb R/\mathbb Z\), represented as \([0,1)\) with wrap-around.

### 1.2 Map
\[
T_a(x)=a\,x(1-x)
\]
Interpret all phases in \(\mathbb T\) (mod 1) when combining with noise and shift.

### 1.3 Periodized Gaussian noise
Let \(\bar\rho_\sigma\) be the periodized Gaussian kernel on \(\mathbb T\). Its Fourier coefficients are
\[
\widehat{\bar\rho_\sigma}(k)=\exp(-2\pi^2\sigma^2 k^2).
\]

### 1.4 Noisy transfer operator with shift
Define for each shift \(c\in\mathbb R\):
\[
(P_c f)(x)=\int_{\mathbb T}\bar\rho_\sigma\big(x - T_a(y) - c\big)\,f(y)\,dy.
\]

### 1.5 Self-consistency
Pick:
- observable \(\Phi\in L^2(\mathbb T)\)
- coupling nonlinearity \(G:\mathbb R\to\mathbb R\) (at least \(C^1\))
- coupling strength \(\delta\in\mathbb R\)

Define:
\[
m(f)=\langle\Phi,f\rangle = \int_{\mathbb T}\Phi(x)\,f(x)\,dx,\qquad
c(f)=\delta\,G(m(f)).
\]

Self-consistent operator:
\[
\mathcal T(f)=P_{c(f)}f.
\]

Goal: find fixed points \(f^\*\) with \(\mathcal T(f^\*)=f^\*\), \(\int f^\*=1\), \(f^\*\ge0\) (numerically).

---

## 2) Discretization (Fourier truncation, MVP)
Represent densities by truncated Fourier series:
\[
f(x)\approx \sum_{k=-N}^N \hat f_k\,e^{2\pi i k x}.
\]
Store coefficients as a length \(L=2N+1\) vector.

### 2.1 Fourier indexing convention
Mode \(k\in\{-N,\dots,N\}\) maps to index `idx = k + N + 1` (1-based Julia).

Provide:
- `modes(N) = -N:N`
- `idx(k,N) = k + N + 1`
- `mode(i,N) = i - (N+1)`

### 2.2 Operator in Fourier
Let
\[
(B_a \hat f)_k := \int_{\mathbb T} e^{-2\pi i k\,T_a(y)}\,f(y)\,dy.
\]
Then:
\[
\widehat{(P_c f)}_k
=
\underbrace{e^{-2\pi^2\sigma^2 k^2}}_{\text{Gaussian}}
\underbrace{e^{-2\pi i k c}}_{\text{shift phase}}
\cdot
(B_a\hat f)_k.
\]

So compute:
1) `tmp = B * fhat`
2) multiply each mode by `rhohat(k)*phase(k,c)`

### 2.3 Building the matrix \(B_a\) via FFT (reuse Plateau code)
Matrix entries:
\[
(B_a)_{k\ell} = \int_{\mathbb T} e^{-2\pi i k T_a(y)}\,e^{2\pi i \ell y}\,dy.
\]

FFT trick:
- For each fixed \(k\), define \(g_k(y)=e^{-2\pi i k T_a(y)}\).
- Sample on grid \(y_j=j/M\), \(j=0,\dots,M-1\).
- Compute Fourier coefficients of \(g_k\) by FFT; these coefficients are exactly \((B_a)_{k\ell}\) (up to normalization + frequency-index mapping).

Normalization convention to enforce:
\[
\hat g(\ell)=\frac1M \sum_{j=0}^{M-1} g(y_j)\,e^{-2\pi i \ell y_j}.
\]
In Julia/FFTW, this is typically `fft(g) / M` if `fft` uses the \(\exp(-2\pi i jk/M)\) convention.

Use oversampling: e.g. `M = os * (2N+1)` with `os=8 or 16`, preferably rounded to a fast FFT length.

Caching:
- Cache `B_a` to disk keyed by `(a,N,M)` to avoid rebuilding.

---

## 3) Default choices for \(\Phi\) and \(G\)

### 3.1 Recommended default observable \(\Phi\)
Use a trig polynomial so \(m(f)\) is read directly from Fourier coefficients.

Default:
- \(\Phi(x)=\cos(2\pi x)\) ⇒ \(m(f)=\Re(\hat f_1)\)

Fourier coefficients:
- \(\hat\Phi_{1}=\hat\Phi_{-1}=\tfrac12\), others 0.

### 3.2 Recommended default \(G\)
Provide at least two built-ins:
- `G(m)=m` (linear, simplest)
- `G(m)=tanh(β*m)` (saturating; include parameter `β`)

Implement as callable structs for easy serialization/logging.

---

## 4) Fixed-point solver

### 4.1 Picard iteration with damping (MVP)
Given initial `fhat`:
1. `m = observable(fhat)`
2. `c = δ * G(m)`
3. `fhat_new = apply_Pc(fhat, c)` (Fourier operator)
4. enforce normalization: set `fhat_new[idx(0)] = 1 + 0im`
5. damping: `fhat ← (1-α)*fhat + α*fhat_new`

Stop when:
- `norm(fhat_new - fhat) / norm(fhat) < tol` or residual below tol.

Typical parameters:
- `tol = 1e-12` (complex coefficient norm)
- `maxit = 5000`
- `α ∈ [0.05, 1]` (start with 0.2)

### 4.2 Warm-start continuation (important for sweeps)
When sweeping δ or a:
- use previous converged solution as initial condition.

### 4.3 Multi-start (to detect multiple fixed points)
For each (a,δ), run from several initial guesses:
- uniform density
- low-mode bump
- random low modes (small amplitude)

Cluster converged solutions by `norm(fhat1 - fhat2)`.

### 4.4 Optional acceleration
Optional but very helpful:
- Anderson acceleration (small memory 5–20)
- or simple “Aitken-like” heuristics

---

## 5) Optional diagnostics: Jacobian and singular values
If you want stability diagnostics, implement Jacobian \(D\mathcal T_N(f)\) at a fixed point in Fourier coordinates.

For the self-consistent shift, the Jacobian has “base + rank-one” form:
\[
D\mathcal T_N(f) = A(c) + b \otimes a^*,
\]
where:
- \(A(c)=\mathrm{diag}(\rhohat(k)\,e^{-2\pi i k c}) \, B_a\)
- \(a\) corresponds to \(c'(f)\), i.e. \(a_k = \delta\,G'(m(f))\,\hat\Phi_k\) (up to conjugation convention)
- \(b\) is the derivative of the phase w.r.t. \(c\):
  \[
  b_k = (-2\pi i k)\,\rhohat(k)\,e^{-2\pi i k c}\,(B_a \hat f)_k.
  \]

Then:
- compute `svdvals(DT)` for moderate N
- record `σmin`, `σmax` or `||DT||2` for stability charts.

This is optional; do not block MVP.

---

## 6) How to structure the forked repo

### 6.1 Repo approach
Fork `PlateauExperiments.jl` and add a new module inside it OR rename the top-level module.

Recommended minimal friction:
- keep Plateau’s module + utilities
- add new code under `src/self_consistent_logistic/`
- export a new user-facing module `SelfConsistentLogisticNoise`

### 6.2 Suggested layout

SelfConsistentLogisticNoise/
Project.toml
src/
SelfConsistentLogisticNoise.jl
sc/
indexing.jl
observable.jl
coupling.jl
operator_build.jl
operator_apply.jl
solver_picard.jl
sweeps.jl
io.jl
diagnostics.jl # optional
experiments/
sweep_delta.jl
sweep_delta_a.jl
multistart_demo.jl
test/
runtests.jl


### 6.3 Dependencies (typical)
- FFTW
- LinearAlgebra, Statistics, Random
- JLD2 (or Serialization)
- DataFrames, CSV (for sweep logs)
- Makie or Plots (optional)
- RigorousInvariantMeasures.jl (phase 2 integration)

---

## 7) Public API specification

### 7.1 Core types
- `struct LogisticMap; a::Float64; end`
- `struct GaussianNoise; σ::Float64; end`
- `struct FourierDisc; N::Int; M::Int; end`  (M is FFT oversampling grid)
- `struct Observable; Phihat::Vector{ComplexF64}; end` (or function-based)
- `struct Coupling; δ::Float64; G; observable::Observable; end`
- `struct SCProblem; map; noise; disc; coupling; B::Matrix{ComplexF64}; end`

### 7.2 Core functions
- `build_B(map::LogisticMap, disc::FourierDisc; cache=true) -> Matrix`
- `apply_Pc(prob::SCProblem, fhat::Vector{ComplexF64}, c::Float64) -> Vector`
- `m(prob, fhat) -> Float64`
- `solve_fixed_point(prob; α=0.2, tol=1e-12, maxit=5000, init=:uniform, accel=:none) -> Result`
- `sweep_delta(...) -> DataFrame`
- `sweep_delta_a(...) -> DataFrame`
- `reconstruct_density(fhat, xgrid) -> Vector{Float64}`

### 7.3 Result schema
Store a `Result` with:
- params: a, δ, σ, N, M, G name, Φ name
- outputs: m, c, residual, iters
- `fhat` saved to JLD2
- optionally `f(x)` samples saved for plotting

---

## 8) Experiments to include

### 8.1 Sweep δ at fixed a
- `a=3.83`, choose `σ` (e.g. 0.02), `N=256`, `M=2048`
- δ grid, warm-start continuation
- output plot m(δ), density snapshots

### 8.2 Sweep a around 3.83 for a few δ
- `a ∈ [3.75, 3.90]` step 0.005 (or whatever)
- δ ∈ {0, 0.2, 0.5}
- compare fixed points / m-values

### 8.3 Multi-start region scan
- pick δ range, run multiple initial seeds, detect multi-solution behavior

---

## 9) Testing / acceptance criteria
Minimum tests:
1. **δ=0 sanity:** solver converges to a fixed point
2. **normalization:** \(\hat f_0 = 1\) always; numerical integral ≈ 1
3. **realness:** reconstructed density real (imag small)
4. **grid convergence:** m stabilizes as N increases

---

## 10) Integration with RigorousInvariantMeasures.jl (phase 2)
MVP should work without deep RIM integration; but structure code so later you can:
- wrap coefficient vectors into RIM basis types
- use RIM norms / rigorous bounds / validated Newton (if desired)

Do not block MVP on full rigor.

---

## 11) Implementation checklist (do these in order)
1. Implement Fourier indexing helpers + tests for coefficient mapping.
2. Implement `build_B` using Plateau’s FFT helpers; verify normalization with a known function.
3. Implement `apply_Pc`.
4. Implement observable \(m(f)\) from Fourier.
5. Implement Picard solver with damping + warm starts.
6. Add sweep scripts and saving (JLD2 + CSV/DataFrame).
7. Add plotting script(s).
8. Optional: Jacobian + singular values.

---

## 12) Notes on conventions
- Work on the torus so periodized Gaussian is diagonal in Fourier.
- Enforce conjugate symmetry if you want strictly real densities:
  - \( \hat f_{-k} = \overline{\hat f_k} \)
  - \( \hat f_0 \in \mathbb R \)
- For FFT frequency mapping from `fft` output indices to modes \(\ell\in[-M/2,M/2]\), reuse Plateau’s existing mapping utilities; otherwise implement:
  - mode `ℓ` corresponds to FFT index `ℓ>=0 ? (ℓ+1) : (M+ℓ+1)` (Julia 1-based)