# Implementation document — taper + smoothing computation of an annealed transfer operator (with rigorous bounds)

## 0. What we are computing

We want a certified finite-rank approximation of the **annealed transfer operator**
\[
(\mathcal P_T f)(x) \;=\;\int_{-1}^{1} \rho_{\sigma_{\rm rds}}\!\big(x - T(y)\big)\, f(y)\,dy,
\]
where

- \(T:[-1,1]\to \mathbb R\) is the deterministic map (e.g. quadratic),
- \(\rho_{\sigma_{\rm rds}}\) is the **physical** Gaussian noise kernel (width \(\sigma_{\rm rds}\)),
- and we will build a periodic/FFT-friendly surrogate \(\widetilde T\) from \(T\) using
  - a **taper** (boundary collar width \(\eta\)),
  - an optional **Gaussian smoothing of the map** (width \(\sigma_{\rm sm}\), purely numerical),
  - a Fourier truncation \(\Pi_N\).

**Important:** \(\sigma_{\rm sm}\) (numerical smoothing of the map) is independent from \(\sigma_{\rm rds}\) (noise of the annealed operator). You can take \(\sigma_{\rm sm}\) small.

We aim to compute a matrix \(P_N\) representing
\[
P_N \approx \Pi_N\,\mathcal P_{\widetilde T}\,\Pi_N
\]
and a rigorous bound on the operator error
\[
\|\mathcal P_T - \Pi_N \mathcal P_{\widetilde T} \Pi_N\|
\le E_{\rm map} + E_{\rm proj} + E_{\rm num},
\]
in a chosen norm (recommended: \(L^1\to L^1\), or mixed strong/weak if you need Newton–Kantorovich later).

---

## 1. Fourier conventions (period 2 on \([-1,1]\))

To use FFT, treat \([-1,1]\) as one period of a length-2 torus. Use basis
\[
e_k(x):=e^{i\pi k x},\qquad k\in\mathbb Z,
\]
so Fourier coefficients are
\[
\hat f(k)=\frac12 \int_{-1}^{1} f(x)e^{-i\pi k x}\,dx.
\]

Let \(\Pi_N\) denote the projection onto modes \(|k|\le N\).

Grid for FFT:
- choose \(M\) even, typically \(M\ge 4N\) and a power of two,
- points \(x_j=-1 + 2j/M,\; j=0,\dots,M-1\).

---

## 2. Why taper + smoothing (the problem we’re fixing)

If we naively periodize a non-periodic map \(T\) on \([-1,1]\), we get derivative jumps at the boundary. That kills Fourier decay and makes a large \(N\) expensive.

We therefore build a *smooth periodic extension* \(\widetilde T\) of \(T\) by:
1) **tapering** near endpoints (modify \(T\) only in collars of width \(\eta\)),
2) optionally **Gaussian smoothing** the tapered map (width \(\sigma_{\rm sm}\)),
3) optionally **Fourier-truncating** \(\widetilde T\) (replace by \(\Pi_N \widetilde T\)).

This ensures rapid decay of the Fourier coefficients of functions like \(e^{-i\pi k \widetilde T(x)}\), which is exactly what we need to assemble the matrix via FFT.

---

## 3. Constructing the tapered periodic map \(\;T_\eta\)

### 3.1 Smooth step (“option 3”)

Define a smooth step \(s:[0,1]\to[0,1]\) with:
- \(s(0)=0,\; s(1)=1\),
- derivatives vanish to desired order at endpoints (for \(C^m\) matching).

A common polynomial choice is the **quintic smoothstep**
\[
s(u) = 6u^5 - 15u^4 + 10u^3,
\]
which is \(C^2\) with \(s'(0)=s'(1)=s''(0)=s''(1)=0\).
(If you need higher regularity, use higher-degree smoothsteps or a \(C^\infty\) bump.)

Define the taper window \(w_\eta(x)\) on \([-1,1]\):
- \(w_\eta(x)=1\) on \([-1+\eta,\,1-\eta]\),
- in the collars, interpolate smoothly to enforce periodic matching.

A concrete “endpoint-flattening” construction (what you were describing):
- work with the derivative \(T'(x)\),
- enforce periodic compatibility by modifying \(T'\) near \(\pm1\) so that the modified derivative satisfies \(T_\eta'(-1)=T_\eta'(1)=0\) (or other matching constraints),
- then integrate to recover \(T_\eta\).

### 3.2 Practical derivative-based recipe (matches your workflow)

Given samples \(T'(x_j)\) on the FFT grid:

1. **Force the endpoint derivative to 0**:
   - set the first sample (corresponding to \(x=-1\)) to 0,
   - similarly enforce 0 at the sample nearest \(x=1\) (same point on the torus).

2. **Blend in a collar**:
   - pick an integer collar size \(K\) (so \(\eta \approx 2K/M\)),
   - replace the first/last \(K\) samples by a moving average (window length \(L\)) that transitions to 0 at the endpoints.
   - This yields a \(C^{L-1}\)-like effective smoothness in practice (heuristic, but works very well).

3. **Mean correction (when needed)**:
   - for a periodic derivative, \(\int_{-1}^{1} T_\eta'(x)\,dx = 0\) must hold.
   - if your modification breaks this, subtract the mean of the derivative samples so the discrete integral is zero.
   - (For an even quadratic \(T(x)=a-(a+1)x^2\), \(T'\) is odd, and if your taper is symmetric you typically do *not* need mean correction; but check numerically.)

4. **Recover the function**:
   - FFT the modified derivative samples,
   - set the zero mode of the derivative to 0,
   - divide mode-by-mode by \(i\pi k\) (for \(k\neq0\)) to integrate,
   - choose the constant mode to match \(T(0)\) (or match the interior average).

This yields a periodic \(T_\eta\) on \([-1,1]\).

---

## 4. Gaussian smoothing of the map (numerical only): \(T_{\eta,\sigma_{\rm sm}}\)

Define the period-2 Gaussian convolution (numerical smoothing operator)
\[
(G_{\sigma_{\rm sm}} f)(x) = (\bar\rho_{\sigma_{\rm sm}} * f)(x),
\]
where \(\bar\rho\) is the periodized Gaussian on the length-2 torus.

In Fourier:
\[
\widehat{G_{\sigma_{\rm sm}} f}(k) = \widehat{\bar\rho_{\sigma_{\rm sm}}}(k)\,\hat f(k),
\quad \widehat{\bar\rho_{\sigma_{\rm sm}}}(k) = \exp\!\left(-\frac{\pi^2\sigma_{\rm sm}^2}{2}\,k^2\right),
\]
because the wavenumber is \(\pi k\) on period 2.

So smoothing is **diagonal in Fourier**, i.e. “componentwise product after the FFT”.

Define
\[
T_{\eta,\sigma_{\rm sm}} := G_{\sigma_{\rm sm}}(T_\eta).
\]

---

## 5. Rigorous bound for the map approximation \(\|T - \widetilde T\|_\infty\)

We will use \(\widetilde T := \Pi_N T_{\eta,\sigma_{\rm sm}}\) (FFT-ready, band-limited).

Split:
\[
\|T - \widetilde T\|_\infty
\le
\underbrace{\|T-T_\eta\|_\infty}_{\text{taper}}
+
\underbrace{\|T_\eta - G_{\sigma_{\rm sm}}T_\eta\|_\infty}_{\text{smoothing}}
+
\underbrace{\|G_{\sigma_{\rm sm}}T_\eta - \Pi_N G_{\sigma_{\rm sm}}T_\eta\|_\infty}_{\text{Fourier trunc}}.
\]

### 5.1 Explicit constants for the quadratic \(T(x)=a-(a+1)x^2\)

- \(\|T'\|_\infty = 2(a+1)\) on \([-1,1]\).

**Taper error.**
If you only modify \(T\) inside collars of width \(\eta\) and keep it unchanged on \([-1+\eta,1-\eta]\), a safe bound is
\[
\boxed{\;\|T-T_\eta\|_\infty \le 2(a+1)\,\eta.\;}
\]
(Reason: you change values over a distance \(\eta\), with slope bounded by \(2(a+1)\).)

**Smoothing error (good when \(\sigma_{\rm sm}\) small).**
Use the second-order bound
\[
\|f - G_{\sigma}f\|_\infty \le \frac{\sigma^2}{2}\,\|f''\|_\infty,
\]
valid for Gaussian convolution (on the line; on the torus it remains valid with the same constant).

For tapered \(T_\eta\), \(\|T_\eta''\|_\infty\) grows like \(O((a+1)/\eta)\) because tapering introduces curvature in the collar. If your smoothstep satisfies \(\|w'_\eta\|_\infty \lesssim C_1/\eta\), a safe bound is
\[
\boxed{\;\|T_\eta''\|_\infty \le 2(a+1)\Bigl(1+\frac{C_1}{\eta}\Bigr)\;}
\]
for a constant \(C_1\) depending only on the chosen smoothstep (compute/upper-bound it once).

Hence
\[
\boxed{\;\|T_\eta - G_{\sigma_{\rm sm}}T_\eta\|_\infty \le (a+1)\sigma_{\rm sm}^2\Bigl(1+\frac{C_1}{\eta}\Bigr).\;}
\]

**Fourier truncation after smoothing (period 2).**
After smoothing, Fourier coefficients decay like \(\exp(-c\sigma_{\rm sm}^2 k^2)\). A convenient explicit \(L^\infty\) tail bound is
\[
\boxed{
\|G_{\sigma_{\rm sm}}T_\eta - \Pi_N G_{\sigma_{\rm sm}}T_\eta\|_\infty
\;\le\;
\frac{2\|T_\eta\|_\infty}{\pi^2\sigma_{\rm sm}^2\,N}\,
\exp\!\Big(-\frac{\pi^2\sigma_{\rm sm}^2}{2}\,N^2\Big),
}
\]
(using a standard Gaussian tail integral for \(\sum_{k>N} e^{-c k^2}\)).
Also \(\|T_\eta\|_\infty \le \|T\|_\infty + 2(a+1)\eta\).

Putting together:
\[
\boxed{
\|T-\widetilde T\|_\infty \le
2(a+1)\eta
+ (a+1)\sigma_{\rm sm}^2\Bigl(1+\frac{C_1}{\eta}\Bigr)
+ \frac{2\|T_\eta\|_\infty}{\pi^2\sigma_{\rm sm}^2\,N}\,
\exp\!\Big(-\frac{\pi^2\sigma_{\rm sm}^2}{2}\,N^2\Big).
}
\]

---

## 6. Rigorous bound for the operator perturbation \(\|\mathcal P_T - \mathcal P_{\widetilde T}\|\)

A key advantage of Gaussian noise: the operator depends *Lipschitzly* on the map in \(L^1\to L^1\).

Let \(u,v\in\mathbb R\). For the (non-periodized) Gaussian \(\rho_\sigma\),
\[
\int_{\mathbb R}\big|\rho_\sigma(x-u)-\rho_\sigma(x-v)\big|\,dx
\le |u-v| \int_{\mathbb R}|\rho_\sigma'(t)|\,dt
= |u-v|\;\frac{\sqrt{2/\pi}}{\sigma}.
\]
The same bound applies on the period-2 torus (periodization does not increase total variation).

Therefore:
\[
\boxed{
\|\mathcal P_T - \mathcal P_{\widetilde T}\|_{L^1\to L^1}
\;\le\;
\frac{\sqrt{2/\pi}}{\sigma_{\rm rds}}\;\|T-\widetilde T\|_\infty.
}
\]
This is your main “map-approximation-to-operator-approximation” certificate.

---

## 7. Building the finite-rank matrix \(P_N\) by FFT

We want a matrix representing \(\Pi_N \mathcal P_{\widetilde T}\Pi_N\) on the basis \(e_k(x)=e^{i\pi k x}\).

### 7.1 Formula (Fourier-side)

For any \(f\),
\[
(\mathcal P_{\widetilde T} f)(x)=\int_{-1}^{1}\bar\rho_{\sigma_{\rm rds}}\!\big(x-\widetilde T(y)\big) f(y)\,dy.
\]
Take Fourier coefficient \(k\):
\[
\widehat{\mathcal P_{\widetilde T} f}(k)
= \widehat{\bar\rho_{\sigma_{\rm rds}}}(k)\cdot
\frac12\int_{-1}^{1} f(y)\,e^{-i\pi k\widetilde T(y)}\,dy.
\]
Now expand \(f(y)=\sum_\ell \hat f(\ell)e^{i\pi \ell y}\), yielding
\[
\widehat{\mathcal P_{\widetilde T} f}(k)
= \widehat{\bar\rho_{\sigma_{\rm rds}}}(k)\sum_{\ell\in\mathbb Z}
\hat f(\ell)\;\underbrace{\left(\frac12\int_{-1}^{1} e^{-i\pi k\widetilde T(y)}\,e^{i\pi \ell y}\,dy\right)}_{=:A_{k\ell}}.
\]
So the matrix entries are
\[
P_{k\ell} = \widehat{\bar\rho_{\sigma_{\rm rds}}}(k)\;A_{k\ell},\qquad |k|,|\ell|\le N.
\]

### 7.2 How to compute \(A_{k\ell}\) efficiently

For each fixed \(k\), define the periodic function
\[
g_k(y):= e^{-i\pi k\widetilde T(y)}.
\]
Then \(A_{k\ell}\) is exactly the \(\ell\)-th Fourier coefficient of \(g_k\):
\[
A_{k\ell} = \widehat{g_k}(\ell).
\]

**FFT approach:**
- sample \(g_k(y)\) on the grid \(y_j\),
- FFT to get \(\widehat{g_k}(\ell)\) for \(|\ell|\le M/2\),
- take \(|\ell|\le N\).

Complexity: computing all \(g_k\) for \(|k|\le N\) is \(O(NM)\) evaluations + \(O(N M\log M)\) FFTs.

---

## 8. Rigorous discretization bounds (projection + quadrature + floating)

Your total certified operator error will usually be assembled as:

\[
\|\mathcal P_T - \Pi_N \mathcal P_{\widetilde T}\Pi_N\|
\le
\underbrace{\|\mathcal P_T-\mathcal P_{\widetilde T}\|}_{E_{\rm map}}
+
\underbrace{\|\mathcal P_{\widetilde T}-\Pi_N \mathcal P_{\widetilde T}\Pi_N\|}_{E_{\rm proj}}
+
\underbrace{E_{\rm num}}_{\text{FFT/quadrature/rounding}}.
\]

### 8.1 The map-induced term \(E_{\rm map}\)
Use Section 6:
\[
\boxed{
E_{\rm map} \le \frac{\sqrt{2/\pi}}{\sigma_{\rm rds}}\,\|T-\widetilde T\|_\infty,
}
\]
and use Section 5 to bound \(\|T-\widetilde T\|_\infty\) explicitly in terms of \(\eta,\sigma_{\rm sm},N\).

### 8.2 The projection term \(E_{\rm proj}\)
There are two common choices:

**(A) Work in \(L^1\to L^1\)** and use a tail bound on the kernel’s Fourier coefficients (from the *physical* noise):
\[
\widehat{\bar\rho_{\sigma_{\rm rds}}}(k)=\exp\!\Big(-\frac{\pi^2\sigma_{\rm rds}^2}{2}\,k^2\Big).
\]
This gives a very strong high-frequency decay in the *output variable*. In practice this often makes \(\Pi_N\) extremely accurate for moderate \(N\).

A usable bound:
\[
\|\mathcal P_{\widetilde T}-\Pi_N\mathcal P_{\widetilde T}\|_{L^1\to L^1}
\;\lesssim\;
\sum_{|k|>N}|\widehat{\bar\rho_{\sigma_{\rm rds}}}(k)|
\;\le\;
C\frac{1}{\sigma_{\rm rds}N}\exp\!\Big(-c\sigma_{\rm rds}^2 N^2\Big),
\]
with explicit constants from Gaussian tail integrals.

**(B) Work in a mixed strong/weak norm** (useful for Newton–Kantorovich later):
- strong space: analytic strip norm \(\mathcal A_\tau\),
- weak space: \(L^2\).

Then the “Galatolo-style” constants are explicit for Gaussian:
\[
S_{\tau,\sigma_{\rm rds}}=\sup_{k\in\mathbb Z}\exp\!\big(2\pi\tau|k|-\tfrac{\pi^2\sigma_{\rm rds}^2}{2}k^2\big),
\]
and the projection tail in \(L^2\) is bounded by
\[
\|(I-\Pi_N)u\|_2 \le e^{-\pi\tau N}\|u\|_{\mathcal A_\tau}
\quad(\text{same form as before, with period-2 scaling}).
\]
This is the right setting if you want to certify a fixed point via Newton–Kantorovich.

### 8.3 Numerical term \(E_{\rm num}\): how to make it rigorous in practice

You have three viable strategies:

1) **BigFloat + a posteriori ball inflation (recommended).**
   - compute \(P_N\) at high precision (e.g. 256 bits),
   - compute again at higher precision (e.g. 384 bits),
   - take the difference as a reliable upper bound for rounding + FFT error,
   - store each entry as a ball: midpoint = high-precision value rounded to Float64, radius = (difference bound) + safety margin.

2) **BallArithmetic throughout.**
   - implement your own FFT over balls (more engineering),
   - or avoid FFT by validated quadrature (slower but simplest rigorous method).

3) **Validated quadrature for coefficients (fallback, slow).**
   - compute each \(A_{k\ell}=\frac12\int_{-1}^1 e^{-i\pi k\widetilde T(y)}e^{i\pi\ell y}dy\)
     with an interval quadrature (Gauss–Legendre + enclosure).
   - Complexity \(O(N^2Q)\) (too slow for large \(N\)), but clean and fully rigorous.

**Engineering recommendation for a Julia package:**
Start with (1). It gives tight radii and keeps the FFT pipeline.

---

## 9. Julia package structure (what to implement)

Suggested module layout:

- `Taper.jl`
  - `smoothstep(u)` (quintic or higher order)
  - `taper_derivative_samples!(Tp_samples; K, L, enforce_zero_endpoints=true, mean_correction=true)`
  - `integrate_derivative_fft(Tp_samples) -> T_samples` (and optionally Fourier coeffs)

- `Smoothing.jl`
  - `gaussian_multiplier_period2(k, sigma)` returning `exp(-(π^2*sigma^2/2)*k^2)`
  - `smooth_samples_fft(T_samples, sigma_sm) -> Ts_samples`
  - `smooth_fourier_coeffs!(That, sigma_sm)` (diagonal multiplier)

- `MapApproxBounds.jl`
  - `bound_taper_error_quadratic(a, eta) -> 2(a+1)eta`
  - `bound_second_derivative_taper_quadratic(a, eta, C1) -> 2(a+1)(1+C1/eta)`
  - `bound_smoothing_error(a, eta, sigma_sm, C1)`
  - `bound_trunc_error(Teta_sup, sigma_sm, N)`
  - `bound_map_sup_error(...)`

- `OperatorAssembly.jl`
  - `assemble_PN_fft(Ttilde_samples; N, M, sigma_rds) -> Pmat (Complex)`
    - loop k = -N:N:
      - form `gk_samples = exp.(-im*pi*k .* Ttilde_samples)`
      - FFT → `gk_hat`
      - take ℓ = -N:N: set `A[k,ℓ]=gk_hat[ℓ]` (with indexing)
      - multiply row by `rho_hat(k,sigma_rds)`
  - `rho_hat_period2(k, sigma_rds)`

- `OperatorBounds.jl`
  - `bound_operator_map_sensitivity(sigma_rds, map_sup_error) -> sqrt(2/pi)/sigma_rds * map_sup_error`
  - plus optional projection bounds in chosen norm
  - plus numerical inflation policy (BigFloat comparisons)

- `RigorousLinearAlgebra.jl`
  - convert `Pmat` → balls
  - compute invariant density / eigenpair with verification (RigorousInvariantMeasures.jl or your own ball-based power method + enclosure)

---

## 10. What constants must be computed (and how)

To run the certified pipeline you must choose/compute:

**User-chosen:**
- \(N\) : Fourier cutoff for the operator.
- \(M\) : FFT grid size (e.g. \(M=4N\) or \(8N\)).
- \(\eta\) : taper collar width (physical length).
- \(\sigma_{\rm sm}\) : map smoothing width (numerical, can be small).
- \(\sigma_{\rm rds}\) : physical noise in the annealed operator.

**Computed (explicit or once-for-all):**
- \(C_1\) : a bound tied to your smoothstep/taper shape (e.g. \(\sup|w'|\) in rescaled coordinates).
- \(\|T\|_\infty\), \(\|T'\|_\infty\) (explicit for quadratic).
- \(\|T_\eta\|_\infty\) (bound via \(\|T\|_\infty + 2(a+1)\eta\)).
- The Gaussian multiplier \(\widehat{\bar\rho_{\sigma}}(k)\) (explicit).
- Numerical inflation radii for FFT/matrix entries (from BigFloat difference test).

**Rigorous bounds produced:**
- \(E_{\rm map} = \frac{\sqrt{2/\pi}}{\sigma_{\rm rds}}\|T-\widetilde T\|_\infty\).
- \(E_{\rm proj}\) in your chosen norm (often dominated by \(\widehat\rho(k)\) tails).
- \(E_{\rm num}\) from your numerical inflation policy.

---

## 11. Parameter heuristics (practical)

If you can choose \(\sigma_{\rm sm}\) small, a robust regime is:
- choose \(K\) collar size so \(\eta \approx 2K/M\),
- pick \(\sigma_{\rm sm} \ll \eta\) so smoothing error behaves like \(\sigma_{\rm sm}^2(1+C_1/\eta)\) and stays small,
- ensure \(N\sigma_{\rm sm}\) is at least “moderate” so the truncation tail after smoothing is tiny.

For the operator itself, \(\sigma_{\rm rds}\) (physical noise) typically already gives excellent spectral decay, so you may not need huge \(N\) unless \(\sigma_{\rm rds}\) is extremely small.

---

## 12. Acceptance tests (sanity checks)

- **Mass preservation:** check numerically that the column corresponding to \(\ell=0\) behaves correctly (Markov property). With balls, verify the property within radii.
- **Stability under \(N\):** increase \(N\to 2N\), verify \(P_N\) converges and certified error bound decreases.
- **Stability under \(\eta\):** reduce \(\eta\), verify taper error bound decreases linearly in \(\eta\), but watch curvature term \(\sim 1/\eta\) in smoothing bound.
- **Stability under \(\sigma_{\rm sm}\):** reduce \(\sigma_{\rm sm}\) and verify the smoothing error bound decreases as \(\sigma_{\rm sm}^2\).

---

## 13. Deliverables (what the implementation should output)

For each parameter set \((N,M,\eta,\sigma_{\rm sm},\sigma_{\rm rds})\), output:

1) `Ttilde_samples` and/or Fourier coefficients of \(\widetilde T\).
2) The matrix `P_N` (midpoints) and ball radii `P_N_ball`.
3) A certificate record:
   - `map_sup_error`,
   - `E_map`,
   - `E_proj`,
   - `E_num`,
   - `E_total`.
4) (Optional) a validated invariant density/eigenpair enclosure.

---

## 14. Notes on extending beyond quadratic

Everything above works for any \(T\) you can sample and bound in sup norm and (for smoothing error) a bound on \(\|T_\eta''\|_\infty\).
For non-smooth maps (e.g. \(|x|^\alpha\)), tapering + smoothing is even more valuable, because smoothing restores analytic-like Fourier decay.

