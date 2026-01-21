# Krawczyk contraction test in Fourier \(L^2\) (mean-zero constraint)

## Goal

We want to certify that the preconditioned Newton map
\[
G(u) := u - A\,F(u)
\]
is a **contraction** on a set \(X\subset \mathbb R^{m-1}\) (the nonzero Fourier modes), so that it has a **unique fixed point** \(u_\ast\in X\), hence \(F(u_\ast)=0\).

Here:

- \(f\) is represented by Fourier coefficients \(\hat f\).
- \(\ell(f)=\int f\) corresponds to the **\(k=0\)** Fourier coefficient.
- The constraint subspace \(\mathcal E=\ker \ell\) is “**zero the \(k=0\) mode**”.
- Therefore the unknown is the vector of **nonzero modes** \(u_\perp \in \mathbb R^{m-1}\) (or \(\mathbb C^{m-1}\), but see “Complex coefficients” below).

We assume we have a discrete nonlinear map (e.g. self-consistent operator)
\[
F_N(u_\perp) \in \mathbb R^{m-1},
\qquad
J_N(u_\perp):=DF_N(u_\perp)\in \mathbb R^{(m-1)\times(m-1)}.
\]

The Krawczyk / contraction bound uses:
\[
Z \;\ge\;\sup_{u\in X}\|I - A\,J_N(u)\|_2,
\qquad
Y := \|A\,F_N(0)\|_2.
\]

If \(Z<1\) and \(G(X)\subseteq X\), then \(G\) has a unique fixed point in \(X\).

---

## Coordinate conventions (mean-zero constraint)

Let `full` denote the full Fourier coefficient vector (including \(k=0\)). Let `perp` denote the reduced vector with the \(k=0\) coordinate removed.

- `embed(u_perp)` creates a full vector with mode 0 set to 0 and the other modes filled from `u_perp`.
- `project(v_full)` removes the \(k=0\) coordinate.

In Stage 1 you work **only with `perp` vectors**, i.e. \(u=(\hat u_k)_{k\neq 0}\).

---

## Required primitives

You need **two rigorously evaluated** routines:

1. `F_perp(u_perp)`  
   Returns the reduced residual \(F_N(u_\perp)\) (nonzero modes only).

2. `J_perp_box(B)`  
   Given a **box** \(B\subset\mathbb R^{m-1}\), returns an **interval matrix** enclosure \( \mathbf J(B)\) such that  
   \[
   J_N(u)\in \mathbf J(B)\quad\forall u\in B.
   \]
   This is the only “hard” part: you need an interval/validated enclosure of the Jacobian on a set.

> ⚠️ Even if you want an \(L^2\)-ball \( \{ \|u\|_2\le r\}\), interval arithmetic works naturally on **boxes**, so we typically use a box \(B_r=[-r,r]^{m-1}\) that contains the ball. This is conservative but robust.

---

## Choosing the preconditioner \(A\)

Common choice:

- Evaluate a point Jacobian at the basepoint \(u=0\): \(J_0 \approx J_N(0)\) (float matrix).
- Set \(A := J_0^{-1}\) (float matrix).

Practical robust choice:

- Compute an interval Jacobian on a *small* box around 0, take midpoint:
  - \( \mathbf J(B_{\text{small}})\) via `J_perp_box`
  - \( J_0 := \mathrm{mid}(\mathbf J(B_{\text{small}}))\)
- Invert \(J_0\) in floating point to get \(A\).

---

## Bounding \(\|I-AJ\|_2\) rigorously with interval arithmetic

Direct spectral norm for an interval matrix is inconvenient, but we can bound it rigorously using:
\[
\|M\|_2 \;\le\; \sqrt{\|M\|_1\,\|M\|_\infty}.
\]
So we compute **upper bounds** on \(\|M\|_1\) and \(\|M\|_\infty\) from interval entries, then combine.

### Interval absolute bound for entries

For a real interval \(\mathbf a=[\underline a,\overline a]\), define
\[
\operatorname{absmax}(\mathbf a) := \max(|\underline a|,|\overline a|).
\]

For an interval matrix \(\mathbf M=(\mathbf m_{ij})\), define numeric bounds:
- Row sum bound: \(r_i := \sum_j \operatorname{absmax}(\mathbf m_{ij})\)
- Column sum bound: \(c_j := \sum_i \operatorname{absmax}(\mathbf m_{ij})\)

Then
\[
\| \mathbf M \|_\infty \le \max_i r_i,
\qquad
\| \mathbf M \|_1 \le \max_j c_j,
\qquad
\| \mathbf M \|_2 \le \sqrt{(\max_j c_j)(\max_i r_i)}.
\]

---

## Contraction + invariance on an \(L^2\) ball (implemented via a box)

We aim to certify contraction on the **ball**
\[
X_r := \{u:\|u\|_2\le r\}.
\]
Implementation uses the superset box \(B_r=[-r,r]^{m-1}\).

### Step A — compute \(Y\)
Compute the base residual:
- `r0 = F_perp(zeros(m-1))` (float or interval vector)
Compute:
\[
Y = \|A\,r_0\|_2.
\]
If `r0` is interval, use an upper bound:
\[
\|v\|_2 \le \sqrt{\sum_i (\sup|v_i|)^2}.
\]

### Step B — compute \(Z(r)\)
1. Build the box \(B_r=[-r,r]^{m-1}\).
2. Enclose Jacobian: \(\mathbf J = \mathbf J(B_r)\).
3. Form interval matrix:
   \[
   \mathbf M := I - A\,\mathbf J.
   \]
4. Compute numeric bounds:
   \[
   \| \mathbf M \|_\infty^{\uparrow},\quad \| \mathbf M \|_1^{\uparrow}
   \]
   from row/column sums of `absmax`.
5. Set
   \[
   Z(r) := \sqrt{\| \mathbf M \|_1^{\uparrow}\,\| \mathbf M \|_\infty^{\uparrow}}.
   \]
Then \(Z(r)\) is a rigorous upper bound on \(\sup_{u\in B_r}\|I-AJ_N(u)\|_2\), hence also on \(\sup_{u\in X_r}\|I-AJ_N(u)\|_2\).

### Step C — certify contraction
If \(Z(r)<1\), then \(G\) is a contraction on \(X_r\) (and on \(B_r\)) with factor \(Z(r)\).

### Step D — certify invariance of the ball (easy norm criterion)
Using the standard estimate:
\[
\|G(u)\|_2 \le \|G(0)\|_2 + Z(r)\,\|u\|_2,
\]
we get invariance \(G(X_r)\subseteq X_r\) if
\[
Y + Z(r)\,r \le r
\quad\Longleftrightarrow\quad
Y \le (1-Z(r))\,r.
\]

### Conclusion
If **both** hold:
- \(Z(r)<1\)
- \(Y \le (1-Z(r))\,r\)

then \(G\) maps \(X_r\) into itself and is a strict contraction there ⇒ **unique root** \(u_\ast\in X_r\).

---

## Practical radius selection loop

Because \(Z(r)\) depends on \(r\), use a simple inflation loop:

1. Start with `r = 1.1*Y` (or something small).
2. Compute `Z = Z(r)`.
3. If `Z >= 1`, shrink the domain or subdivide / improve bounds (see Notes).
4. Else set `r_new = Y/(1-Z)`.
5. If `r_new <= r*(1+tol)`, accept.
6. Else set `r = r_new` and repeat.

---

## Julia-like pseudocode skeleton

```julia
using IntervalArithmetic, LinearAlgebra

# absmax for a real interval
absmax(a::Interval) = max(abs(inf(a)), abs(sup(a)))

# 2-norm upper bound for interval vector
function norm2_upper(v::Vector{Interval})
    s = 0.0
    for vi in v
        a = absmax(vi)
        s += a*a
    end
    return sqrt(s)
end

# induced 1 and infinity norm upper bounds for interval matrix
function norm1_inf_upper(M::Matrix{Interval})
    n, m = size(M)
    colmax = 0.0
    for j in 1:m
        s = 0.0
        for i in 1:n
            s += absmax(M[i,j])
        end
        colmax = max(colmax, s)
    end
    rowmax = 0.0
    for i in 1:n
        s = 0.0
        for j in 1:m
            s += absmax(M[i,j])
        end
        rowmax = max(rowmax, s)
    end
    return colmax, rowmax
end

# main certification loop
function certify_krawczyk_l2(F_perp, J_perp_box, A; tol=1e-3, maxit=20)

    m1 = size(A,1)

    # Y = ||A F(0)||2 (allow F(0) to be interval-valued)
    r0 = F_perp(zeros(m1))                 # Vector{Float64} or Vector{Interval}
    yvec = A * r0
    Y = (eltype(yvec) <: Interval) ? norm2_upper(yvec) : norm(yvec, 2)

    r = 1.1 * Y

    for it in 1:maxit
        # Build box [-r,r]^(m-1)
        B = fill((-r)..(r), m1)            # Vector{Interval} representing a box

        # Interval Jacobian enclosure on the box
        J = J_perp_box(B)                  # Matrix{Interval}

        # M = I - A*J (interval matrix)
        AJ = A * J                         # promote to intervals
        M = similar(AJ)
        for i in 1:m1, j in 1:m1
            M[i,j] = (i==j ? 1..1 : 0..0) - AJ[i,j]
        end

        n1, ninf = norm1_inf_upper(M)
        Z = sqrt(n1 * ninf)                # rigorous upper bound on ||M||2

        if Z >= 1
            return (ok=false, reason="Z>=1", Y=Y, Z=Z, r=r, it=it)
        end

        r_new = Y / (1 - Z)

        if r_new <= r * (1 + tol)
            # invariance check: Y + Z*r <= r (should hold if r >= r_new)
            ok = (Y + Z*r <= r + 1e-15)
            return (ok=ok, Y=Y, Z=Z, r=r, it=it)
        end

        r = r_new
    end

    return (ok=false, reason="maxit", Y=Y, r=r)
end
