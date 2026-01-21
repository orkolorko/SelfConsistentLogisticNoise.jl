# Rigorous periodized Gaussian derivative constants

This note derives **rigorous upper bounds** for the $L^2$ norms of the
*periodized* Gaussian kernel derivatives on the torus. The constants are
computed from the Fourier-series definition and a geometric tail bound, so they
are certified upper bounds for $\|\rho_\sigma'\|_2$ and $\|\rho_\sigma''\|_2$
on $\mathbb T$.

## Setup

Let the periodized Gaussian on the torus have Fourier coefficients

\[
\hat\rho_\sigma(k) = \exp(-2\pi^2\sigma^2 k^2), \qquad k\in\mathbb Z.
\]

The $L^2(\mathbb T)$ norms of the derivatives are given by Parseval:

\[
\|\rho_\sigma'\|_2^2 = \sum_{k\in\mathbb Z} (2\pi k)^2 |\hat\rho_\sigma(k)|^2
 = \sum_{k\in\mathbb Z} (2\pi k)^2 e^{-4\pi^2\sigma^2 k^2},
\]
\[
\|\rho_\sigma''\|_2^2 = \sum_{k\in\mathbb Z} (2\pi k)^4 |\hat\rho_\sigma(k)|^2
 = \sum_{k\in\mathbb Z} (2\pi k)^4 e^{-4\pi^2\sigma^2 k^2}.
\]

By symmetry this is twice the sum over $k\ge 1$.

We use a rigorous tail bound based on the ratio test. For $p=2$ or $p=4$, define

\[
t_k := (2\pi k)^p e^{-4\pi^2\sigma^2 k^2}.
\]

For any $K\ge 1$,

\[
\frac{t_{k+1}}{t_k} = \Bigl(\frac{k+1}{k}\Bigr)^p \exp\bigl(-4\pi^2\sigma^2(2k+1)\bigr)
\le \Bigl(\frac{K+1}{K}\Bigr)^p \exp\bigl(-4\pi^2\sigma^2(2K+1)\bigr) =: q.
\]

If $q<1$ then the tail is bounded by a geometric series:

\[
\sum_{k=K}^{\infty} t_k \le \frac{t_K}{1-q}.
\]

Therefore,

\[
\sum_{k=1}^{\infty} t_k \le \sum_{k=1}^{K-1} t_k + \frac{t_K}{1-q}.
\]

This is the bound implemented in `periodized_gaussian_derivative_norms`.

## (Reference) Non-periodized Gaussian on $\mathbb R$

For comparison, the non-periodized Gaussian on $\mathbb R$ is

\[
\rho_\sigma(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\,\exp\Bigl(-\frac{x^2}{2\sigma^2}\Bigr).
\]

Then

\[
\rho_\sigma'(x) = -\frac{x}{\sigma^2}\,\rho_\sigma(x),
\qquad
\rho_\sigma''(x) = \Bigl(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\Bigr)\rho_\sigma(x).
\]

We also use the standard Gaussian integrals (for $a>0$):

\[
\int_{\mathbb R} e^{-a x^2}\,dx = \sqrt{\frac{\pi}{a}},
\quad
\int_{\mathbb R} x^2 e^{-a x^2}\,dx = \frac{\sqrt\pi}{2 a^{3/2}},
\quad
\int_{\mathbb R} x^4 e^{-a x^2}\,dx = \frac{3\sqrt\pi}{4 a^{5/2}}.
\]

## $\|\rho_\sigma'\|_2$ on $\mathbb R$

Since $\rho_\sigma'(x) = -(x/\sigma^2)\rho_\sigma(x)$ and
$\rho_\sigma(x)^2 = (2\pi\sigma^2)^{-1} e^{-x^2/\sigma^2}$,

\[
\|\rho_\sigma'\|_2^2
= \frac{1}{2\pi\sigma^2}\cdot\frac{1}{\sigma^4}\int_{\mathbb R}x^2 e^{-x^2/\sigma^2}\,dx.
\]

With $a = 1/\sigma^2$,

\[
\int_{\mathbb R}x^2 e^{-x^2/\sigma^2}\,dx = \frac{\sqrt\pi}{2} \sigma^3.
\]

Hence

\[
\|\rho_\sigma'\|_2^2 = \frac{1}{4\sqrt\pi\,\sigma^3},
\qquad
\|\rho_\sigma'\|_2 = \frac{1}{2\,\pi^{1/4}\,\sigma^{3/2}}.
\]

## $\|\rho_\sigma''\|_2$ on $\mathbb R$

We have

\[
\|\rho_\sigma''\|_2^2
= \frac{1}{2\pi\sigma^2}\int_{\mathbb R}
\Bigl(\frac{x^2}{\sigma^4}-\frac{1}{\sigma^2}\Bigr)^2 e^{-x^2/\sigma^2}\,dx.
\]

Let $a=1/\sigma^2$, so $(x^2/\sigma^4-1/\sigma^2)^2 = a^4 x^4 - 2a^3 x^2 + a^2$.
Using the Gaussian integrals above,

\[
\int_{\mathbb R} \bigl(a^4 x^4 - 2a^3 x^2 + a^2\bigr) e^{-a x^2}\,dx
= \sqrt\pi\,a^{3/2}\Bigl(\frac{3}{4} - 1 + 1\Bigr)
= \frac{3}{4}\sqrt\pi\,a^{3/2}.
\]

Therefore

\[
\|\rho_\sigma''\|_2^2 = \frac{3}{8\sqrt\pi\,\sigma^5},
\qquad
\|\rho_\sigma''\|_2 = \frac{\sqrt{3}}{2\sqrt{2}\,\pi^{1/4}\,\sigma^{5/2}}.
\]

## Resulting constants on $\mathbb R$

We use the exact $L^2$ norms as the rigorous constants:

\[
C_J = \|\rho_\sigma'\|_2 = \frac{1}{2\,\pi^{1/4}\,\sigma^{3/2}},
\qquad
C_J^{(2)} = \|\rho_\sigma''\|_2 = \frac{\sqrt{3}}{2\sqrt{2}\,\pi^{1/4}\,\sigma^{5/2}}.
\]

Because these are exact integrals, they are rigorous upper bounds (in fact
exact values) for the non-periodized Gaussian.
