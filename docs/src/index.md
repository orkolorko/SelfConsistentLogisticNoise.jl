# SelfConsistentLogisticNoise.jl

A Julia package for computing fixed points of **self-consistent noisy transfer operators** associated with the quadratic family on the period-2 torus.

## Overview

This package implements numerical methods for finding fixed points of the self-consistent operator:

```math
\mathcal{T}(f) = P_{c(f)} f
```

where:
- ``T_a(x) = a - (a+1)x^2`` is the quadratic map on ``[-1,1]``
- ``P_c`` is the noisy transfer operator with periodized Gaussian noise and shift ``c``
- ``c(f) = \delta \cdot G(m(f))`` is a self-consistent shift depending on the density ``f``
- ``m(f) = \langle \Phi, f \rangle`` is an observable (e.g., ``\Phi(x) = \cos(\pi x)``)

## Features

- **Fourier discretization** with FFT-based operator construction
- **Damped Picard iteration** for fixed-point computation
- **Warm-start continuation** for parameter sweeps
- **Rigorous verification** via Newton-Kantorovich theorem using [BallArithmetic.jl](https://github.com/orkolorko/BallArithmetic.jl)

## Quick Start

```julia
using SelfConsistentLogisticNoise

# Build problem: quadratic map with a=0.915, noise σ=0.02, coupling δ=0.1
prob = build_problem(a=0.915, σ=0.02, N=64, δ=0.1)

# Solve for fixed point
result = solve_fixed_point(prob; α=0.3, tol=1e-10, verbose=true)

# Reconstruct and plot density
x, f = reconstruct_density(result.fhat; npts=500)
```

## Installation

This package is not registered. Install via:

```julia
using Pkg
Pkg.add(url="https://github.com/orkolorko/SelfConsistentLogisticNoise.jl")
```

## Contents

```@contents
Pages = ["theory.md", "tutorials/fixed_point_tutorial.md", "api.md"]
Depth = 2
```
