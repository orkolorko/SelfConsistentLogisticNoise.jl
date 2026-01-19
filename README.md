# SelfConsistentLogisticNoise.jl

[![CI](https://github.com/orkolorko/SelfConsistentLogisticNoise.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/orkolorko/SelfConsistentLogisticNoise.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://orkolorko.github.io/SelfConsistentLogisticNoise.jl/dev/)

A Julia package for computing fixed points of **self-consistent noisy transfer operators** associated with the logistic family.

## Overview

This package implements numerical methods for finding fixed points of the self-consistent operator:

$$\mathcal{T}(f) = P_{c(f)} f$$

where:
- $T_a(x) = a \cdot x(1-x)$ is the logistic map
- $P_c$ is the noisy transfer operator with periodized Gaussian noise and shift $c$
- $c(f) = \delta \cdot G(m(f))$ is a self-consistent shift depending on the density $f$
- $m(f) = \langle \Phi, f \rangle$ is an observable (e.g., $\Phi(x) = \cos(2\pi x)$)

## Features

- **Fourier discretization** with FFT-based operator construction
- **Damped Picard iteration** for fixed-point computation
- **Warm-start continuation** for parameter sweeps
- **Rigorous verification** via Newton-Kantorovich theorem using [BallArithmetic.jl](https://github.com/orkolorko/BallArithmetic.jl)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/orkolorko/SelfConsistentLogisticNoise.jl")
```

## Quick Start

```julia
using SelfConsistentLogisticNoise

# Build problem: logistic map with a=3.83, noise σ=0.02, coupling δ=0.1
prob = build_problem(a=3.83, σ=0.02, N=64, δ=0.1)

# Solve for fixed point
result = solve_fixed_point(prob; α=0.3, tol=1e-10)

# Reconstruct density
x, f = reconstruct_density(result.fhat; npts=500)
```

## Documentation

See the [documentation](https://orkolorko.github.io/SelfConsistentLogisticNoise.jl/dev/) for detailed usage and API reference.

## Tutorial

A Jupyter notebook tutorial is available in `notebooks/fixed_point_tutorial.ipynb`.
