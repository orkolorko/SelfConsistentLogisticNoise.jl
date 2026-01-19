# API Reference

## Core Types

```@docs
LogisticMap
GaussianNoise
FourierDisc
SCProblem
```

## Observables and Coupling

```@docs
CosineObservable
LinearCoupling
TanhCoupling
```

## Problem Construction

```@docs
build_problem
build_B
```

## Solver

```@docs
solve_fixed_point
FixedPointResult
```

## Operator Application

```@docs
apply_Pc
reconstruct_density
```

## Parameter Sweeps

```@docs
sweep_delta
sweep_delta_a
multistart_solve
```

## Rigorous Verification

```@docs
verify_fixed_point
RigorousResult
GaussianConstants
compute_gaussian_constants
compute_jacobian_matrix
```

## I/O

```@docs
save_result
load_result
```

## Indexing Utilities

```@docs
modes
idx
mode
```
