module SelfConsistentLogisticNoise

using LinearAlgebra
using Statistics
using Random
using FFTW
using JLD2
using DataFrames
using CSV
using BallArithmetic

# Fourier indexing utilities
include("sc/indexing.jl")

# Core types
include("sc/types.jl")

# Observable and coupling
include("sc/observable.jl")
include("sc/coupling.jl")

# Operator construction and application
include("sc/operator_build.jl")
include("sc/operator_apply.jl")

# Fixed-point solvers
include("sc/solver_picard.jl")
include("sc/solver_newton.jl")

# Sweeps and parameter studies
include("sc/sweeps.jl")

# I/O utilities
include("sc/io.jl")

# Rigorous diagnostics (Newton-Kantorovich verification)
include("sc/diagnostics.jl")

# Exports
export LogisticMap, GaussianNoise, FourierDisc, rhohat
export Observable, CosineObservable, Coupling
export LinearCoupling, TanhCoupling
export SCProblem
export modes, idx, mode
export build_B, apply_Pc, compute_m
export get_observable, get_delta
export solve_fixed_point, solve_newton, solve_hybrid, FixedPointResult
export sweep_delta, sweep_delta_a
export reconstruct_density
export save_result, load_result

# Rigorous verification exports
export GaussianConstants, compute_gaussian_constants
export RigorousResult, verify_fixed_point
export compute_jacobian_matrix

end # module
