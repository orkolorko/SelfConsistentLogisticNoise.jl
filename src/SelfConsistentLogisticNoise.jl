module SelfConsistentLogisticNoise

using LinearAlgebra
using Statistics
using Random
using FFTW
using JLD2
using DataFrames
using CSV
using BallArithmetic
using IntervalArithmetic

# Fourier indexing utilities
include("sc/indexing.jl")

# Rigorous constants (must be loaded early, before other modules use them)
include("sc/rigorous_constants.jl")

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

# Krawczyk CAP verification
include("sc/krawczyk.jl")

# ============================================================================
# Annealed transfer operator (taper + smoothing)
# ============================================================================

# Tapering for periodic extension
include("annealed/Taper.jl")

# Gaussian smoothing
include("annealed/Smoothing.jl")

# Map approximation error bounds
include("annealed/MapApproxBounds.jl")

# Operator matrix assembly via FFT
include("annealed/OperatorAssembly.jl")

# Operator error bounds
include("annealed/OperatorBounds.jl")

# Rigorous assembly using BallArithmetic
include("annealed/RigorousAssembly.jl")

# Exports
export QuadraticMap, GaussianNoise, FourierDisc, rhohat
export Observable, CosineObservable, Coupling
export LinearCoupling, TanhCoupling
export SCProblem
export modes, idx, mode
export build_B, apply_Pc, compute_m
export bound_fft_aliasing_entry_error, bound_fft_aliasing_opnorm
export fft_aliasing_bound_matrix
export get_observable, get_delta
export solve_fixed_point, solve_newton, solve_hybrid, FixedPointResult
export sweep_delta, sweep_delta_a
export reconstruct_density
export save_result, load_result

# Rigorous constants exports
export pi_interval, e_interval, sqrt_interval, exp_interval
export sqrt_2_inv_upper, sqrt_2_inv_interval
export sqrt_3_inv_upper, sqrt_3_inv_interval
export sqrt_2_over_pi_upper, sqrt_2_over_pi_interval
export sqrt_pi_interval, sqrt_e_inv_upper, sqrt_e_inv_interval
export pi_sq_interval
export periodized_gaussian_derivative_norms_rigorous
export GaussianConstantsRigorous
export compute_gaussian_constants_rigorous, compute_gaussian_constants_bounds_rigorous
export CouplingConstantsRigorous
export compute_tanh_coupling_constants_rigorous, compute_linear_coupling_constants_rigorous
export cosine_observable_L2_norm_rigorous, cosine_observable_L2_norm_interval

# Rigorous verification exports
export GaussianConstants, compute_gaussian_constants
export RigorousResult, verify_fixed_point
export compute_jacobian_matrix

# Krawczyk CAP verification exports
export embed_perp, project_perp, perp_index, perp_mode
export compute_F_perp, compute_J_perp_matrix, compute_J_perp_ball
export compute_jacobian_lipschitz
export KrawczykResult, certify_krawczyk
export CAPResult, verify_fixed_point_CAP, print_CAP_certificate

# Annealed transfer operator exports (Taper)
export smoothstep, smoothstep_quintic, smoothstep_septic
export taper_window
export taper_derivative_samples!, integrate_derivative_fft
export build_tapered_map_samples, build_tapered_quadratic
export QuadraticMap

# Annealed transfer operator exports (Smoothing)
export gaussian_multiplier_period2
export smooth_fourier_coeffs!, smooth_fourier_coeffs
export smooth_samples_fft
export truncate_fourier_coeffs!, truncate_fourier_coeffs
export build_smoothed_tapered_map
export extract_fourier_band

# Annealed transfer operator exports (MapApproxBounds)
export MapApproxParams, MapApproxError
export bound_T_sup, bound_Tp_sup, bound_T_eta_sup
export bound_taper_error, bound_second_derivative_taper
export bound_smoothing_error, bound_truncation_error
export bound_map_sup_error, compute_map_approx_error
export print_map_error_summary

# Annealed transfer operator exports (OperatorAssembly)
export rho_hat_period2
export compute_gk_samples, compute_Akl_row
export assemble_PN_fft
export AnnealedOperatorProblem, AnnealedOperatorResult
export build_annealed_operator
export verify_markov_property

# Annealed transfer operator exports (OperatorBounds)
export bound_operator_map_sensitivity, bound_E_map
export bound_projection_tail_L1, bound_projection_tail_L2
export AnalyticStripConstants, compute_analytic_strip_constants
export bound_analytic_to_L2_projection
export OperatorErrorBounds, compute_operator_error_bounds
export print_operator_error_summary
export to_ball_matrix, operator_norm_ball
export OperatorCertificate, certify_operator, print_certificate

# Annealed transfer operator exports (RigorousAssembly)
export assemble_PN_rigorous
export extract_midpoints, extract_radii, max_radius
export RigorousOperatorResult, build_annealed_operator_rigorous
export rigorous_operator_norm_bound, rigorous_spectral_gap_bound

end # module
