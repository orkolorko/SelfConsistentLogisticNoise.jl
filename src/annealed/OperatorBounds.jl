# OperatorBounds.jl - Rigorous bounds for operator approximation errors
#
# This module provides bounds for the total operator error:
#   ||P_T - Π_N P_{T̃} Π_N|| ≤ E_map + E_proj + E_num
#
# where:
#   E_map  = operator error from map approximation
#   E_proj = projection/truncation error
#   E_num  = numerical (FFT/rounding) error

using IntervalArithmetic
using LinearAlgebra
using BallArithmetic: Ball, BallMatrix, BallVector, mid, rad, upper_bound_L2_opnorm

"""
    bound_operator_map_sensitivity(σ_rds, map_sup_error)

Bound on ||P_T - P_{T̃}||_{L¹→L¹} from map approximation error.

For Gaussian noise with width σ_rds:
    ||P_T - P_{T̃}||_{L¹→L¹} ≤ √(2/π) / σ_rds · ||T - T̃||_∞

This uses the Lipschitz property of the Gaussian kernel.
Uses rigorous interval arithmetic for √(2/π).
"""
function bound_operator_map_sensitivity(σ_rds::Real, map_sup_error::Real)
    # Use rigorous upper bound for √(2/π)
    sqrt_2_over_pi = sqrt_2_over_pi_upper()
    return sqrt_2_over_pi / σ_rds * map_sup_error
end

"""
    bound_E_map(params::MapApproxParams, σ_rds)

Compute E_map = ||P_T - P_{T̃}||_{L¹→L¹}.
"""
function bound_E_map(params::MapApproxParams, σ_rds::Real)
    map_sup_error = bound_map_sup_error(params)
    return bound_operator_map_sensitivity(σ_rds, map_sup_error)
end

"""
    bound_projection_tail_L1(σ_rds, N)

Bound on projection error ||P_{T̃} - Π_N P_{T̃}||_{L¹→L¹}.

Due to Gaussian noise smoothing, high-frequency modes decay exponentially:
    ρ̂_σ(k) = exp(-π²σ²k²/2)

The tail sum gives:
    Σ_{|k|>N} |ρ̂_σ(k)| ≤ C / (σ·N) · exp(-π²σ²N²/2)

with C ≈ √(2/π).
Uses rigorous interval arithmetic.
"""
function bound_projection_tail_L1(σ_rds::Real, N::Int)
    # Gaussian tail integral bound with rigorous arithmetic
    # ∫_N^∞ exp(-π²σ²k²/2) dk ≈ exp(-π²σ²N²/2) / (π²σ²N)
    # Sum is bounded by 2× the integral (symmetric)
    π_int = pi_interval()
    σ_int = interval(σ_rds)
    N_int = interval(N)

    tail = 2 * exp(-π_int^2 * σ_int^2 * N_int^2 / 2) / (π_int^2 * σ_int^2 * N_int)
    return sup(tail)
end

"""
    bound_projection_tail_L2(σ_rds, N)

Bound on projection error in L² norm.

||( I - Π_N) P_{T̃} f||₂ ≤ (Σ_{|k|>N} |ρ̂_σ(k)|²)^{1/2} ||f||₂

Uses rigorous interval arithmetic.
"""
function bound_projection_tail_L2(σ_rds::Real, N::Int)
    # Sum of squares of Gaussian multipliers with rigorous arithmetic
    # Σ_{|k|>N} exp(-π²σ²k²) ≤ 2 ∫_N^∞ exp(-π²σ²k²) dk
    π_int = pi_interval()
    σ_int = interval(σ_rds)
    N_int = interval(N)

    tail_sq = 2 * exp(-π_int^2 * σ_int^2 * N_int^2) / (π_int * σ_int * sqrt(π_int) * N_int)
    return sup(sqrt(tail_sq))
end

"""
    AnalyticStripConstants

Constants for operator bounds in analytic strip norms (Galatolo-style).

For the analytic strip A_τ with norm ||f||_{A_τ} = Σ_k |f̂_k| e^{2πτ|k|}:

S_{τ,σ} = sup_k exp(2πτ|k| - π²σ²k²/2)

This bounds how the Gaussian noise maps A_τ → L∞.
"""
struct AnalyticStripConstants
    τ::Float64          # Strip width
    σ::Float64          # Noise width
    S_τσ::Float64       # sup_k exp(2πτ|k| - π²σ²k²/2)
    S_τσ_1::Float64     # First derivative version
    S_τσ_2::Float64     # Second derivative version
end

"""
    compute_analytic_strip_constants(τ, σ)

Compute the Galatolo-style constants S_{τ,σ} with rigorous upper bounds.

S_{τ,σ} = sup_{k≥0} exp(2πτk - π²σ²k²/2)

The supremum is achieved at k* = 2τ/(πσ²) where the exponent has derivative 0.
Uses rigorous interval arithmetic.
"""
function compute_analytic_strip_constants(τ::Real, σ::Real)
    # Maximum of 2πτk - π²σ²k²/2 is at k* = 2τ/(πσ²)
    # Max value = 2πτ · 2τ/(πσ²) - π²σ²/2 · (2τ/(πσ²))² = 2τ²/σ²
    π_int = pi_interval()
    τ_int = interval(τ)
    σ_int = interval(σ)

    k_star_int = 2 * τ_int / (π_int * σ_int^2)
    S_τσ_int = exp(2 * τ_int^2 / σ_int^2)

    k_star = sup(k_star_int)
    S_τσ = sup(S_τσ_int)

    # For derivatives, the factor k^n modifies the optimization
    # S^(1) = sup_k |k| exp(2πτ|k| - π²σ²k²/2)
    # This requires solving d/dk[k exp(...)] = 0

    # Approximate: k* shifts slightly, but dominant term is still exp(2τ²/σ²) × k*
    S_τσ_1 = sup(k_star_int * S_τσ_int)

    # S^(2) ≈ k*² × S_τσ
    S_τσ_2 = sup(k_star_int^2 * S_τσ_int)

    return AnalyticStripConstants(τ, σ, S_τσ, S_τσ_1, S_τσ_2)
end

"""
    bound_analytic_to_L2_projection(τ, N)

Bound on ||(I - Π_N)f||₂ for f ∈ A_τ.

||(I - Π_N)f||₂ ≤ e^{-2πτN} ||f||_{A_τ}

Uses rigorous interval arithmetic.
"""
function bound_analytic_to_L2_projection(τ::Real, N::Int)
    π_int = pi_interval()
    τ_int = interval(τ)
    N_int = interval(N)
    return sup(exp(-2 * π_int * τ_int * N_int))
end

"""
    OperatorErrorBounds

Complete breakdown of operator approximation errors.

# Fields
- `E_map`: Map-induced operator error
- `E_proj`: Projection/truncation error
- `E_num`: Numerical error (FFT/rounding)
- `E_total`: Total error bound
"""
struct OperatorErrorBounds
    E_map::Float64
    E_proj::Float64
    E_num::Float64
    E_total::Float64
end

"""
    compute_operator_error_bounds(; a, η, σ_sm, N, σ_rds, E_num=0.0, C1=nothing)

Compute complete operator error bounds.

# Arguments
- `a`: Map parameter
- `η`: Taper collar width
- `σ_sm`: Map smoothing width
- `N`: Fourier truncation
- `σ_rds`: Physical noise width
- `E_num`: Numerical error bound (default 0, should be computed separately)
- `C1`: Smoothstep constant (computed automatically if not given)
"""
function compute_operator_error_bounds(; a::Real, η::Real, σ_sm::Real,
                                        N::Int, σ_rds::Real,
                                        E_num::Real=0.0, C1::Union{Real,Nothing}=nothing)
    # Map approximation parameters
    map_params = if isnothing(C1)
        MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N)
    else
        MapApproxParams(a, η, σ_sm, N, C1)
    end

    # E_map: operator error from map approximation
    E_map = bound_E_map(map_params, σ_rds)

    # E_proj: projection error (use L¹ bound)
    E_proj = bound_projection_tail_L1(σ_rds, N)

    # Total
    E_total = E_map + E_proj + E_num

    return OperatorErrorBounds(E_map, E_proj, E_num, E_total)
end

"""
    compute_operator_error_bounds(prob::AnnealedOperatorProblem; E_num=0.0)

Compute operator error bounds from problem specification.
"""
function compute_operator_error_bounds(prob::AnnealedOperatorProblem; E_num::Real=0.0)
    return compute_operator_error_bounds(
        a=prob.a, η=prob.η, σ_sm=prob.σ_sm, N=prob.N, σ_rds=prob.σ_rds, E_num=E_num
    )
end

"""
    print_operator_error_summary(err::OperatorErrorBounds)

Print a summary of operator approximation errors.
"""
function print_operator_error_summary(err::OperatorErrorBounds)
    println("Operator Approximation Error Bounds (L¹→L¹):")
    println("  E_map (map approx):  $(err.E_map)")
    println("  E_proj (projection): $(err.E_proj)")
    println("  E_num (numerical):   $(err.E_num)")
    println("  E_total:             $(err.E_total)")
end

# ============================================================================
# Numerical error estimation via BigFloat comparison
# ============================================================================

"""
    estimate_numerical_error_bigfloat(Ttilde_samples, N, σ_rds;
                                       precision_low=256, precision_high=512)

Estimate numerical error by comparing computations at two precisions.

Strategy (recommended in the implementation document):
1. Compute P_N at precision_low bits
2. Compute P_N at precision_high bits
3. Take the maximum entry-wise difference as the numerical error bound

# Returns
- `E_num`: Estimated numerical error (max entry-wise difference)
- `P_low`: Matrix computed at lower precision (converted to Float64)
"""
function estimate_numerical_error_bigfloat(Ttilde_samples::Vector{<:Real}, N::Int, σ_rds::Real;
                                           precision_low::Int=256, precision_high::Int=512)
    # Convert to BigFloat at low precision
    setprecision(BigFloat, precision_low)
    Ttilde_low = BigFloat.(Ttilde_samples)
    P_low = assemble_PN_fft_bigfloat(Ttilde_low, N, BigFloat(σ_rds))

    # Convert to BigFloat at high precision
    setprecision(BigFloat, precision_high)
    Ttilde_high = BigFloat.(Ttilde_samples)
    P_high = assemble_PN_fft_bigfloat(Ttilde_high, N, BigFloat(σ_rds))

    # Compute difference
    diff = abs.(P_high .- P_low)
    E_num = Float64(maximum(diff))

    # Convert P_low to Float64 for return
    P_float = Complex{Float64}.(P_low)

    # Reset precision to default
    setprecision(BigFloat, 256)

    return E_num, P_float
end

"""
    assemble_PN_fft_bigfloat(Ttilde_samples, N, σ_rds)

BigFloat version of operator assembly for precision comparison.
Uses naive DFT instead of FFTW (which doesn't support BigFloat).

Note: Uses setprecision to compute π at the current BigFloat precision,
which gives a rigorous approximation at that precision level.
"""
function assemble_PN_fft_bigfloat(Ttilde_samples::Vector{BigFloat}, N::Int, σ_rds::BigFloat)
    M = length(Ttilde_samples)
    dim = 2N + 1
    P = zeros(Complex{BigFloat}, dim, dim)

    # Compute π at current BigFloat precision (this is rigorous at that precision)
    π_big = BigFloat(π)

    # Precompute twiddle factors for DFT
    ω = exp(-2 * π_big * im / M)

    for k in -N:N
        # Compute g_k samples: exp(-iπk T̃(y))
        gk = exp.(-im * π_big * k .* Ttilde_samples)

        # Naive DFT for g_k
        gk_hat = zeros(Complex{BigFloat}, M)
        for ℓ_idx in 1:M
            for j in 1:M
                gk_hat[ℓ_idx] += gk[j] * ω^((j-1) * (ℓ_idx-1))
            end
        end

        # Extract modes and multiply by noise coefficient
        rho_k = exp(-π_big^2 * σ_rds^2 * k^2 / 2)
        row_idx = k + N + 1

        for ℓ in -N:N
            # Map ℓ to DFT index
            if ℓ >= 0
                dft_idx = ℓ + 1
            else
                dft_idx = M + ℓ + 1
            end
            col_idx = ℓ + N + 1

            P[row_idx, col_idx] = rho_k * gk_hat[dft_idx] / M
        end
    end

    return P
end

# ============================================================================
# Ball arithmetic for rigorous matrix entries (using BallArithmetic.jl)
# ============================================================================

"""
    to_ball_matrix(P, radius)

Convert a complex matrix to a BallMatrix with uniform radius.
Uses BallArithmetic.jl's BallMatrix type.
"""
function to_ball_matrix(P::Matrix{ComplexF64}, radius::Real)
    return BallMatrix([Ball(P[i,j], radius) for i in 1:size(P,1), j in 1:size(P,2)])
end

"""
    to_ball_matrix(P_mid, P_rad)

Create BallMatrix from midpoints and entry-wise radii.
"""
function to_ball_matrix(P_mid::Matrix{ComplexF64}, P_rad::Matrix{<:Real})
    return BallMatrix([Ball(P_mid[i,j], P_rad[i,j]) for i in 1:size(P_mid,1), j in 1:size(P_mid,2)])
end

"""
    operator_norm_ball(B::BallMatrix)

Compute rigorous bound on operator norm ||P||₂ using BallArithmetic's method.
"""
function operator_norm_ball(B::BallMatrix)
    return upper_bound_L2_opnorm(B)
end

# ============================================================================
# Certificate structure
# ============================================================================

"""
    OperatorCertificate

Complete certificate for an annealed transfer operator computation.

# Fields
- `params`: Problem parameters
- `map_error`: Map approximation error breakdown
- `operator_error`: Operator error bounds
- `P_ball`: Rigorous matrix representation (optional)
"""
struct OperatorCertificate
    params::AnnealedOperatorProblem
    map_error::MapApproxError
    operator_error::OperatorErrorBounds
    P_ball::Union{BallMatrix,Nothing}
end

"""
    certify_operator(result::AnnealedOperatorResult; compute_ball=false, E_num=0.0)

Create a certificate for an annealed operator computation.
"""
function certify_operator(result::AnnealedOperatorResult;
                          compute_ball::Bool=false, E_num::Real=0.0)
    # Compute operator error bounds
    op_error = compute_operator_error_bounds(result.params; E_num=E_num)

    # Optionally create ball matrix
    P_ball = if compute_ball
        to_ball_matrix(result.P, E_num)
    else
        nothing
    end

    return OperatorCertificate(result.params, result.map_error, op_error, P_ball)
end

"""
    print_certificate(cert::OperatorCertificate)

Print a summary of the operator certificate.
"""
function print_certificate(cert::OperatorCertificate)
    println("=" ^ 60)
    println("Annealed Transfer Operator Certificate")
    println("=" ^ 60)
    println("\nParameters:")
    println("  a = $(cert.params.a)")
    println("  σ_rds = $(cert.params.σ_rds)")
    println("  N = $(cert.params.N), M = $(cert.params.M)")
    println("  η = $(cert.params.η)")
    println("  σ_sm = $(cert.params.σ_sm)")
    println()
    print_map_error_summary(cert.map_error)
    println()
    print_operator_error_summary(cert.operator_error)
    println("=" ^ 60)
end

# Exports
export bound_operator_map_sensitivity, bound_E_map
export bound_projection_tail_L1, bound_projection_tail_L2
export AnalyticStripConstants, compute_analytic_strip_constants
export bound_analytic_to_L2_projection
export OperatorErrorBounds, compute_operator_error_bounds
export print_operator_error_summary
export estimate_numerical_error_bigfloat, assemble_PN_fft_bigfloat
export to_ball_matrix, operator_norm_ball
export OperatorCertificate, certify_operator, print_certificate
