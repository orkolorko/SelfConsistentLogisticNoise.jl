# rigorous_constants.jl - Rigorous computation of mathematical constants
#
# This module provides rigorous upper and lower bounds for mathematical constants
# using interval arithmetic. All bounds are computed using IntervalArithmetic.jl
# to ensure correct rounding.
#
# Key principle: When computing upper bounds for error estimates, we must use
# rigorous upper bounds on constants. When computing lower bounds, we use
# rigorous lower bounds. Interval arithmetic handles this automatically.

using IntervalArithmetic

# ============================================================================
# Rigorous constant enclosures
# ============================================================================

"""
    pi_interval()

Return a rigorous interval enclosure of π.
"""
@inline pi_interval() = interval(π)

"""
    e_interval()

Return a rigorous interval enclosure of Euler's number e.
"""
@inline e_interval() = interval(ℯ)

"""
    sqrt_interval(x)

Compute a rigorous interval enclosure of √x.
"""
@inline sqrt_interval(x::Real) = sqrt(interval(x))
@inline sqrt_interval(x::Interval) = sqrt(x)

"""
    exp_interval(x)

Compute a rigorous interval enclosure of exp(x).
"""
@inline exp_interval(x::Real) = exp(interval(x))
@inline exp_interval(x::Interval) = exp(x)

# ============================================================================
# Common constant combinations (rigorous upper bounds)
# ============================================================================

"""
    sqrt_2_inv_upper()

Rigorous upper bound for 1/√2.
"""
function sqrt_2_inv_upper()
    return sup(interval(1) / sqrt(interval(2)))
end

"""
    sqrt_2_inv_interval()

Rigorous interval enclosure for 1/√2.
"""
function sqrt_2_inv_interval()
    return interval(1) / sqrt(interval(2))
end

"""
    sqrt_3_inv_upper()

Rigorous upper bound for 1/√3.
"""
function sqrt_3_inv_upper()
    return sup(interval(1) / sqrt(interval(3)))
end

"""
    sqrt_3_inv_interval()

Rigorous interval enclosure for 1/√3.
"""
function sqrt_3_inv_interval()
    return interval(1) / sqrt(interval(3))
end

"""
    sqrt_2_over_pi_upper()

Rigorous upper bound for √(2/π).
"""
function sqrt_2_over_pi_upper()
    return sup(sqrt(interval(2) / pi_interval()))
end

"""
    sqrt_2_over_pi_interval()

Rigorous interval enclosure for √(2/π).
"""
function sqrt_2_over_pi_interval()
    return sqrt(interval(2) / pi_interval())
end

"""
    sqrt_pi_interval()

Rigorous interval enclosure for √π.
"""
function sqrt_pi_interval()
    return sqrt(pi_interval())
end

"""
    sqrt_e_inv_upper()

Rigorous upper bound for 1/√e.
"""
function sqrt_e_inv_upper()
    return sup(interval(1) / sqrt(e_interval()))
end

"""
    sqrt_e_inv_interval()

Rigorous interval enclosure for 1/√e.
"""
function sqrt_e_inv_interval()
    return interval(1) / sqrt(e_interval())
end

"""
    pi_sq_interval()

Rigorous interval enclosure for π².
"""
function pi_sq_interval()
    return pi_interval()^2
end

# ============================================================================
# Rigorous bounds for Gaussian-related constants
# ============================================================================

"""
    gaussian_fourier_coeff_bound(σ, k)

Rigorous interval enclosure for the Gaussian Fourier coefficient:
    ρ̂_σ(k) = exp(-π²σ²k²/2)
"""
function gaussian_fourier_coeff_interval(σ::Real, k::Integer)
    σ_int = interval(σ)
    k_int = interval(k)
    return exp(-pi_sq_interval() * σ_int^2 * k_int^2 / 2)
end

"""
    gaussian_derivative_factor_interval(σ, k, p)

Rigorous interval enclosure for (πk)^p * exp(-π²σ²k²/2).
Used in periodized Gaussian derivative norms.
"""
function gaussian_derivative_factor_interval(σ::Real, k::Integer, p::Integer)
    σ_int = interval(σ)
    k_int = interval(abs(k))
    return (pi_interval() * k_int)^p * exp(-pi_sq_interval() * σ_int^2 * k_int^2 / 2)
end

# ============================================================================
# Rigorous periodized Gaussian derivative norms
# ============================================================================

"""
    periodized_gaussian_derivative_norms_rigorous(σ; K_start=1, K_max=10_000)

Compute rigorous upper bounds for the L2 norms of the derivatives of the
periodized Gaussian kernel on the torus:

    ||ρ'_σ||_2^2 = Σ_{k∈ℤ} (πk)^2 exp(-π²σ²k²)
    ||ρ''_σ||_2^2 = Σ_{k∈ℤ} (πk)^4 exp(-π²σ²k²)

Returns (C_J_upper, C_J_2_upper) as rigorous upper bounds.
"""
function periodized_gaussian_derivative_norms_rigorous(σ::Real; K_start::Int=1, K_max::Int=10_000)
    σ <= 0 && error("σ must be positive")

    σ_int = interval(σ)
    a_int = pi_sq_interval() * σ_int^2

    function tail_bound_rigorous(p::Int)
        K = K_start
        while K <= K_max
            K_int = interval(K)
            # ratio = ((K+1)/K)^p * exp(-a * (2K+1))
            ratio_int = ((K_int + 1) / K_int)^p * exp(-a_int * (2 * K_int + 1))
            if sup(ratio_int) < 1
                term_K = (pi_interval() * K_int)^p * exp(-a_int * K_int^2)
                # tail = term_K / (1 - ratio)
                tail_int = term_K / (interval(1) - ratio_int)
                return K, sup(tail_int)
            end
            K += 1
        end
        error("Failed to find tail bound with ratio < 1 up to K_max=$K_max")
    end

    function series_with_tail_rigorous(p::Int)
        K, tail_upper = tail_bound_rigorous(p)
        # Finite sum with interval arithmetic
        finite_int = sum(gaussian_derivative_factor_interval(σ, k, p) for k in 1:K-1; init=interval(0))
        # Total = 2 * (finite + tail) for symmetric sum over k ≠ 0
        total_upper = 2 * (sup(finite_int) + tail_upper)
        return total_upper
    end

    # C_J = sqrt(Σ (πk)^2 exp(-π²σ²k²))
    C_J_upper = sqrt(series_with_tail_rigorous(2))
    # C_J_2 = sqrt(Σ (πk)^4 exp(-π²σ²k²))
    C_J_2_upper = sqrt(series_with_tail_rigorous(4))

    return C_J_upper, C_J_2_upper
end

# ============================================================================
# Rigorous Gaussian smoothing constants (Lemma 13)
# ============================================================================

"""
    GaussianConstantsRigorous

Container for rigorous upper bounds on Gaussian smoothing constants.
"""
struct GaussianConstantsRigorous
    S_τσ::Float64       # Rigorous upper bound on S_{τ,σ}
    S1_τσ::Float64      # Rigorous upper bound on S^(1)_{τ,σ}
    S2_τσ::Float64      # Rigorous upper bound on S^(2)_{τ,σ}
    S1_0σ::Float64      # Rigorous upper bound on S^(1)_{0,σ}
    S2_0σ::Float64      # Rigorous upper bound on S^(2)_{0,σ}
end

"""
    compute_gaussian_constants_rigorous(σ, τ)

Compute rigorous upper bounds for Gaussian smoothing constants.
Uses interval arithmetic to ensure correct rounding.

S_{τ,σ} := sup_k exp(πτ|k| - (π²/2)σ²k²)
"""
function compute_gaussian_constants_rigorous(σ::Float64, τ::Float64)
    σ_int = interval(σ)
    τ_int = interval(τ)
    π_int = pi_interval()

    # For S_{τ,σ}: maximizer near k₀ = τ/(πσ²)
    k0_approx = τ / (π * σ^2)
    k0 = max(0, floor(Int, k0_approx) - 2)
    k0_max = ceil(Int, k0_approx) + 2

    f_int(k) = exp(π_int * τ_int * interval(abs(k)) - interval(0.5) * π_int^2 * σ_int^2 * interval(k)^2)
    f1_int(k) = (π_int * interval(abs(k))) * f_int(k)
    f2_int(k) = (π_int * interval(abs(k)))^2 * f_int(k)

    # S_{τ,σ}: check neighborhood of k₀
    S_τσ = maximum(sup(f_int(k)) for k in k0:k0_max)

    # S^(1)_{τ,σ}: maximizer of k * exp(...)
    k1_approx = (1 + π * τ) / (π^2 * σ^2)
    k1 = max(1, floor(Int, k1_approx) - 2)
    k1_max = ceil(Int, k1_approx) + 2
    S1_τσ = maximum(sup(f1_int(k)) for k in k1:k1_max)

    # S^(2)_{τ,σ}
    k2_approx = (2 + π * τ) / (π^2 * σ^2)
    k2 = max(1, floor(Int, k2_approx) - 2)
    k2_max = ceil(Int, k2_approx) + 2
    S2_τσ = maximum(sup(f2_int(k)) for k in k2:k2_max)

    # For τ = 0 cases
    g_int(k) = exp(-interval(0.5) * π_int^2 * σ_int^2 * interval(k)^2)
    g1_int(k) = (π_int * interval(k)) * g_int(k)
    g2_int(k) = (π_int * interval(k))^2 * g_int(k)

    # S^(1)_{0,σ}: maximizer at k = 1/(πσ√2)
    k1_0_approx = 1 / (π * σ * sqrt(2))
    k1_0 = max(1, floor(Int, k1_0_approx) - 2)
    k1_0_max = ceil(Int, k1_0_approx) + 2
    S1_0σ = maximum(sup(g1_int(k)) for k in k1_0:k1_0_max)

    # S^(2)_{0,σ}: maximizer at k = √2/(πσ)
    k2_0_approx = sqrt(2) / (π * σ)
    k2_0 = max(1, floor(Int, k2_0_approx) - 2)
    k2_0_max = ceil(Int, k2_0_approx) + 2
    S2_0σ = maximum(sup(g2_int(k)) for k in k2_0:k2_0_max)

    return GaussianConstantsRigorous(S_τσ, S1_τσ, S2_τσ, S1_0σ, S2_0σ)
end

"""
    compute_gaussian_constants_bounds_rigorous(σ, τ)

Compute explicit rigorous upper bounds (Lemma 13 formulas).
Uses interval arithmetic for all computations.
"""
function compute_gaussian_constants_bounds_rigorous(σ::Float64, τ::Float64)
    σ_int = interval(σ)
    τ_int = interval(τ)
    π_int = pi_interval()
    e_int = e_interval()

    exp_factor = exp(τ_int^2 / (2 * σ_int^2))

    S_τσ = sup(exp_factor)
    S1_τσ = sup(exp_factor * (τ_int / σ_int^2 + interval(1) / (π_int * σ_int * sqrt(e_int))))
    S2_τσ = sup(exp_factor * (τ_int^2 / σ_int^4 +
                              2 * τ_int / (π_int * σ_int^3 * sqrt(e_int)) +
                              interval(2) / (π_int^2 * σ_int^2 * e_int)))
    S1_0σ = sup(interval(1) / (π_int * σ_int * sqrt(e_int)))
    S2_0σ = sup(interval(2) / (π_int^2 * σ_int^2 * e_int))

    return GaussianConstantsRigorous(S_τσ, S1_τσ, S2_τσ, S1_0σ, S2_0σ)
end

# ============================================================================
# Rigorous coupling constants
# ============================================================================

"""
    CouplingConstantsRigorous

Rigorous upper bounds for coupling function constants.
"""
struct CouplingConstantsRigorous
    Lip_G::Float64      # Lipschitz constant of G
    L_G::Float64        # sup|G'|
    L_Gp::Float64       # Lipschitz constant of G' (sup|G''|)
end

"""
    compute_tanh_coupling_constants_rigorous(β)

Compute rigorous upper bounds for tanh coupling constants.
G(m) = tanh(βm), so:
- G'(m) = β sech²(βm), with ||G'||_∞ = β at m=0
- G''(m) = -2β² sech²(βm) tanh(βm)
- max|G''| = 2β²/(3√3) at tanh(βm) = ±1/√3
"""
function compute_tanh_coupling_constants_rigorous(β::Real)
    β_int = interval(β)

    # Lip(G) = ||G'||_∞ = β
    Lip_G = sup(β_int)
    L_G = Lip_G

    # L_Gp = max|G''| = 2β² * (2/3)^(3/2) = 2β²/(3√3)
    # More precisely: max of |2β² sech²(x) tanh(x)|
    # Occurs at tanh(x) = 1/√3, giving sech²(x) = 2/3
    # So max = 2β² * (2/3) * (1/√3) = 4β²/(3√3)
    # Alternative form: 2β² * (2/3)^(3/2) ≈ 0.544β²
    coeff = interval(2) * (interval(2) / interval(3))^(interval(3) / interval(2))
    L_Gp = sup(β_int^2 * coeff)

    return CouplingConstantsRigorous(Lip_G, L_G, L_Gp)
end

"""
    compute_linear_coupling_constants_rigorous()

Rigorous constants for linear coupling G(m) = m.
"""
function compute_linear_coupling_constants_rigorous()
    return CouplingConstantsRigorous(1.0, 1.0, 0.0)
end

# ============================================================================
# Rigorous observable norms
# ============================================================================

"""
    cosine_observable_L2_norm_rigorous()

Rigorous upper bound for ||cos(πx)||_2 on the torus.
||cos(πx)||_2 = 1/√2
"""
function cosine_observable_L2_norm_rigorous()
    return sqrt_2_inv_upper()
end

"""
    cosine_observable_L2_norm_interval()

Rigorous interval enclosure for ||cos(πx)||_2.
"""
function cosine_observable_L2_norm_interval()
    return sqrt_2_inv_interval()
end

# ============================================================================
# Exports
# ============================================================================

export pi_interval, e_interval, sqrt_interval, exp_interval
export sqrt_2_inv_upper, sqrt_2_inv_interval
export sqrt_3_inv_upper, sqrt_3_inv_interval
export sqrt_2_over_pi_upper, sqrt_2_over_pi_interval
export sqrt_pi_interval, sqrt_e_inv_upper, sqrt_e_inv_interval
export pi_sq_interval
export gaussian_fourier_coeff_interval, gaussian_derivative_factor_interval
export periodized_gaussian_derivative_norms_rigorous
export GaussianConstantsRigorous
export compute_gaussian_constants_rigorous, compute_gaussian_constants_bounds_rigorous
export CouplingConstantsRigorous
export compute_tanh_coupling_constants_rigorous, compute_linear_coupling_constants_rigorous
export cosine_observable_L2_norm_rigorous, cosine_observable_L2_norm_interval
