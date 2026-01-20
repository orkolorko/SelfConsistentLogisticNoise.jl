# MapApproxBounds.jl - Rigorous bounds for map approximation errors
#
# This module provides explicit bounds for the error ||T - T̃||_∞ where
# T̃ = Π_N G_σ_sm T_η is the smoothed, tapered, truncated approximation.
#
# The total error is decomposed as:
#   ||T - T̃||_∞ ≤ ||T - T_η||_∞ + ||T_η - G_σ T_η||_∞ + ||G_σ T_η - Π_N G_σ T_η||_∞
#                 = E_taper     + E_smooth              + E_trunc

using IntervalArithmetic

"""
    MapApproxParams

Parameters for the map approximation.

# Fields
- `a`: Parameter of the quadratic map T(x) = a - (a+1)x²
- `η`: Taper collar width (physical length)
- `σ_sm`: Gaussian smoothing width (numerical)
- `N`: Fourier truncation parameter
- `C1`: Smoothstep constant (bound on |w'_η| in rescaled coordinates)
"""
struct MapApproxParams
    a::Float64      # Map parameter
    η::Float64      # Taper collar width
    σ_sm::Float64   # Smoothing width
    N::Int          # Fourier truncation
    C1::Float64     # Smoothstep constant
end

"""
    MapApproxParams(; a, η, σ_sm, N, smoothstep_order=2)

Construct MapApproxParams with automatic C1 computation based on smoothstep order.
"""
function MapApproxParams(; a::Real, η::Real, σ_sm::Real, N::Int, smoothstep_order::Int=2)
    # C1 is the supremum of |w'| for the smoothstep in rescaled coordinates
    # For quintic smoothstep s(u) = 6u⁵ - 15u⁴ + 10u³:
    #   s'(u) = 30u⁴ - 60u³ + 30u² = 30u²(u-1)²
    #   Maximum at u = 1/2: s'(1/2) = 30 * 1/4 * 1/4 = 30/16 = 15/8
    # Since w_η(x) transitions over length η, we have |w'_η| ≤ (15/8)/η
    # So C1 = 15/8 ≈ 1.875 for quintic smoothstep

    if smoothstep_order == 2
        C1 = 15.0 / 8.0  # Quintic smoothstep
    elseif smoothstep_order == 3
        # For septic smoothstep, compute max of derivative
        # s(u) = 35u⁴ - 84u⁵ + 70u⁶ - 20u⁷
        # s'(u) = 140u³ - 420u⁴ + 420u⁵ - 140u⁶ = 140u³(1-u)³
        # Maximum at u = 1/2: s'(1/2) = 140 * (1/8) * (1/8) = 140/64 = 35/16
        C1 = 35.0 / 16.0
    else
        error("Unsupported smoothstep order: $smoothstep_order")
    end

    return MapApproxParams(a, η, σ_sm, N, C1)
end

"""
    bound_T_sup(a)

Bound on ||T||_∞ for T(x) = a - (a+1)x² on [-1,1].

The maximum is at x=0: T(0) = a.
The minimum is at x=±1: T(±1) = a - (a+1) = -1.

So ||T||_∞ = max(|a|, 1).
"""
function bound_T_sup(a::Real)
    return max(abs(a), 1.0)
end

"""
    bound_Tp_sup(a)

Bound on ||T'||_∞ for T(x) = a - (a+1)x² on [-1,1].

T'(x) = -2(a+1)x, so ||T'||_∞ = 2(a+1) (achieved at x=±1).
"""
function bound_Tp_sup(a::Real)
    return 2 * (a + 1)
end

"""
    bound_taper_error(a, η)

Bound on ||T - T_η||_∞ (taper error) for the quadratic map.

Since T is modified only in collars of width η, and the slope is bounded by 2(a+1):
    ||T - T_η||_∞ ≤ 2(a+1)η
"""
function bound_taper_error(a::Real, η::Real)
    return 2 * (a + 1) * η
end

"""
    bound_taper_error(params::MapApproxParams)

Taper error bound using MapApproxParams.
"""
function bound_taper_error(params::MapApproxParams)
    return bound_taper_error(params.a, params.η)
end

"""
    bound_T_eta_sup(a, η)

Bound on ||T_η||_∞.

||T_η||_∞ ≤ ||T||_∞ + ||T - T_η||_∞ ≤ max(|a|, 1) + 2(a+1)η
"""
function bound_T_eta_sup(a::Real, η::Real)
    return bound_T_sup(a) + bound_taper_error(a, η)
end

"""
    bound_second_derivative_taper(a, η, C1)

Bound on ||T_η''||_∞ for the tapered quadratic map.

The second derivative of T is constant: T''(x) = -2(a+1).
Tapering introduces additional curvature in the collar proportional to C1/η.

Bound: ||T_η''||_∞ ≤ 2(a+1)(1 + C1/η)
"""
function bound_second_derivative_taper(a::Real, η::Real, C1::Real)
    return 2 * (a + 1) * (1 + C1 / η)
end

"""
    bound_second_derivative_taper(params::MapApproxParams)

Second derivative bound using MapApproxParams.
"""
function bound_second_derivative_taper(params::MapApproxParams)
    return bound_second_derivative_taper(params.a, params.η, params.C1)
end

"""
    bound_smoothing_error(a, η, σ_sm, C1)

Bound on ||T_η - G_σ T_η||_∞ (smoothing error).

Using the second-order bound for Gaussian convolution:
    ||f - G_σ f||_∞ ≤ (σ²/2) ||f''||_∞

So: ||T_η - G_σ T_η||_∞ ≤ (σ_sm²/2) · 2(a+1)(1 + C1/η)
                        = (a+1) σ_sm² (1 + C1/η)
"""
function bound_smoothing_error(a::Real, η::Real, σ_sm::Real, C1::Real)
    return (a + 1) * σ_sm^2 * (1 + C1 / η)
end

"""
    bound_smoothing_error(params::MapApproxParams)

Smoothing error bound using MapApproxParams.
"""
function bound_smoothing_error(params::MapApproxParams)
    return bound_smoothing_error(params.a, params.η, params.σ_sm, params.C1)
end

"""
    bound_truncation_error(T_eta_sup, σ_sm, N)

Bound on ||G_σ T_η - Π_N G_σ T_η||_∞ (Fourier truncation error after smoothing).

After Gaussian smoothing, Fourier coefficients decay like exp(-cσ²k²).
Using Gaussian tail bounds:

    ||G_σ T_η - Π_N G_σ T_η||_∞ ≤ (2||T_η||_∞)/(π²σ_sm²N) · exp(-π²σ_sm²N²/2)
"""
function bound_truncation_error(T_eta_sup::Real, σ_sm::Real, N::Int)
    return (2 * T_eta_sup) / (π^2 * σ_sm^2 * N) * exp(-π^2 * σ_sm^2 * N^2 / 2)
end

"""
    bound_truncation_error(params::MapApproxParams)

Truncation error bound using MapApproxParams.
"""
function bound_truncation_error(params::MapApproxParams)
    T_eta_sup = bound_T_eta_sup(params.a, params.η)
    return bound_truncation_error(T_eta_sup, params.σ_sm, params.N)
end

"""
    bound_map_sup_error(a, η, σ_sm, N, C1)

Total bound on ||T - T̃||_∞ where T̃ = Π_N G_σ T_η.

||T - T̃||_∞ ≤ E_taper + E_smooth + E_trunc
"""
function bound_map_sup_error(a::Real, η::Real, σ_sm::Real, N::Int, C1::Real)
    E_taper = bound_taper_error(a, η)
    E_smooth = bound_smoothing_error(a, η, σ_sm, C1)
    T_eta_sup = bound_T_eta_sup(a, η)
    E_trunc = bound_truncation_error(T_eta_sup, σ_sm, N)

    return E_taper + E_smooth + E_trunc
end

"""
    bound_map_sup_error(params::MapApproxParams)

Total map approximation error using MapApproxParams.
"""
function bound_map_sup_error(params::MapApproxParams)
    return bound_map_sup_error(params.a, params.η, params.σ_sm, params.N, params.C1)
end

"""
    MapApproxError

Detailed breakdown of map approximation errors.

# Fields
- `E_taper`: Taper error ||T - T_η||_∞
- `E_smooth`: Smoothing error ||T_η - G_σ T_η||_∞
- `E_trunc`: Truncation error ||G_σ T_η - Π_N G_σ T_η||_∞
- `E_total`: Total error ||T - T̃||_∞
"""
struct MapApproxError
    E_taper::Float64
    E_smooth::Float64
    E_trunc::Float64
    E_total::Float64
end

"""
    compute_map_approx_error(params::MapApproxParams)

Compute detailed map approximation error breakdown.
"""
function compute_map_approx_error(params::MapApproxParams)
    E_taper = bound_taper_error(params)
    E_smooth = bound_smoothing_error(params)
    E_trunc = bound_truncation_error(params)
    E_total = E_taper + E_smooth + E_trunc

    return MapApproxError(E_taper, E_smooth, E_trunc, E_total)
end

"""
    print_map_error_summary(err::MapApproxError)

Print a summary of the map approximation errors.
"""
function print_map_error_summary(err::MapApproxError)
    println("Map Approximation Error Breakdown:")
    println("  Taper error:      $(err.E_taper)")
    println("  Smoothing error:  $(err.E_smooth)")
    println("  Truncation error: $(err.E_trunc)")
    println("  Total error:      $(err.E_total)")
end

# ============================================================================
# Interval arithmetic versions for rigorous bounds
# ============================================================================

"""
    bound_map_sup_error_interval(a, η, σ_sm, N, C1)

Rigorous bound using interval arithmetic.
"""
function bound_map_sup_error_interval(a::Interval, η::Interval, σ_sm::Interval,
                                      N::Int, C1::Interval)
    E_taper = 2 * (a + 1) * η

    T_eta_sup = max(abs(a), interval(1)) + 2 * (a + 1) * η

    E_smooth = (a + 1) * σ_sm^2 * (1 + C1 / η)

    E_trunc = (2 * T_eta_sup) / (interval(π)^2 * σ_sm^2 * N) *
              exp(-interval(π)^2 * σ_sm^2 * N^2 / 2)

    return E_taper + E_smooth + E_trunc
end

# Exports
export MapApproxParams, MapApproxError
export bound_T_sup, bound_Tp_sup, bound_T_eta_sup
export bound_taper_error, bound_second_derivative_taper
export bound_smoothing_error, bound_truncation_error
export bound_map_sup_error, compute_map_approx_error
export print_map_error_summary
export bound_map_sup_error_interval
