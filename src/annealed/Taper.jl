# Taper.jl - Tapering functions for periodic extension of maps
#
# This module provides tools to construct smooth periodic extensions of
# non-periodic maps on [-1,1] by tapering near endpoints.

"""
    smoothstep_quintic(u)

Quintic smoothstep function s:[0,1]→[0,1] with s(0)=0, s(1)=1,
and s'(0)=s'(1)=s''(0)=s''(1)=0 (C² matching).

Formula: s(u) = 6u⁵ - 15u⁴ + 10u³
"""
function smoothstep_quintic(u::Real)
    u = clamp(u, 0.0, 1.0)
    return u^3 * (10 - 15u + 6u^2)
end

"""
    smoothstep_septic(u)

Septic (degree 7) smoothstep function for C³ matching.
s'(0)=s'(1)=s''(0)=s''(1)=s'''(0)=s'''(1)=0.

Formula: s(u) = -20u⁷ + 70u⁶ - 84u⁵ + 35u⁴
"""
function smoothstep_septic(u::Real)
    u = clamp(u, 0.0, 1.0)
    return u^4 * (35 - 84u + 70u^2 - 20u^3)
end

"""
    smoothstep(u; order=2)

General smoothstep function with configurable order.
- order=2: quintic (C² matching)
- order=3: septic (C³ matching)
"""
function smoothstep(u::Real; order::Int=2)
    if order == 2
        return smoothstep_quintic(u)
    elseif order == 3
        return smoothstep_septic(u)
    else
        error("Unsupported smoothstep order: $order. Use 2 or 3.")
    end
end

"""
    taper_window(x, η; order=2)

Taper window function w_η(x) on [-1,1]:
- w_η(x) = 1 on [-1+η, 1-η]
- Smooth transition to 0 at endpoints using smoothstep

Returns a value in [0,1].
"""
function taper_window(x::Real, η::Real; order::Int=2)
    if x < -1 + η
        # Left collar: transition from 0 at x=-1 to 1 at x=-1+η
        u = (x + 1) / η
        return smoothstep(u; order=order)
    elseif x > 1 - η
        # Right collar: transition from 1 at x=1-η to 0 at x=1
        u = (1 - x) / η
        return smoothstep(u; order=order)
    else
        return 1.0
    end
end

"""
    taper_derivative_samples!(Tp_samples, M; K, L=5, enforce_zero_endpoints=true, mean_correction=true)

Modify derivative samples in-place to create a periodic function.

# Arguments
- `Tp_samples`: Vector of length M containing T'(x_j) samples on the grid x_j = -1 + 2j/M
- `M`: Grid size
- `K`: Collar size in grid points (η ≈ 2K/M)
- `L`: Moving average window length for smoothing transition
- `enforce_zero_endpoints`: If true, force derivative to 0 at endpoints
- `mean_correction`: If true, subtract mean to ensure ∫T'dx = 0

# Returns
Modified `Tp_samples` in-place.
"""
function taper_derivative_samples!(Tp_samples::Vector{<:Real}, M::Int;
                                   K::Int, L::Int=5,
                                   enforce_zero_endpoints::Bool=true,
                                   mean_correction::Bool=true)
    @assert length(Tp_samples) == M "Sample vector must have length M=$M"
    @assert K > 0 "Collar size K must be positive"
    @assert L >= 1 "Window length L must be at least 1"
    @assert K >= L "Collar size K must be >= window length L"

    if enforce_zero_endpoints
        # Force endpoint derivatives to zero
        Tp_samples[1] = 0.0
        # Note: For periodic, index M wraps to index 1, but we treat them separately
    end

    # Apply smooth transition in left collar (indices 1 to K)
    for j in 1:K
        # Blend factor: 0 at j=1, 1 at j=K
        blend = smoothstep_quintic((j - 1) / (K - 1))

        # Moving average of nearby interior values
        avg_start = max(K + 1, j)
        avg_end = min(K + L, M - K)
        if avg_start <= avg_end
            interior_avg = mean(Tp_samples[avg_start:avg_end])
        else
            interior_avg = Tp_samples[K + 1]
        end

        # Blend from 0 (at endpoint) to interior value
        Tp_samples[j] = blend * interior_avg
    end

    # Apply smooth transition in right collar (indices M-K+1 to M)
    for j in (M - K + 1):M
        # Blend factor: 1 at j=M-K+1, 0 at j=M
        blend = smoothstep_quintic((M - j) / (K - 1))

        # Moving average of nearby interior values
        avg_start = max(K + 1, M - K - L + 1)
        avg_end = min(M - K, M - 1)
        if avg_start <= avg_end
            interior_avg = mean(Tp_samples[avg_start:avg_end])
        else
            interior_avg = Tp_samples[M - K]
        end

        # Blend from interior value to 0 (at endpoint)
        Tp_samples[j] = blend * interior_avg
    end

    if mean_correction
        # Subtract mean to ensure periodicity of the integrated function
        # (∫T'dx = 0 over one period)
        Tp_samples .-= mean(Tp_samples)
    end

    return Tp_samples
end

"""
    integrate_derivative_fft(Tp_samples; T0=0.0)

Integrate derivative samples using FFT to recover the function.

Given samples of T'(x) on the grid, compute T(x) by:
1. FFT the derivative samples
2. Divide by iπk for k≠0 (integration in Fourier space)
3. Set zero mode to match T(0) = T0
4. IFFT to get function samples

# Arguments
- `Tp_samples`: Vector of T'(x_j) samples
- `T0`: Value of T(0) to fix the integration constant

# Returns
- `T_samples`: Vector of T(x_j) samples
- `That`: Fourier coefficients of T (optional, for further processing)
"""
function integrate_derivative_fft(Tp_samples::Vector{<:Number}; T0::Real=0.0)
    M = length(Tp_samples)

    # FFT of derivative (unnormalized)
    Tp_hat = fft(Tp_samples)

    # Integrate in Fourier space: T̂(k) = T̂'(k) / (iπk) for k≠0
    # For period 2, the wavenumber is πk
    That = similar(Tp_hat)
    That[1] = 0.0  # Will be set by T0 condition

    for j in 2:M
        # Map FFT index j to wavenumber k
        # FFT convention: j=1 → k=0, j=2 → k=1, ..., j=M/2+1 → k=M/2, j=M/2+2 → k=-M/2+1, ...
        if j <= M ÷ 2 + 1
            k = j - 1
        else
            k = j - 1 - M
        end

        if k != 0
            # T̂(k) = T̂'(k) / (iπk)
            # Note: FFT gives sum, not average, so scale appropriately
            That[j] = Tp_hat[j] / (im * π * k)
        end
    end

    # IFFT to get function samples
    T_samples = real.(ifft(That))

    # Adjust constant to match T(0) = T0
    # The sample at index 1 corresponds to x = -1
    # Find index for x = 0: x_j = -1 + 2j/M = 0 → j = M/2
    j0 = M ÷ 2 + 1
    current_T0 = T_samples[j0]
    T_samples .+= (T0 - current_T0)

    # Also adjust That[1] (the zero mode / average)
    That[1] = mean(T_samples) * M  # FFT convention

    return T_samples, That
end

"""
    build_tapered_map_samples(T_func, Tp_func, M; K, L=5, T0=nothing)

Build tapered periodic map samples from a map function and its derivative.

# Arguments
- `T_func`: The original map T(x)
- `Tp_func`: The derivative T'(x)
- `M`: Grid size (should be even, typically power of 2)
- `K`: Collar size in grid points
- `L`: Moving average window length
- `T0`: Value at x=0 (if nothing, use T_func(0))

# Returns
- `T_eta_samples`: Samples of the tapered map T_η(x_j)
- `That_eta`: Fourier coefficients of T_η
- `x_grid`: The grid points x_j = -1 + 2j/M for j=0,...,M-1
"""
function build_tapered_map_samples(T_func, Tp_func, M::Int;
                                    K::Int, L::Int=5, T0=nothing)
    @assert iseven(M) "M must be even"

    # Build grid: x_j = -1 + 2j/M for j = 0, ..., M-1
    x_grid = [-1.0 + 2.0 * j / M for j in 0:(M-1)]

    # Sample derivative
    Tp_samples = [Tp_func(x) for x in x_grid]

    # Apply tapering to derivative
    taper_derivative_samples!(Tp_samples, M; K=K, L=L)

    # Integrate to get function
    T0_val = isnothing(T0) ? T_func(0.0) : T0
    T_eta_samples, That_eta = integrate_derivative_fft(Tp_samples; T0=T0_val)

    return T_eta_samples, That_eta, x_grid
end

"""
    QuadraticMap

Represents the quadratic map T(x) = a - (a+1)x² on [-1,1].
"""
struct QuadraticMap
    a::Float64
end

(T::QuadraticMap)(x) = T.a - (T.a + 1) * x^2
derivative(T::QuadraticMap, x) = -2 * (T.a + 1) * x

"""
    build_tapered_quadratic(a, M; K, L=5)

Convenience function to build tapered samples for the quadratic map.
"""
function build_tapered_quadratic(a::Real, M::Int; K::Int, L::Int=5)
    T = QuadraticMap(a)
    T_func = x -> T(x)
    Tp_func = x -> derivative(T, x)
    return build_tapered_map_samples(T_func, Tp_func, M; K=K, L=L, T0=T(0.0))
end

# Exports
export smoothstep, smoothstep_quintic, smoothstep_septic
export taper_window
export taper_derivative_samples!, integrate_derivative_fft
export build_tapered_map_samples, build_tapered_quadratic
export QuadraticMap, derivative
