# Smoothing.jl - Gaussian smoothing for periodic maps
#
# This module provides tools for Gaussian smoothing of maps on the period-2 torus [-1,1].
# The smoothing is purely numerical (width σ_sm) and independent from the physical
# noise in the annealed operator.

"""
    gaussian_multiplier_period2(k, σ)

Fourier multiplier for Gaussian convolution on period-2 torus.

For period 2, the wavenumber is πk, so the multiplier is:
    ρ̂_σ(k) = exp(-π²σ²k²/2)

This is the Fourier transform of the periodized Gaussian kernel.
"""
function gaussian_multiplier_period2(k::Integer, σ::Real)
    return exp(-π^2 * σ^2 * k^2 / 2)
end

"""
    gaussian_multiplier_period2(k, σ::Interval)

Interval arithmetic version for rigorous bounds.
"""
function gaussian_multiplier_period2(k::Integer, σ::Interval)
    return exp(-π^2 * σ^2 * k^2 / 2)
end

"""
    smooth_fourier_coeffs!(That, σ_sm)

Apply Gaussian smoothing to Fourier coefficients in-place.

Multiplies each coefficient by the Gaussian multiplier:
    T̂_smoothed(k) = ρ̂_σ(k) · T̂(k)

# Arguments
- `That`: Vector of Fourier coefficients (FFT convention: index 1 = k=0)
- `σ_sm`: Smoothing width
"""
function smooth_fourier_coeffs!(That::Vector{<:Number}, σ_sm::Real)
    M = length(That)

    for j in 1:M
        # Map FFT index to wavenumber
        if j <= M ÷ 2 + 1
            k = j - 1
        else
            k = j - 1 - M
        end

        That[j] *= gaussian_multiplier_period2(k, σ_sm)
    end

    return That
end

"""
    smooth_fourier_coeffs(That, σ_sm)

Non-mutating version of smooth_fourier_coeffs!
"""
function smooth_fourier_coeffs(That::Vector{<:Number}, σ_sm::Real)
    That_smooth = copy(That)
    smooth_fourier_coeffs!(That_smooth, σ_sm)
    return That_smooth
end

"""
    smooth_samples_fft(T_samples, σ_sm)

Apply Gaussian smoothing to function samples using FFT.

# Arguments
- `T_samples`: Vector of function samples T(x_j) on the grid
- `σ_sm`: Smoothing width

# Returns
- `T_smooth_samples`: Smoothed function samples
- `That_smooth`: Fourier coefficients of smoothed function
"""
function smooth_samples_fft(T_samples::Vector{<:Number}, σ_sm::Real)
    # FFT
    That = fft(T_samples)

    # Apply Gaussian multiplier
    smooth_fourier_coeffs!(That, σ_sm)

    # IFFT
    T_smooth_samples = real.(ifft(That))

    return T_smooth_samples, That
end

"""
    truncate_fourier_coeffs!(That, N)

Truncate Fourier coefficients to modes |k| ≤ N.

Sets coefficients with |k| > N to zero.

# Arguments
- `That`: Vector of Fourier coefficients (FFT convention)
- `N`: Maximum mode to keep
"""
function truncate_fourier_coeffs!(That::Vector{<:Number}, N::Int)
    M = length(That)
    @assert N <= M ÷ 2 "N must be ≤ M/2"

    for j in 1:M
        # Map FFT index to wavenumber
        if j <= M ÷ 2 + 1
            k = j - 1
        else
            k = j - 1 - M
        end

        if abs(k) > N
            That[j] = 0
        end
    end

    return That
end

"""
    truncate_fourier_coeffs(That, N)

Non-mutating version of truncate_fourier_coeffs!
"""
function truncate_fourier_coeffs(That::Vector{<:Number}, N::Int)
    That_trunc = copy(That)
    truncate_fourier_coeffs!(That_trunc, N)
    return That_trunc
end

"""
    build_smoothed_tapered_map(T_eta_samples, σ_sm, N)

Build the fully processed map T̃ = Π_N G_σ T_η from tapered samples.

# Arguments
- `T_eta_samples`: Samples of tapered map T_η(x_j)
- `σ_sm`: Smoothing width (numerical, can be small)
- `N`: Fourier truncation parameter

# Returns
- `Ttilde_samples`: Samples of T̃(x_j)
- `Ttilde_hat`: Fourier coefficients of T̃ (band-limited to |k| ≤ N)
"""
function build_smoothed_tapered_map(T_eta_samples::Vector{<:Number}, σ_sm::Real, N::Int)
    M = length(T_eta_samples)
    @assert N <= M ÷ 2 "N must be ≤ M/2"

    # FFT
    That = fft(T_eta_samples)

    # Apply Gaussian smoothing
    smooth_fourier_coeffs!(That, σ_sm)

    # Truncate to band-limited
    truncate_fourier_coeffs!(That, N)

    # IFFT
    Ttilde_samples = real.(ifft(That))

    return Ttilde_samples, That
end

"""
    extract_fourier_band(That_full, N)

Extract Fourier coefficients for modes -N to N from full FFT result.

Returns a vector of length 2N+1 with coefficients indexed as:
    result[1] = T̂(-N), result[N+1] = T̂(0), result[2N+1] = T̂(N)

# Arguments
- `That_full`: Full FFT result (length M)
- `N`: Maximum mode to extract

# Returns
- `That_band`: Vector of length 2N+1 with coefficients for k ∈ [-N, N]
"""
function extract_fourier_band(That_full::Vector{<:Number}, N::Int)
    M = length(That_full)
    @assert N <= M ÷ 2 "N must be ≤ M/2"

    That_band = zeros(eltype(That_full), 2N + 1)

    # FFT convention: index 1 = k=0, index 2 = k=1, ..., index M = k=-1
    # We want: result[k+N+1] = T̂(k) for k = -N, ..., N

    for k in -N:N
        # Map wavenumber k to FFT index
        if k >= 0
            fft_idx = k + 1
        else
            fft_idx = M + k + 1
        end

        # Map wavenumber k to output index
        out_idx = k + N + 1

        # Normalize: FFT gives sum, we want average (coefficient)
        That_band[out_idx] = That_full[fft_idx] / M
    end

    return That_band
end

# Exports
export gaussian_multiplier_period2
export smooth_fourier_coeffs!, smooth_fourier_coeffs
export smooth_samples_fft
export truncate_fourier_coeffs!, truncate_fourier_coeffs
export build_smoothed_tapered_map
export extract_fourier_band
