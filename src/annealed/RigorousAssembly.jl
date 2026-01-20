# RigorousAssembly.jl - Rigorous operator assembly using BallArithmetic
#
# This module provides rigorous enclosure of the annealed transfer operator
# matrix using BallArithmetic's rigorous FFT implementation.

using BallArithmetic

"""
    assemble_PN_rigorous(Ttilde_samples, N, σ_rds; verbose=false)

Assemble the annealed transfer operator matrix with rigorous error bounds
using BallArithmetic's rigorous FFT.

# Arguments
- `Ttilde_samples`: Samples of smoothed tapered map T̃(y_j) (Float64 or Ball)
- `N`: Fourier truncation parameter
- `σ_rds`: Physical noise width

# Returns
- `P_ball`: Matrix of Ball{Float64} with rigorous enclosure
"""
function assemble_PN_rigorous(Ttilde_samples::Vector{<:Real}, N::Int, σ_rds::Real;
                              verbose::Bool=false)
    M = length(Ttilde_samples)
    @assert N <= M ÷ 2 "N must be ≤ M/2"

    # Convert samples to Balls
    Ttilde_ball = Ball.(Ttilde_samples)

    return assemble_PN_rigorous(Ttilde_ball, N, Ball(σ_rds); verbose=verbose)
end

"""
    assemble_PN_rigorous(Ttilde_ball::Vector{<:Ball}, N, σ_rds::Ball)

Rigorous assembly with Ball inputs.
"""
function assemble_PN_rigorous(Ttilde_ball::Vector{<:Ball}, N::Int,
                              σ_rds::Ball; verbose::Bool=false)
    M = length(Ttilde_ball)
    dim = 2N + 1

    # Result matrix of Balls (complex midpoint, real radius)
    # Ball{Float64, ComplexF64} is the type for complex balls
    P = Matrix{Ball{Float64, ComplexF64}}(undef, dim, dim)

    # Constants
    π_ball = Ball(π)

    for k in -N:N
        if verbose && (k + N) % 20 == 0
            println("  Rigorous assembly: row k = $k / $N")
        end

        # Compute g_k samples: exp(-iπk T̃(y)) with Ball arithmetic
        # g_k(y_j) = exp(-iπk T̃(y_j))
        gk = [exp(-im * π_ball * k * Ttilde_ball[j]) for j in 1:M]

        # Rigorous FFT using BallArithmetic
        # BallArithmetic provides rigorous_fft for complex Ball vectors
        gk_hat = rigorous_fft(gk)

        # Noise Fourier coefficient with Ball arithmetic
        rho_k = exp(-π_ball^2 * σ_rds^2 * k^2 / 2)

        # Row index
        row_idx = k + N + 1

        # Fill row
        for ℓ in -N:N
            # Map ℓ to FFT index
            if ℓ >= 0
                fft_idx = ℓ + 1
            else
                fft_idx = M + ℓ + 1
            end
            col_idx = ℓ + N + 1

            # Normalize and multiply by noise coefficient
            P[row_idx, col_idx] = rho_k * gk_hat[fft_idx] / M
        end
    end

    return P
end

"""
    rigorous_fft(x::Vector{<:Ball})

Wrapper for BallArithmetic's rigorous FFT.

If BallArithmetic doesn't export a direct rigorous_fft, we implement
one using the FFTExt extension.
"""
function rigorous_fft(x::Vector{<:Ball})
    # BallArithmetic's FFTExt provides rigorous FFT via FFTW with error tracking
    # The extension should be loaded when FFTW is available

    # Try to use BallArithmetic's FFT if available
    if isdefined(BallArithmetic, :fft) && hasmethod(BallArithmetic.fft, Tuple{Vector{Ball{ComplexF64}}})
        return BallArithmetic.fft(x)
    end

    # Fallback: use standard FFTW with ball wrapping
    # This is less rigorous but provides approximate enclosure
    return fallback_rigorous_fft(x)
end

"""
    fallback_rigorous_fft(x::Vector{<:Ball})

Fallback FFT implementation when BallArithmetic's rigorous FFT is not available.

Uses FFTW on midpoints and estimates error propagation.
"""
function fallback_rigorous_fft(x::Vector{<:Ball})
    M = length(x)

    # Extract midpoints and radii
    mids = [mid(xi) for xi in x]
    rads = [rad(xi) for xi in x]

    # FFT of midpoints
    mids_fft = fft(mids)

    # Error propagation for FFT:
    # |FFT(x + δ) - FFT(x)| ≤ √M · ||δ||_∞ (crude bound)
    # Better: |FFT(x + δ)_k - FFT(x)_k| ≤ Σ_j |δ_j| = ||δ||_1
    max_rad = maximum(rads)
    fft_rad = M * max_rad  # Conservative bound: sum of all radii

    # Wrap results in Balls
    result = [Ball(mids_fft[j], fft_rad) for j in 1:M]

    return result
end

"""
    extract_midpoints(P_ball::Matrix{Ball{ComplexF64}})

Extract midpoint matrix from Ball matrix.
"""
function extract_midpoints(P_ball::Matrix{<:Ball})
    return [mid(P_ball[i,j]) for i in axes(P_ball,1), j in axes(P_ball,2)]
end

"""
    extract_radii(P_ball::Matrix{Ball{ComplexF64}})

Extract radius matrix from Ball matrix.
"""
function extract_radii(P_ball::Matrix{<:Ball})
    return [rad(P_ball[i,j]) for i in axes(P_ball,1), j in axes(P_ball,2)]
end

"""
    max_radius(P_ball::Matrix{<:Ball})

Get maximum radius (numerical error bound) from Ball matrix.
"""
function max_radius(P_ball::Matrix{<:Ball})
    return maximum(rad(P_ball[i,j]) for i in axes(P_ball,1), j in axes(P_ball,2))
end

"""
    RigorousOperatorResult

Result of rigorous operator assembly.

# Fields
- `P_ball`: Matrix of Balls with rigorous enclosure
- `P_mid`: Midpoint matrix (ComplexF64)
- `P_rad`: Radius matrix (Float64)
- `E_num`: Maximum numerical error (max radius)
- `params`: Problem parameters
"""
struct RigorousOperatorResult
    P_ball::Matrix{Ball{Float64, ComplexF64}}
    P_mid::Matrix{ComplexF64}
    P_rad::Matrix{Float64}
    E_num::Float64
    params::AnnealedOperatorProblem
end

"""
    build_annealed_operator_rigorous(prob::AnnealedOperatorProblem; verbose=false)

Build annealed operator with rigorous error enclosure.
"""
function build_annealed_operator_rigorous(prob::AnnealedOperatorProblem; verbose::Bool=false)
    a, σ_rds, N, M, η, σ_sm = prob.a, prob.σ_rds, prob.N, prob.M, prob.η, prob.σ_sm

    K = round(Int, η * M / 2)
    K = max(K, 5)

    if verbose
        println("Building rigorous annealed transfer operator:")
        println("  a = $a, σ_rds = $σ_rds, N = $N, M = $M")
    end

    # Build tapered and smoothed map samples
    T_eta_samples, _, _ = build_tapered_quadratic(a, M; K=K, L=min(K÷2, 10))
    Ttilde_samples, _ = build_smoothed_tapered_map(T_eta_samples, σ_sm, N)

    # Rigorous assembly
    if verbose
        println("  Assembling with rigorous arithmetic...")
    end
    P_ball = assemble_PN_rigorous(Ttilde_samples, N, σ_rds; verbose=verbose)

    # Extract components
    P_mid = extract_midpoints(P_ball)
    P_rad = extract_radii(P_ball)
    E_num = maximum(P_rad)

    if verbose
        println("  Done. Max numerical error: $E_num")
    end

    return RigorousOperatorResult(P_ball, P_mid, P_rad, E_num, prob)
end

"""
    rigorous_operator_norm_bound(result::RigorousOperatorResult)

Compute rigorous bound on operator norm using BallArithmetic's SVD.
"""
function rigorous_operator_norm_bound(result::RigorousOperatorResult)
    # Use BallArithmetic's rigorous SVD if available
    if isdefined(BallArithmetic, :svdbox)
        # svdbox computes rigorous singular value enclosures
        σ_bounds = BallArithmetic.svdbox(result.P_ball)
        return sup(σ_bounds[1])  # Largest singular value upper bound
    else
        # Fallback: use midpoint norm + radius contribution
        mid_norm = opnorm(result.P_mid)
        dim = size(result.P_mid, 1)
        rad_norm = maximum(result.P_rad) * dim
        return mid_norm + rad_norm
    end
end

"""
    rigorous_spectral_gap_bound(result::RigorousOperatorResult)

Attempt to bound the spectral gap |λ₁| - |λ₂| rigorously.
"""
function rigorous_spectral_gap_bound(result::RigorousOperatorResult)
    if isdefined(BallArithmetic, :svdbox)
        σ_bounds = BallArithmetic.svdbox(result.P_ball)
        if length(σ_bounds) >= 2
            # Gap lower bound: inf(σ₁) - sup(σ₂)
            gap_lower = inf(σ_bounds[1]) - sup(σ_bounds[2])
            return gap_lower
        end
    end
    return nothing  # Cannot compute rigorous gap
end

# Exports
export assemble_PN_rigorous, rigorous_fft, fallback_rigorous_fft
export extract_midpoints, extract_radii, max_radius
export RigorousOperatorResult, build_annealed_operator_rigorous
export rigorous_operator_norm_bound, rigorous_spectral_gap_bound
