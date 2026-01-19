# Fourier indexing utilities for truncated Fourier series
# Densities are represented as: f(x) ≈ Σ_{k=-N}^{N} f̂_k e^{2πikx}
# Stored as a length L = 2N+1 vector

"""
    modes(N)

Return the range of Fourier modes: -N:N
"""
modes(N::Int) = -N:N

"""
    idx(k, N)

Convert Fourier mode k ∈ {-N,...,N} to 1-based array index.
Mode k maps to index k + N + 1.
"""
idx(k::Int, N::Int) = k + N + 1

"""
    mode(i, N)

Convert 1-based array index i to Fourier mode k.
Index i maps to mode i - (N+1).
"""
mode(i::Int, N::Int) = i - (N + 1)

"""
    fft_mode_to_idx(ℓ, M)

Convert FFT output mode ℓ ∈ {-M/2,...,M/2-1} to 1-based FFT array index.
For Julia's FFT convention:
- mode ℓ ≥ 0 corresponds to index ℓ + 1
- mode ℓ < 0 corresponds to index M + ℓ + 1
"""
function fft_mode_to_idx(ℓ::Int, M::Int)
    return ℓ >= 0 ? (ℓ + 1) : (M + ℓ + 1)
end

"""
    fft_idx_to_mode(i, M)

Convert 1-based FFT array index to FFT mode.
"""
function fft_idx_to_mode(i::Int, M::Int)
    return i <= M ÷ 2 + 1 ? (i - 1) : (i - 1 - M)
end
