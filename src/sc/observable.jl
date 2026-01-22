# Observables for computing m(f) = ⟨Φ, f⟩

"""
    Observable

Abstract type for observables Φ ∈ L²(𝕋).
"""
abstract type Observable end

"""
    CosineObservable()

Default observable Φ(x) = cos(πx).
Fourier coefficients: Φ̂₁ = Φ̂₋₁ = 1/2, others = 0.

For this observable: m(f) = Re(f̂₁)
"""
struct CosineObservable <: Observable end

"""
    compute_m(obs::CosineObservable, fhat, N)

Compute m(f) = ⟨Φ, f⟩ for the cosine observable.
For Φ(x) = cos(πx): m(f) = Re(f̂₁)
"""
function compute_m(obs::CosineObservable, fhat::Vector{ComplexF64}, N::Int)
    # Mode k=1 is at index idx(1, N) = N + 2
    return real(fhat[idx(1, N)])
end

"""
    SineObservable()

Observable Φ(x) = sin(πx).
For this observable: m(f) = -Im(f̂₁)
"""
struct SineObservable <: Observable end

function compute_m(obs::SineObservable, fhat::Vector{ComplexF64}, N::Int)
    return -imag(fhat[idx(1, N)])
end

"""
    FourierObservable(Phihat)

General observable specified by its Fourier coefficients.
"""
struct FourierObservable <: Observable
    Phihat::Vector{ComplexF64}
end

function compute_m(obs::FourierObservable, fhat::Vector{ComplexF64}, N::Int)
    # m(f) = ⟨Φ, f⟩ = Σ_k Φ̂_k * conj(f̂_k) (for L² inner product)
    # But for densities, we want ∫ Φ(x) f(x) dx = Σ_k Φ̂_k * f̂_{-k}
    # With conjugate symmetry f̂_{-k} = conj(f̂_k), this gives Σ_k Φ̂_k * conj(f̂_k)
    result = zero(ComplexF64)
    N_phi = (length(obs.Phihat) - 1) ÷ 2
    for k in modes(min(N, N_phi))
        result += obs.Phihat[idx(k, N_phi)] * conj(fhat[idx(k, N)])
    end
    return real(result)
end
