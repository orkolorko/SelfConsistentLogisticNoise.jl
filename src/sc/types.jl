# Core types for self-consistent logistic noise computations

"""
    LogisticMap(a)

Logistic map T_a(x) = a*x*(1-x) on the torus.
"""
struct LogisticMap
    a::Float64
end

(T::LogisticMap)(x) = T.a * x * (1 - x)

"""
    GaussianNoise(σ)

Periodized Gaussian noise kernel on the torus with standard deviation σ.
Fourier coefficients: ρ̂_σ(k) = exp(-2π²σ²k²)
"""
struct GaussianNoise
    σ::Float64
end

"""
    rhohat(noise::GaussianNoise, k)

Fourier coefficient of the periodized Gaussian kernel at mode k.
"""
rhohat(noise::GaussianNoise, k::Int) = exp(-2 * π^2 * noise.σ^2 * k^2)

"""
    FourierDisc(N, M)

Fourier discretization parameters.
- N: truncation level (modes from -N to N, total 2N+1 coefficients)
- M: FFT oversampling grid size (typically M = os * (2N+1) with os = 8 or 16)
"""
struct FourierDisc
    N::Int
    M::Int

    function FourierDisc(N::Int, M::Int)
        M >= 2N + 1 || error("M must be at least 2N+1")
        new(N, M)
    end
end

"""
    FourierDisc(N; oversample=8)

Create FourierDisc with automatic oversampling.
"""
function FourierDisc(N::Int; oversample::Int=8)
    M = oversample * (2N + 1)
    # Round to a nice FFT length (power of 2)
    M = nextpow(2, M)
    FourierDisc(N, M)
end

"""
    SCProblem

Self-consistent problem specification containing all parameters
and the precomputed B matrix.
"""
mutable struct SCProblem
    map::LogisticMap
    noise::GaussianNoise
    disc::FourierDisc
    coupling::Any  # Coupling type
    B::Matrix{ComplexF64}  # Precomputed transfer operator matrix
end
