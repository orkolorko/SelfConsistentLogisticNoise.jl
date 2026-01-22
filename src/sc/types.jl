# Core types for self-consistent quadratic-noise computations

"""
    QuadraticMap(a)

Quadratic map T_a(x) = a - (a+1)x^2 on the period-2 torus [-1,1].
"""
struct QuadraticMap
    a::Float64
end

(T::QuadraticMap)(x) = T.a - (T.a + 1) * x^2
derivative(T::QuadraticMap, x) = -2 * (T.a + 1) * x

"""
    GaussianNoise(σ)

Periodized Gaussian noise kernel on the period-2 torus with standard deviation σ.
Fourier coefficients: ρ̂_σ(k) = exp(-(π²/2)σ²k²)
"""
struct GaussianNoise
    σ::Float64
end

"""
    rhohat(noise::GaussianNoise, k)

Fourier coefficient of the periodized Gaussian kernel at mode k.
"""
rhohat(noise::GaussianNoise, k::Int) = exp(-0.5 * π^2 * noise.σ^2 * k^2)

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
    map::QuadraticMap
    noise::GaussianNoise
    disc::FourierDisc
    coupling::Any  # Coupling type
    B::Matrix{ComplexF64}  # Precomputed transfer operator matrix
    map_params::Any
    map_error::Any
    operator_error::Any
end
