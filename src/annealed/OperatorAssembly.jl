# OperatorAssembly.jl - FFT-based assembly of the annealed transfer operator matrix
#
# This module builds the finite-rank matrix P_N representing Π_N P_T̃ Π_N
# where P_T̃ is the annealed transfer operator with smoothed tapered map T̃.
#
# Matrix entry formula:
#   P_{k,ℓ} = ρ̂_σ_rds(k) · A_{k,ℓ}
# where
#   A_{k,ℓ} = (1/2) ∫_{-1}^{1} e^{-iπk T̃(y)} e^{iπℓy} dy = ĝ_k(ℓ)
# with g_k(y) = e^{-iπk T̃(y)}

"""
    rho_hat_period2(k, σ)

Fourier coefficient of the periodized Gaussian noise kernel on period-2 torus.

For period 2 with wavenumber πk:
    ρ̂_σ(k) = exp(-π²σ²k²/2)

This is the same as gaussian_multiplier_period2.
"""
function rho_hat_period2(k::Integer, σ::Real)
    return exp(-π^2 * σ^2 * k^2 / 2)
end

"""
    compute_gk_samples(Ttilde_samples, k)

Compute samples of g_k(y) = exp(-iπk T̃(y)) on the FFT grid.

# Arguments
- `Ttilde_samples`: Samples of T̃(y_j) on grid y_j = -1 + 2j/M
- `k`: Wavenumber

# Returns
- Vector of g_k(y_j) values
"""
function compute_gk_samples(Ttilde_samples::Vector{<:Real}, k::Integer)
    return exp.(-im * π * k .* Ttilde_samples)
end

"""
    compute_Akl_row(Ttilde_samples, k, N)

Compute row k of the A matrix: A_{k,ℓ} for ℓ = -N,...,N.

Uses FFT to compute Fourier coefficients of g_k(y) = exp(-iπk T̃(y)).

# Arguments
- `Ttilde_samples`: Samples of T̃(y_j)
- `k`: Row wavenumber
- `N`: Maximum column mode

# Returns
- Vector of length 2N+1 with A_{k,ℓ} for ℓ = -N,...,N
  Indexing: result[ℓ+N+1] = A_{k,ℓ}
"""
function compute_Akl_row(Ttilde_samples::Vector{<:Real}, k::Integer, N::Int)
    M = length(Ttilde_samples)

    # Compute g_k samples
    gk = compute_gk_samples(Ttilde_samples, k)

    # FFT to get Fourier coefficients
    gk_hat = fft(gk)

    # Extract modes ℓ = -N,...,N with proper normalization
    # FFT gives sum; we want the Fourier coefficient (average)
    Ak = zeros(ComplexF64, 2N + 1)

    for ℓ in -N:N
        # Map wavenumber to FFT index
        if ℓ >= 0
            fft_idx = ℓ + 1
        else
            fft_idx = M + ℓ + 1
        end

        # Map ℓ to output index
        out_idx = ℓ + N + 1

        # Normalize: FFT sum → integral average
        # FFT gives ∑_{j=0}^{M-1} f(x_j) e^{-2πi jℓ/M}
        # We want (1/2)∫_{-1}^{1} f(x) e^{-iπℓx} dx ≈ (1/M) ∑ f(x_j) e^{-iπℓx_j}
        # The FFT with our grid gives the right phases, so just divide by M
        Ak[out_idx] = gk_hat[fft_idx] / M
    end

    return Ak
end

"""
    assemble_PN_fft(Ttilde_samples, N, σ_rds; verbose=false)

Assemble the finite-rank annealed transfer operator matrix P_N.

P_N represents Π_N P_{T̃} Π_N in the Fourier basis e_k(x) = e^{iπkx}.
Matrix size is (2N+1) × (2N+1).

Indexing convention:
- Row index i corresponds to output mode k = i - N - 1 (so i=1 → k=-N, i=N+1 → k=0)
- Column index j corresponds to input mode ℓ = j - N - 1

# Arguments
- `Ttilde_samples`: Samples of smoothed tapered map T̃(y_j)
- `N`: Fourier truncation parameter
- `σ_rds`: Physical noise width (the σ in the annealed operator)
- `verbose`: Print progress

# Returns
- `P`: Complex matrix of size (2N+1) × (2N+1)
"""
function assemble_PN_fft(Ttilde_samples::Vector{<:Real}, N::Int, σ_rds::Real;
                         verbose::Bool=false)
    M = length(Ttilde_samples)
    @assert N <= M ÷ 2 "N must be ≤ M/2"

    dim = 2N + 1
    P = zeros(ComplexF64, dim, dim)

    for k in -N:N
        if verbose && (k + N) % 50 == 0
            println("  Computing row k = $k / $N")
        end

        # Compute row of A matrix
        Ak = compute_Akl_row(Ttilde_samples, k, N)

        # Multiply by noise Fourier coefficient
        rho_k = rho_hat_period2(k, σ_rds)

        # Row index in output matrix
        row_idx = k + N + 1

        # Fill row of P
        P[row_idx, :] = rho_k .* Ak
    end

    return P
end

"""
    AnnealedOperatorProblem

Parameters for building an annealed transfer operator.

# Fields
- `a`: Map parameter (for T(x) = a - (a+1)x²)
- `σ_rds`: Physical noise width
- `N`: Fourier truncation for operator
- `M`: FFT grid size
- `η`: Taper collar width
- `σ_sm`: Map smoothing width (numerical)
"""
struct AnnealedOperatorProblem
    a::Float64
    σ_rds::Float64
    N::Int
    M::Int
    η::Float64
    σ_sm::Float64
end

"""
    AnnealedOperatorProblem(; a, σ_rds, N, M=nothing, η=nothing, σ_sm=nothing, K=nothing)

Construct problem with default parameter choices.

# Default parameter heuristics:
- M = 8N (sufficient oversampling)
- K = M ÷ 16 (collar of 1/16 of domain on each side)
- η = 2K/M (physical collar width)
- σ_sm = η / 10 (smoothing width much smaller than collar)
"""
function AnnealedOperatorProblem(; a::Real, σ_rds::Real, N::Int,
                                   M::Union{Int,Nothing}=nothing,
                                   η::Union{Real,Nothing}=nothing,
                                   σ_sm::Union{Real,Nothing}=nothing,
                                   K::Union{Int,Nothing}=nothing)
    # Default M
    M_val = isnothing(M) ? 8 * N : M
    @assert iseven(M_val) "M must be even"
    @assert M_val >= 4 * N "M should be at least 4N for accurate integration"

    # Default K (collar size in grid points)
    K_val = isnothing(K) ? M_val ÷ 16 : K
    K_val = max(K_val, 5)  # At least 5 points

    # Default η
    η_val = isnothing(η) ? 2.0 * K_val / M_val : η

    # Default σ_sm
    σ_sm_val = isnothing(σ_sm) ? η_val / 10 : σ_sm

    return AnnealedOperatorProblem(Float64(a), Float64(σ_rds), N, M_val,
                                    Float64(η_val), Float64(σ_sm_val))
end

"""
    AnnealedOperatorResult

Result of building an annealed transfer operator.

# Fields
- `P`: The operator matrix (2N+1) × (2N+1)
- `Ttilde_samples`: Samples of the smoothed tapered map
- `Ttilde_hat`: Fourier coefficients of T̃
- `x_grid`: Grid points
- `params`: The problem parameters
- `map_error`: Map approximation error breakdown
"""
struct AnnealedOperatorResult
    P::Matrix{ComplexF64}
    Ttilde_samples::Vector{Float64}
    Ttilde_hat::Vector{ComplexF64}
    x_grid::Vector{Float64}
    params::AnnealedOperatorProblem
    map_error::Any  # MapApproxError
end

"""
    build_annealed_operator(prob::AnnealedOperatorProblem; verbose=false)

Build the full annealed transfer operator from problem specification.

# Returns
- AnnealedOperatorResult containing the matrix and all intermediate data
"""
function build_annealed_operator(prob::AnnealedOperatorProblem; verbose::Bool=false)
    a, σ_rds, N, M, η, σ_sm = prob.a, prob.σ_rds, prob.N, prob.M, prob.η, prob.σ_sm

    # Collar size in grid points
    K = round(Int, η * M / 2)
    K = max(K, 5)

    if verbose
        println("Building annealed transfer operator:")
        println("  a = $a, σ_rds = $σ_rds")
        println("  N = $N, M = $M")
        println("  η = $η (K = $K grid points)")
        println("  σ_sm = $σ_sm")
    end

    # Step 1: Build tapered map samples
    if verbose
        println("  Step 1: Building tapered map...")
    end
    T_eta_samples, That_eta, x_grid = build_tapered_quadratic(a, M; K=K, L=min(K÷2, 10))

    # Step 2: Apply Gaussian smoothing and truncation
    if verbose
        println("  Step 2: Smoothing and truncating...")
    end
    Ttilde_samples, Ttilde_hat = build_smoothed_tapered_map(T_eta_samples, σ_sm, N)

    # Step 3: Assemble operator matrix
    if verbose
        println("  Step 3: Assembling operator matrix...")
    end
    P = assemble_PN_fft(Ttilde_samples, N, σ_rds; verbose=verbose)

    # Compute error bounds
    map_params = MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N)
    map_error = compute_map_approx_error(map_params)

    if verbose
        println("  Done. Map approximation error: $(map_error.E_total)")
    end

    return AnnealedOperatorResult(P, Ttilde_samples, Ttilde_hat, x_grid, prob, map_error)
end

"""
    build_annealed_operator(; a, σ_rds, N, kwargs...)

Convenience function to build annealed operator from keyword arguments.
"""
function build_annealed_operator(; a::Real, σ_rds::Real, N::Int, kwargs...)
    prob = AnnealedOperatorProblem(; a=a, σ_rds=σ_rds, N=N, kwargs...)
    return build_annealed_operator(prob; kwargs...)
end

"""
    verify_markov_property(P, tol=1e-10)

Check that the operator preserves probability (column sums relate to mass).

For a Markov operator in Fourier basis, the column corresponding to ℓ=0
(constant input) should have special structure.
"""
function verify_markov_property(P::Matrix{<:Number}; tol::Real=1e-10)
    dim = size(P, 1)
    N = (dim - 1) ÷ 2

    # Column ℓ=0 is at index N+1
    col_zero = P[:, N + 1]

    # The output of applying to constant (ℓ=0) should be a probability density
    # In Fourier, this means the k=0 component should be 1 (mass preservation)
    k0_idx = N + 1
    mass = real(col_zero[k0_idx])

    mass_preserved = abs(mass - 1.0) < tol

    if !mass_preserved
        @warn "Mass not preserved: P[0,0] = $mass (expected 1.0)"
    end

    return mass_preserved, mass
end

# Exports
export rho_hat_period2
export compute_gk_samples, compute_Akl_row
export assemble_PN_fft
export AnnealedOperatorProblem, AnnealedOperatorResult
export build_annealed_operator
export verify_markov_property
