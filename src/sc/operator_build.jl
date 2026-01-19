# Building the transfer operator matrix B_a via FFT

"""
    build_B(map::LogisticMap, disc::FourierDisc; cache=true, cache_dir=".cache")

Build the matrix B_a where:
    (B_a)_{k,ℓ} = ∫_𝕋 e^{-2πik T_a(y)} e^{2πiℓy} dy

Uses FFT trick: for each k, compute Fourier coefficients of g_k(y) = e^{-2πik T_a(y)}
by sampling on a fine grid and taking FFT.

If `cache=true`, attempts to load from / save to disk.
"""
function build_B(map::LogisticMap, disc::FourierDisc; cache::Bool=true, cache_dir::String=".cache")
    N, M = disc.N, disc.M
    a = map.a

    # Try to load from cache
    if cache
        cache_file = joinpath(cache_dir, "B_a$(a)_N$(N)_M$(M).jld2")
        if isfile(cache_file)
            @info "Loading B matrix from cache: $cache_file"
            return JLD2.load(cache_file, "B")
        end
    end

    # Build B matrix
    @info "Building B matrix for a=$a, N=$N, M=$M..."

    B = zeros(ComplexF64, 2N + 1, 2N + 1)

    # Grid points y_j = j/M for j = 0, ..., M-1
    ygrid = (0:M-1) ./ M

    # Evaluate T_a at grid points
    Ty = map.(ygrid)

    # For each output mode k ∈ {-N, ..., N}
    for k in modes(N)
        # g_k(y) = e^{-2πik T_a(y)}
        gk = exp.(-2π * im * k .* Ty)

        # FFT to get Fourier coefficients
        # Julia's fft uses convention: FFT[j] = Σ_n x[n] * e^{-2πi(j-1)(n-1)/M}
        # We want: ĝ(ℓ) = (1/M) Σ_{j=0}^{M-1} g(y_j) e^{-2πiℓ y_j}
        #                = (1/M) Σ_{j=0}^{M-1} g[j+1] e^{-2πiℓj/M}
        gk_fft = fft(gk) / M

        # Extract coefficients for ℓ ∈ {-N, ..., N}
        k_idx = idx(k, N)
        for ℓ in modes(N)
            ℓ_fft_idx = fft_mode_to_idx(ℓ, M)
            B[k_idx, idx(ℓ, N)] = gk_fft[ℓ_fft_idx]
        end
    end

    # Save to cache
    if cache
        mkpath(cache_dir)
        cache_file = joinpath(cache_dir, "B_a$(a)_N$(N)_M$(M).jld2")
        @info "Saving B matrix to cache: $cache_file"
        JLD2.save(cache_file, "B", B)
    end

    return B
end

"""
    build_problem(; a=3.83, σ=0.02, N=256, δ=0.0, coupling_type=:linear, β=1.0, oversample=8, cache=true)

Convenience function to build a complete SCProblem.
"""
function build_problem(;
    a::Float64=3.83,
    σ::Float64=0.02,
    N::Int=256,
    δ::Float64=0.0,
    coupling_type::Symbol=:linear,
    β::Float64=1.0,
    oversample::Int=8,
    cache::Bool=true
)
    map = LogisticMap(a)
    noise = GaussianNoise(σ)
    disc = FourierDisc(N; oversample=oversample)

    coupling = if coupling_type == :linear
        LinearCoupling(δ)
    elseif coupling_type == :tanh
        TanhCoupling(δ, β)
    else
        error("Unknown coupling type: $coupling_type")
    end

    B = build_B(map, disc; cache=cache)

    return SCProblem(map, noise, disc, coupling, B)
end

export build_problem
