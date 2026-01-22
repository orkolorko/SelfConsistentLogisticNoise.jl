# Building the transfer operator matrix B_a via FFT

"""
    build_B(map::QuadraticMap, disc::FourierDisc; cache=true, cache_dir=".cache",
            η=0.1, σ_sm=0.005, taper=true)

Build the matrix B_a where:
    (B_a)_{k,ℓ} = (1/2)∫_{-1}^{1} e^{-iπk T_a(y)} e^{iπℓy} dy

Uses FFT trick: for each k, compute Fourier coefficients of g_k(y) = e^{-iπk T_a(y)}.
When `taper=true`, uses the tapered + smoothed map T̃_a.
by sampling on a fine grid and taking FFT.

If `cache=true`, attempts to load from / save to disk.
"""
function build_B(map::QuadraticMap, disc::FourierDisc; cache::Bool=true, cache_dir::String=".cache",
                 η::Float64=0.1, σ_sm::Float64=0.005, taper::Bool=true)
    N, M = disc.N, disc.M
    a = map.a

    # Try to load from cache
    if cache
        cache_file = joinpath(cache_dir, "B_a$(a)_N$(N)_M$(M)_taper$(taper)_eta$(η)_sigsm$(σ_sm).jld2")
        if isfile(cache_file)
            @info "Loading B matrix from cache: $cache_file"
            return JLD2.load(cache_file, "B")
        end
    end

    B = zeros(ComplexF64, 2N + 1, 2N + 1)
    if taper
        # Build tapered + smoothed map samples
        @info "Building B matrix for a=$a, N=$N, M=$M (η=$η, σ_sm=$σ_sm)..."
        K = round(Int, η * M / 2)
        K = max(K, 5)
        T_eta_samples, _, _ = build_tapered_quadratic(a, M; K=K, L=min(K÷2, 10))
        Ttilde_samples, _ = build_smoothed_tapered_map(T_eta_samples, σ_sm, N)

        for k in modes(N)
            k_idx = idx(k, N)
            B[k_idx, :] = compute_Akl_row(Ttilde_samples, k, N)
        end
    else
        # Raw map sampling without tapering or smoothing
        @info "Building B matrix for a=$a, N=$N, M=$M (raw map, no taper)..."
        ygrid = -1 .+ 2 .* ((0:M-1) ./ M)
        Ty = map.(ygrid)

        for k in modes(N)
            gk = exp.(-π * im * k .* Ty)
            gk_fft = fft(gk) / M
            k_idx = idx(k, N)
            for ℓ in modes(N)
                ℓ_fft_idx = fft_mode_to_idx(ℓ, M)
                B[k_idx, idx(ℓ, N)] = (isodd(ℓ) ? -1.0 : 1.0) * gk_fft[ℓ_fft_idx]
            end
        end
    end

    # Save to cache
    if cache
        mkpath(cache_dir)
        cache_file = joinpath(cache_dir, "B_a$(a)_N$(N)_M$(M)_taper$(taper)_eta$(η)_sigsm$(σ_sm).jld2")
        @info "Saving B matrix to cache: $cache_file"
        JLD2.save(cache_file, "B", B)
    end

    return B
end

"""
    bound_fft_aliasing_entry_error(map::QuadraticMap, disc::FourierDisc; kmax=disc.N)

Upper bound on the aliasing error for FFT-based Fourier coefficients when
sampling with grid size `M`, using the aliasing formula and the decay
|f̂(k)| ≤ C/|k|^2 based on ||f''||_1. Here f(x) = exp(-iπ k T_a(x)).
"""
function bound_fft_aliasing_entry_error(map::QuadraticMap, disc::FourierDisc; kmax::Real=disc.N)
    M = disc.M
    if M <= 1
        return Inf
    end
    a1 = abs(map.a + 1)
    kk = abs(float(kmax))
    if kk == 0
        return 0.0
    end
    # ||F''||_1 bound with sup|F| = 1 for purely imaginary exponent.
    term1 = 4 * π * kk * a1
    term2 = (8 / 3) * π^2 * kk^2 * a1^2
    C = (term1 + term2) / (2 * π^2)
    # Alias tail: sum_{m≠0} |f̂(n+mM)| ≤ 2C/M^2 for |n| ≤ M/2.
    return 2 * C / M^2
end

"""
    bound_fft_aliasing_opnorm(map::QuadraticMap, disc::FourierDisc; kmax=disc.N)

Crude L2 operator-norm bound from entrywise aliasing error.
"""
function bound_fft_aliasing_opnorm(map::QuadraticMap, disc::FourierDisc; kmax::Real=disc.N)
    N = disc.N
    entry_err = bound_fft_aliasing_entry_error(map, disc; kmax=kmax)
    return (2N + 1) * entry_err
end

"""
    fft_aliasing_bound_matrix(map::QuadraticMap, disc::FourierDisc)

Build a matrix of entrywise aliasing bounds using a k-dependent estimate.
Each row k uses kmax = |k|.
"""
function fft_aliasing_bound_matrix(map::QuadraticMap, disc::FourierDisc)
    N = disc.N
    M = zeros(Float64, 2N + 1, 2N + 1)
    for k in modes(N)
        row_err = bound_fft_aliasing_entry_error(map, disc; kmax=abs(k))
        M[idx(k, N), :] .= row_err
    end
    return M
end

"""
    build_problem(; a=0.915, σ=0.02, N=256, δ=0.0, coupling_type=:linear,
                  β=1.0, oversample=8, η=0.1, σ_sm=0.005,
                  taper=true, M_override=nothing, cache=true)

Convenience function to build a complete SCProblem.
"""
function build_problem(;
    a::Float64=0.915,
    σ::Float64=0.02,
    N::Int=256,
    δ::Float64=0.0,
    coupling_type::Symbol=:linear,
    β::Float64=1.0,
    oversample::Int=8,
    η::Float64=0.1,
    σ_sm::Float64=0.005,
    taper::Bool=true,
    M_override::Union{Nothing,Int}=nothing,
    cache::Bool=true
)
    map = QuadraticMap(a)
    noise = GaussianNoise(σ)
    disc = isnothing(M_override) ? FourierDisc(N; oversample=oversample) : FourierDisc(N, M_override)

    coupling = if coupling_type == :linear
        LinearCoupling(δ)
    elseif coupling_type == :tanh
        TanhCoupling(δ, β)
    else
        error("Unknown coupling type: $coupling_type")
    end

    B = build_B(map, disc; cache=cache, η=η, σ_sm=σ_sm, taper=taper)

    if taper
        map_params = MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N)
        map_error = compute_map_approx_error(map_params)
        operator_error = compute_operator_error_bounds(
            a=a, η=η, σ_sm=σ_sm, N=N, σ_rds=σ, E_num=0.0
        )
    else
        map_params = nothing
        map_error = nothing
        operator_error = nothing
    end

    return SCProblem(map, noise, disc, coupling, B, map_params, map_error, operator_error)
end

export build_problem
