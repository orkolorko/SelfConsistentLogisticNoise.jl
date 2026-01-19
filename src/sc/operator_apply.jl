# Applying the noisy transfer operator P_c

"""
    apply_Pc(prob::SCProblem, fhat::Vector{ComplexF64}, c::Float64)

Apply the noisy transfer operator with shift c:
    (P_c f)̂_k = ρ̂_σ(k) * e^{-2πikc} * (B_a f̂)_k

where ρ̂_σ(k) = exp(-2π²σ²k²) is the Gaussian kernel.
"""
function apply_Pc(prob::SCProblem, fhat::Vector{ComplexF64}, c::Float64)
    N = prob.disc.N

    # Step 1: Apply B matrix
    tmp = prob.B * fhat

    # Step 2: Multiply by Gaussian damping and shift phase
    result = similar(tmp)
    for k in modes(N)
        k_idx = idx(k, N)
        gauss = rhohat(prob.noise, k)
        phase = exp(-2π * im * k * c)
        result[k_idx] = gauss * phase * tmp[k_idx]
    end

    return result
end

"""
    apply_Pc!(result, prob::SCProblem, fhat::Vector{ComplexF64}, c::Float64)

In-place version of apply_Pc.
"""
function apply_Pc!(result::Vector{ComplexF64}, prob::SCProblem, fhat::Vector{ComplexF64}, c::Float64)
    N = prob.disc.N

    # Step 1: Apply B matrix
    mul!(result, prob.B, fhat)

    # Step 2: Multiply by Gaussian damping and shift phase (in-place)
    for k in modes(N)
        k_idx = idx(k, N)
        gauss = rhohat(prob.noise, k)
        phase = exp(-2π * im * k * c)
        result[k_idx] *= gauss * phase
    end

    return result
end

"""
    apply_self_consistent(prob::SCProblem, fhat::Vector{ComplexF64})

Apply the self-consistent operator 𝒯(f) = P_{c(f)} f.
"""
function apply_self_consistent(prob::SCProblem, fhat::Vector{ComplexF64})
    N = prob.disc.N
    c = compute_shift(prob.coupling, fhat, N)
    return apply_Pc(prob, fhat, c)
end

"""
    reconstruct_density(fhat::Vector{ComplexF64}, xgrid::AbstractVector)

Reconstruct density f(x) from Fourier coefficients on the given grid.
"""
function reconstruct_density(fhat::Vector{ComplexF64}, xgrid::AbstractVector)
    N = (length(fhat) - 1) ÷ 2
    f = zeros(Float64, length(xgrid))

    for (i, x) in enumerate(xgrid)
        val = zero(ComplexF64)
        for k in modes(N)
            val += fhat[idx(k, N)] * exp(2π * im * k * x)
        end
        f[i] = real(val)
    end

    return f
end

"""
    reconstruct_density(fhat::Vector{ComplexF64}; npts=1000)

Reconstruct density on a uniform grid with npts points.
"""
function reconstruct_density(fhat::Vector{ComplexF64}; npts::Int=1000)
    xgrid = range(0, 1, length=npts)
    return collect(xgrid), reconstruct_density(fhat, xgrid)
end
