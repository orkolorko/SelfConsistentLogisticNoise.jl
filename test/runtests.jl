using Test
using SelfConsistentLogisticNoise
using LinearAlgebra

@testset "SelfConsistentLogisticNoise.jl" begin

    @testset "Fourier indexing" begin
        N = 10
        @test modes(N) == -10:10
        @test idx(0, N) == N + 1
        @test idx(-N, N) == 1
        @test idx(N, N) == 2N + 1
        @test mode(1, N) == -N
        @test mode(N + 1, N) == 0
        @test mode(2N + 1, N) == N

        # Round-trip
        for k in modes(N)
            @test mode(idx(k, N), N) == k
        end
    end

    @testset "Types" begin
        T = LogisticMap(3.83)
        @test T(0.5) ≈ 3.83 * 0.5 * 0.5

        noise = GaussianNoise(0.02)
        @test rhohat(noise, 0) ≈ 1.0
        @test rhohat(noise, 1) < 1.0

        disc = FourierDisc(16)
        @test disc.N == 16
        @test disc.M >= 2 * 16 + 1
    end

    @testset "Observable" begin
        N = 10
        fhat = zeros(ComplexF64, 2N + 1)
        fhat[idx(0, N)] = 1.0
        fhat[idx(1, N)] = 0.3 + 0.1im
        fhat[idx(-1, N)] = 0.3 - 0.1im  # conjugate symmetry

        obs = CosineObservable()
        m = compute_m(obs, fhat, N)
        @test m ≈ 0.3  # Re(f̂₁)
    end

    @testset "Coupling" begin
        c = LinearCoupling(0.5)
        @test c(0.3) ≈ 0.15

        c2 = TanhCoupling(0.5, 2.0)
        @test c2(0.0) ≈ 0.0
        @test abs(c2(10.0)) < 0.5 + 0.01  # saturates
    end

    @testset "δ=0 sanity check" begin
        # With δ=0, solver should converge to invariant measure of noisy logistic map
        prob = build_problem(a=3.83, σ=0.02, N=32, δ=0.0, cache=false)
        result = solve_fixed_point(prob; α=0.3, tol=1e-10, maxit=2000)

        @test result.converged
        @test result.residual < 1e-9

        # Normalization check: f̂₀ = 1
        N = prob.disc.N
        @test abs(result.fhat[idx(0, N)] - 1.0) < 1e-12

        # Real density check
        x, f = reconstruct_density(result.fhat; npts=100)
        @test all(isreal, f) || maximum(abs.(imag.(f))) < 1e-10
    end

    @testset "Density reconstruction" begin
        N = 10
        fhat = zeros(ComplexF64, 2N + 1)
        fhat[idx(0, N)] = 1.0  # uniform density

        x, f = reconstruct_density(fhat; npts=50)
        @test length(x) == 50
        @test length(f) == 50
        @test all(f .≈ 1.0)  # uniform should give constant 1
    end

    @testset "Grid convergence" begin
        # m should stabilize as N increases
        m_values = Float64[]
        for N in [16, 32, 64]
            prob = build_problem(a=3.83, σ=0.02, N=N, δ=0.0, cache=false)
            result = solve_fixed_point(prob; α=0.3, tol=1e-10, maxit=2000)
            push!(m_values, result.m)
        end

        # Check that m values are converging (differences should decrease)
        @test abs(m_values[3] - m_values[2]) < abs(m_values[2] - m_values[1])
    end

end
