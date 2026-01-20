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

    # =========================================================================
    # Annealed Transfer Operator Tests
    # =========================================================================

    @testset "Taper - Smoothstep functions" begin
        # Test boundary conditions
        @test smoothstep_quintic(0.0) ≈ 0.0
        @test smoothstep_quintic(1.0) ≈ 1.0
        @test smoothstep_quintic(0.5) ≈ 0.5  # Symmetry

        @test smoothstep_septic(0.0) ≈ 0.0
        @test smoothstep_septic(1.0) ≈ 1.0

        # Generic smoothstep
        @test smoothstep(0.5; order=2) ≈ smoothstep_quintic(0.5)
        @test smoothstep(0.5; order=3) ≈ smoothstep_septic(0.5)

        # Taper window
        η = 0.1
        @test taper_window(0.0, η) ≈ 1.0  # Interior
        @test taper_window(-0.95, η) < 1.0  # Left collar
        @test taper_window(0.95, η) < 1.0   # Right collar
    end

    @testset "Taper - Quadratic map" begin
        a = 1.5
        T = QuadraticMap(a)

        @test T(0.0) ≈ a
        @test T(1.0) ≈ -1.0
        @test T(-1.0) ≈ -1.0

        # Derivative
        @test SelfConsistentLogisticNoise.derivative(T, 0.0) ≈ 0.0
        @test SelfConsistentLogisticNoise.derivative(T, 0.5) ≈ -2 * (a + 1) * 0.5
    end

    @testset "Taper - Build tapered map" begin
        a = 1.5
        M = 256
        K = 16

        T_eta, That_eta, x_grid = build_tapered_quadratic(a, M; K=K)

        @test length(T_eta) == M
        @test length(x_grid) == M

        # Grid should span [-1, 1)
        @test x_grid[1] ≈ -1.0
        @test x_grid[end] < 1.0

        # Interior values should match original map
        T = QuadraticMap(a)
        interior_idx = M ÷ 4 : 3M ÷ 4
        for j in interior_idx
            @test abs(T_eta[j] - T(x_grid[j])) < 0.1  # Approximate match
        end
    end

    @testset "Smoothing - Gaussian multiplier" begin
        σ = 0.1

        # Zero mode unchanged
        @test gaussian_multiplier_period2(0, σ) ≈ 1.0

        # Higher modes decay
        @test gaussian_multiplier_period2(1, σ) < 1.0
        @test gaussian_multiplier_period2(10, σ) < gaussian_multiplier_period2(1, σ)

        # Formula check
        k = 5
        expected = exp(-π^2 * σ^2 * k^2 / 2)
        @test gaussian_multiplier_period2(k, σ) ≈ expected
    end

    @testset "Smoothing - Truncation" begin
        M = 64
        N = 10

        # Create test coefficients
        That = ones(ComplexF64, M)

        # Truncate
        That_trunc = truncate_fourier_coeffs(That, N)

        # Check that high modes are zero
        for j in 1:M
            k = j <= M ÷ 2 + 1 ? j - 1 : j - 1 - M
            if abs(k) > N
                @test That_trunc[j] == 0
            else
                @test That_trunc[j] == 1
            end
        end
    end

    @testset "MapApproxBounds - Error bounds" begin
        a = 1.5
        η = 0.05
        σ_sm = 0.01
        N = 64

        params = MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N)

        # All bounds should be positive
        @test bound_taper_error(params) > 0
        @test bound_smoothing_error(params) > 0
        @test bound_truncation_error(params) > 0

        # Total error should be sum
        err = compute_map_approx_error(params)
        @test err.E_total ≈ err.E_taper + err.E_smooth + err.E_trunc

        # Taper error should scale with η
        params2 = MapApproxParams(a=a, η=η/2, σ_sm=σ_sm, N=N)
        @test bound_taper_error(params2) < bound_taper_error(params)
    end

    @testset "OperatorAssembly - rho_hat" begin
        σ = 0.1

        # Same as Gaussian multiplier for period 2
        @test rho_hat_period2(0, σ) ≈ 1.0
        @test rho_hat_period2(5, σ) ≈ gaussian_multiplier_period2(5, σ)
    end

    @testset "OperatorAssembly - Build operator" begin
        # Small test case
        a = 1.5
        σ_rds = 0.1
        N = 8
        M = 64

        prob = AnnealedOperatorProblem(a=a, σ_rds=σ_rds, N=N, M=M)
        result = build_annealed_operator(prob)

        # Check dimensions
        dim = 2N + 1
        @test size(result.P) == (dim, dim)

        # Check Markov property (mass preservation)
        mass_ok, mass = verify_markov_property(result.P)
        @test abs(mass - 1.0) < 0.01  # Should be close to 1

        # Check that operator norm is bounded
        op_norm = opnorm(result.P)
        @test op_norm < 2.0  # Reasonable bound for transfer operator
    end

    @testset "OperatorBounds - Error bounds" begin
        a = 1.5
        η = 0.05
        σ_sm = 0.01
        N = 64
        σ_rds = 0.1

        err = compute_operator_error_bounds(
            a=a, η=η, σ_sm=σ_sm, N=N, σ_rds=σ_rds
        )

        # All bounds should be positive
        @test err.E_map > 0
        @test err.E_proj > 0

        # Total should be sum
        @test err.E_total ≈ err.E_map + err.E_proj + err.E_num

        # Operator sensitivity bound
        map_err = bound_map_sup_error(MapApproxParams(a=a, η=η, σ_sm=σ_sm, N=N))
        expected_E_map = sqrt(2/π) / σ_rds * map_err
        @test err.E_map ≈ expected_E_map
    end

    @testset "OperatorAssembly - Stability" begin
        # Check that increasing N improves approximation
        a = 1.5
        σ_rds = 0.1

        errors = Float64[]
        for N in [8, 16, 32]
            prob = AnnealedOperatorProblem(a=a, σ_rds=σ_rds, N=N)
            result = build_annealed_operator(prob)
            push!(errors, result.map_error.E_total)
        end

        # Errors should decrease with N
        @test errors[2] < errors[1]
        @test errors[3] < errors[2]
    end

end
