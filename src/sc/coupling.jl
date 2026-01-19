# Coupling functions G and shift computation c(f) = δ * G(m(f))

"""
    Coupling

Abstract type for coupling specifications.
"""
abstract type Coupling end

"""
    LinearCoupling(δ, observable)

Linear coupling: G(m) = m, so c(f) = δ * m(f).
"""
struct LinearCoupling <: Coupling
    δ::Float64
    observable::Observable
end

LinearCoupling(δ::Float64) = LinearCoupling(δ, CosineObservable())

(c::LinearCoupling)(m::Real) = c.δ * m
derivative(c::LinearCoupling, m::Real) = c.δ

"""
    TanhCoupling(δ, β, observable)

Saturating coupling: G(m) = tanh(β*m), so c(f) = δ * tanh(β * m(f)).
"""
struct TanhCoupling <: Coupling
    δ::Float64
    β::Float64
    observable::Observable
end

TanhCoupling(δ::Float64, β::Float64) = TanhCoupling(δ, β, CosineObservable())

(c::TanhCoupling)(m::Real) = c.δ * tanh(c.β * m)
derivative(c::TanhCoupling, m::Real) = c.δ * c.β * sech(c.β * m)^2

"""
    compute_shift(coupling::Coupling, fhat, N)

Compute the shift c(f) = δ * G(m(f)).
"""
function compute_shift(coupling::Coupling, fhat::Vector{ComplexF64}, N::Int)
    m = compute_m(coupling.observable, fhat, N)
    return coupling(m)
end

"""
    get_observable(coupling::Coupling)

Get the observable associated with the coupling.
"""
get_observable(c::LinearCoupling) = c.observable
get_observable(c::TanhCoupling) = c.observable

"""
    get_delta(coupling::Coupling)

Get the coupling strength δ.
"""
get_delta(c::LinearCoupling) = c.δ
get_delta(c::TanhCoupling) = c.δ
