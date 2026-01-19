# I/O utilities for saving and loading results

"""
    save_result(filename::String, result::FixedPointResult)

Save a FixedPointResult to a JLD2 file.
"""
function save_result(filename::String, result::FixedPointResult)
    JLD2.save(filename,
        "converged", result.converged,
        "fhat", result.fhat,
        "m", result.m,
        "c", result.c,
        "residual", result.residual,
        "iterations", result.iterations,
        "params", result.params
    )
end

"""
    load_result(filename::String)

Load a FixedPointResult from a JLD2 file.
"""
function load_result(filename::String)
    data = JLD2.load(filename)
    return FixedPointResult(
        data["converged"],
        data["fhat"],
        data["m"],
        data["c"],
        data["residual"],
        data["iterations"],
        data["params"]
    )
end

"""
    save_sweep(filename::String, df::DataFrame)

Save sweep results to CSV.
"""
function save_sweep(filename::String, df::DataFrame)
    CSV.write(filename, df)
end

"""
    load_sweep(filename::String)

Load sweep results from CSV.
"""
function load_sweep(filename::String)
    return CSV.read(filename, DataFrame)
end

"""
    save_fhat(filename::String, fhat::Vector{ComplexF64}; params=nothing)

Save Fourier coefficients and optional parameters.
"""
function save_fhat(filename::String, fhat::Vector{ComplexF64}; params=nothing)
    if params === nothing
        JLD2.save(filename, "fhat", fhat)
    else
        JLD2.save(filename, "fhat", fhat, "params", params)
    end
end

"""
    load_fhat(filename::String)

Load Fourier coefficients from file.
"""
function load_fhat(filename::String)
    return JLD2.load(filename, "fhat")
end

export save_sweep, load_sweep, save_fhat, load_fhat
