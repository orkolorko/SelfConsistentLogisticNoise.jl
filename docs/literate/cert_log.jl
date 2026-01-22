using Printf
using JLD2

function write_cert_log(path; title, key_label, key_format=string, config, records)
    open(path, "w") do io
        println(io, "# ", title)
        println(io, "")
        println(io, "## Configuration")
        println(io, "```")
        for (k, v) in sort(collect(config); by=kv -> string(kv[1]))
            println(io, "$(k): $(v)")
        end
        println(io, "```")
        println(io, "")
        println(io, "## Results")
        println(io, "| $(key_label) | verified | N | total_error | finite_error | map_shift | fft_error | trunc_error |")
        println(io, "|---|---|---|---|---|---|---|---|")
        for key in sort(collect(keys(records)))
            rec = records[key]
            verified = get(rec, :verified, false)
            used_N = get(rec, :used_N, nothing)
            cap = get(rec, :cap, nothing)
            total_error = get(rec, :total_error, nothing)
            finite_error = cap === nothing ? nothing : cap.krawczyk.r
            map_shift = cap === nothing ? nothing : cap.map_shift_bound
            fft_error = cap === nothing ? nothing : cap.fft_error
            trunc_error = cap === nothing ? nothing : cap.truncation_error
            println(io, "| $(key_format(key)) | $(verified) | $(fmt_int(used_N)) | $(fmt_num(total_error)) | $(fmt_num(finite_error)) | $(fmt_num(map_shift)) | $(fmt_num(fft_error)) | $(fmt_num(trunc_error)) |")
        end
    end
end

function fmt_num(x)
    if x === nothing
        return "NA"
    end
    return @sprintf("%.3e", x)
end

function fmt_int(x)
    if x === nothing
        return "NA"
    end
    return string(x)
end

function write_latex_summary(path; title, key_label, key_values, key_format, records,
                             fig_path, fig_caption, fig_label)
    open(path, "w") do io
        println(io, "% Auto-generated summary snippet")
        println(io, "\\begin{table}[t]")
        println(io, "\\centering")
        println(io, "\\caption{$(title)}")
        println(io, "\\label{tab:$(fig_label)_table}")
        println(io, "\\begin{tabular}{lrrrrr}")
        println(io, "\\toprule")
        println(io, "$(key_label) & N & total & finite & fft & trunc \\\\")
        println(io, "\\midrule")
        for key in key_values
            rec = get(records, key, nothing)
            if rec === nothing
                println(io, "$(key_format(key)) & -- & -- & -- & -- & -- \\\\")
                continue
            end
            cap = get(rec, :cap, nothing)
            used_N = get(rec, :used_N, nothing)
            total_error = get(rec, :total_error, nothing)
            finite_error = cap === nothing ? nothing : cap.krawczyk.r
            fft_error = cap === nothing ? nothing : cap.fft_error
            trunc_error = cap === nothing ? nothing : cap.truncation_error
            println(io, "$(key_format(key)) & $(fmt_int(used_N)) & $(fmt_tex(total_error)) & $(fmt_tex(finite_error)) & $(fmt_tex(fft_error)) & $(fmt_tex(trunc_error)) \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")
        println(io, "")
        println(io, "\\begin{figure}[t]")
        println(io, "\\centering")
        println(io, "\\includegraphics[width=0.85\\linewidth]{$(fig_path)}")
        println(io, "\\caption{$(fig_caption)}")
        println(io, "\\label{fig:$(fig_label)}")
        println(io, "\\end{figure}")
    end
end

function fmt_tex(x)
    if x === nothing
        return "--"
    end
    return @sprintf("%.2e", x)
end

function load_state_jld2(path, config)
    if !isfile(path)
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    state = try
        JLD2.load(path)
    catch err
        println("JLD2 load failed ($(err)); ignoring saved state.")
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    if get(state, "config", nothing) != config
        println("JLD2 config mismatch; ignoring saved state.")
        return Dict{Float64, NamedTuple}(), Dict{Float64, NamedTuple}()
    end
    verified = get(state, "verified", Dict{Float64, NamedTuple}())
    records = get(state, "records", Dict{Float64, NamedTuple}())
    return verified, records
end

function save_state_jld2(path, config, verified, records)
    mkpath(dirname(path))
    JLD2.save(path, "config", config, "verified", verified, "records", records, "timestamp", time())
end
