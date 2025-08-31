module Serialization

export save_weights, load_weights

using ..autodiff.Node: Param, value

"""
    save_weights(filename, ps)

Save parameters to file (ps = Vector{Param}).
"""
function save_weights(filename::AbstractString, ps::AbstractVector{Param})
    arrs = [value(p) for p in ps]
    open(filename, "w") do io
        serialize(io, arrs)
    end
    return nothing
end

"""
    load_weights!(filename, ps)

Load weights into existing params in-place.
"""
function load_weights!(filename::AbstractString, ps::AbstractVector{Param})
    arrs = open(deserialize, filename)
    @assert length(arrs) == length(ps) "Mismatch in number of params"
    for (a, p) in zip(arrs, ps)
        copyto!(value(p), a)
    end
    return ps
end

end # module
