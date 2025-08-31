module Scheduler

export ExponentialDecay

"""
    ExponentialDecay(η0, γ)

Learning rate schedule: η_t = η0 * γ^t
"""
struct ExponentialDecay{T<:AbstractFloat}
    η0::T
    γ::T
end

( sch::ExponentialDecay )(t::Int) = sch.η0 * sch.γ^t

end # module
