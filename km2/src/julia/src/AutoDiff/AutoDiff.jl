module AutoDiff

include("core.jl")
using .CoreAD

export TrackedValue, value, grad, backward!, zero_grad!, ensure_grad

include("ops.jl")
using .OpsAD
export relu, matmul, sum_elements, broadcast_add


println("✅ AutoDiff Loaded ⚙️")

end