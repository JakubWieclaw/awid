using ..AutoDiff
import ..AutoDiff: TrackedValue, value, matmul, relu, broadcast_add
import Random

params(x) = TrackedValue[]

abstract type Layer end

# Dense layer
struct Dense <: Layer
    W::TrackedValue
    b::TrackedValue

    function Dense(in_dim::Int, out_dim::Int)
        W_data = randn(Float32, out_dim, in_dim) .* sqrt(2.0f0 / in_dim)
        b_data = zeros(Float32, out_dim, 1)  # Make bias a column vector for broadcasting
        
        new(TrackedValue(W_data; op_name=:W), TrackedValue(b_data; op_name=:b))
    end
end

# Dense layer forward pass
function (layer::Dense)(x_input)
    x = isa(x_input, TrackedValue) ? x_input : TrackedValue(x_input; op_name=:data_input)
    
    # Matrix multiplication: W * x
    linear_output = matmul(layer.W, x)
    
    # Add bias - simple addition should work if bias is right shape
    biased_output = broadcast_add(linear_output, layer.b)
    
    return biased_output
end

# Parameters for Dense layer
params(layer::Dense) = [layer.W, layer.b]

# ReLU activation
struct ReLU <: Layer end

function (activation::ReLU)(x_input)
    x = isa(x_input, TrackedValue) ? x_input : TrackedValue(x_input; op_name=:relu_input)
    return relu(x)
end

# ReLU has no parameters
params(::ReLU) = TrackedValue[]

# Chain of layers
struct Chain <: Layer
    layers::Vector{Any}
end
Chain(layers...) = Chain([layers...])

# Chain forward pass
function (c::Chain)(x_input)
    current_output = x_input
    for layer in c.layers
        current_output = layer(current_output)
    end
    return current_output
end

# Chain parameters: collect from all layers
params(c::Chain) = vcat([params(layer) for layer in c.layers]...)

# Export everything
export Dense, ReLU, Chain, params