module OpsAD

using ..CoreAD
import Base: +, -, *, /, ^

import ..CoreAD: TrackedValue, value, grad, ensure_grad


function Base.:+(a::TrackedValue, b::TrackedValue)
    data_a = value(a)
    data_b = value(b)
    out_data = data_a + data_b

    function _backward(dout)
        grad_a = dout
        grad_b = dout
        return (grad_a, grad_b)
    end
    
    return TrackedValue(out_data, op_name=Symbol("+"), _backward_fn_data=(_backward, a, b))
end

function Base.:+(a::Number, b::TrackedValue)
    data_a = value(a)
    data_b = value(b)
    out_data = data_a + data_b

    function _backward(dout)
        return (nothing, dout)
    end
    return TrackedValue(out_data, op_name=Symbol("num+"), _backward_fn_data=(_backward, TrackedValue(a; op_name=:const), b)) # opakowujemy stałą dla spójności
end
Base.:+(a::TrackedValue, b::Number) = b + a


function Base.:*(a::TrackedValue, b::TrackedValue)
    data_a = value(a)
    data_b = value(b)
    out_data = data_a .* data_b

    function _backward(dout)
        grad_a = dout .* data_b
        grad_b = dout .* data_a
        return (grad_a, grad_b)
    end
    return TrackedValue(out_data, op_name=Symbol("*"), _backward_fn_data=(_backward, a, b))
end

function Base.:*(a::Number, b::TrackedValue)
    data_a = value(a)
    data_b = value(b)
    out_data = data_a .* data_b

    function _backward(dout)
        return (nothing, dout .* data_a)
    end
    return TrackedValue(out_data, op_name=Symbol("num*"), _backward_fn_data=(_backward, TrackedValue(a; op_name=:const), b))
end
Base.:*(a::TrackedValue, b::Number) = b * a

function Base.:^(base::TrackedValue, p::Number)
    data_base = value(base)
    out_data = data_base .^ p

    function _backward(dout)
        grad_base = dout .* (p .* (data_base .^ (p - 1)))
        return (grad_base,)
    end
    return TrackedValue(out_data, op_name=Symbol("^"), _backward_fn_data=(_backward, base))
end


function relu(x::TrackedValue)
    data_x = value(x)
    out_data = max.(0.0f0, data_x)

    function _backward(dout)
        grad_x = dout .* (data_x .> 0.0f0)
        return (grad_x,)
    end
    return TrackedValue(out_data, op_name=:relu, _backward_fn_data=(_backward, x))
end

function matmul(A::TrackedValue, x_val)
    data_A = value(A)
    data_x = value(x_val)
    out_data = data_A * data_x

    function _backward(dout)
        grad_A = dout * data_x'
        grad_x = data_A' * dout
        
        return (grad_A, isa(x_val, TrackedValue) ? grad_x : nothing)
    end
    
    input_nodes_for_op = isa(x_val, TrackedValue) ? (A, x_val) : (A, TrackedValue(x_val; op_name=:const_input))

    return TrackedValue(out_data, op_name=:matmul, _backward_fn_data=(_backward, input_nodes_for_op...))
end


function sum_elements(x::TrackedValue)
    data_x = value(x)
    out_data = sum(data_x)

    function _backward(dout)
        grad_x = fill(dout, size(data_x))
        return (grad_x,)
    end
    return TrackedValue(out_data, op_name=:sum, _backward_fn_data=(_backward, x))
end

function Base.:/(x::TrackedValue, s::Number)
    data_x = value(x)
    out_data = data_x / s

    function _backward(dout)
        grad_x = dout / s
        return (grad_x,)
    end
    return TrackedValue(out_data, op_name=Symbol("/scalar"), _backward_fn_data=(_backward, x)) # s jest stałą
end


function Base.:-(a::TrackedValue, b::TrackedValue)
    data_a = value(a)
    data_b = value(b)
    out_data = data_a - data_b

    function _backward(dout)
        grad_a = dout
        grad_b = -dout
        return (grad_a, grad_b)
    end
    return TrackedValue(out_data, op_name=Symbol("-"), _backward_fn_data=(_backward, a, b))
end

function Base.:-(a::TrackedValue)
    data_a = value(a)
    out_data = -data_a
    function _backward(dout)
        return (-dout,)
    end
    return TrackedValue(out_data, op_name=Symbol("unary-"), _backward_fn_data=(_backward, a))
end

function Base.:-(a::Number, b::TrackedValue)
    return a + (-b)
end
function Base.:-(a::TrackedValue, b::Number)
    return a + (-b)
end

function broadcast_add(matrix::TrackedValue, vector::TrackedValue)
    data_matrix = value(matrix)
    data_vector = value(vector)
    out_data = data_matrix .+ data_vector

    function _backward(dout)
        grad_matrix = dout
        grad_vector = sum(dout, dims=2)[:, 1]
        return (grad_matrix, grad_vector)
    end
    
    return TrackedValue(out_data, op_name=:broadcast_add, _backward_fn_data=(_backward, matrix, vector))
end


function Base.:-(a::TrackedValue, b::M) where M <: AbstractMatrix
    data_a = value(a)
    out_data = data_a .- b

    function _backward(dout)
        return (dout, nothing)
    end

    b_const_tv = TrackedValue(b; op_name=:const_matrix_input, _backward_fn_data=nothing)
    
    return TrackedValue(out_data, op_name=Symbol("tv-matrix"), _backward_fn_data=(_backward, a, b_const_tv))
end

function Base.:-(b::M, a::TrackedValue) where M <: AbstractMatrix
    data_a = value(a)
    
    out_data = b .- data_a

    function _backward(dout)
        return (nothing, -dout)
    end
    
    b_const_tv = TrackedValue(b; op_name=:const_matrix_input, _backward_fn_data=nothing)
    
    return TrackedValue(out_data, op_name=Symbol("matrix-tv"), _backward_fn_data=(_backward, b_const_tv, a))
end



export relu, matmul, sum_elements, broadcast_add

end # module OpsAD