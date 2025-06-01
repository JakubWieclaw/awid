module CoreAD


import Base: show

mutable struct TrackedValue{T}
    data::T
    grad::Union{Nothing, T}
    _backward_fn_data::Union{Nothing, Tuple} 
    op_name::Symbol 

    function TrackedValue(data::T; op_name::Symbol=:leaf, _backward_fn_data=nothing) where T
        new{T}(data, nothing, _backward_fn_data, op_name)
    end
end


value(x::TrackedValue) = x.data
value(x::Number) = x

grad(x::TrackedValue) = x.grad
function ensure_grad(x::TrackedValue{T}) where T
    if x.grad === nothing
        x.grad = zero(x.data)
    end
end

function zero_grad!(tv::TrackedValue)
    tv.grad = nothing
end

# Backward pass
function backward!(output_node::TrackedValue, initial_gradient=1.0f0)
    visited = Set{TrackedValue}()
    nodes_topo_order = TrackedValue[]

    function build_topo(node::TrackedValue)
        if node in visited
            return
        end
        push!(visited, node)
        if node._backward_fn_data !== nothing
            input_nodes = node._backward_fn_data[2:end]
            for input_node in input_nodes
                if isa(input_node, TrackedValue)
                    build_topo(input_node)
                end
            end
        end
        push!(nodes_topo_order, node)
    end

    build_topo(output_node)

    output_node.grad = initial_gradient

    for node in reverse(nodes_topo_order)
        if node._backward_fn_data !== nothing && node.grad !== nothing
            backward_fn = node._backward_fn_data[1]
            input_nodes_for_op = node._backward_fn_data[2:end]
            
            grads_for_inputs = backward_fn(node.grad)

            for (i, input_node) in enumerate(input_nodes_for_op)
                if isa(input_node, TrackedValue)
                    ensure_grad(input_node)
                    if isa(grads_for_inputs, Tuple) && i <= length(grads_for_inputs)
                        grad_val = grads_for_inputs[i]
                        if grad_val !== nothing
                            if isa(input_node.grad, AbstractArray)
                                input_node.grad .+= grad_val
                            elseif input_node.grad !== nothing
                                input_node.grad += grad_val
                            end
                        end
                    elseif !isa(grads_for_inputs, Tuple) && i == 1
                         if grads_for_inputs !== nothing
                            if isa(input_node.grad, AbstractArray)
                                input_node.grad .+= grads_for_inputs
                            elseif input_node.grad !== nothing 
                                input_node.grad += grads_for_inputs
                            end
                         end
                    end
                end
            end
        end
    end
end


function Base.show(io::IO, tv::TrackedValue)
    print(io, "TrackedValue(data=$(tv.data), grad=$(tv.grad === nothing ? "nothing" : tv.grad), op=$(tv.op_name))")
end

export TrackedValue, value, grad, ensure_grad, zero_grad!, backward!
end