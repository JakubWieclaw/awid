module Models

export Sequential

"""
    Sequential(layers...)

Simple sequential container. Applies layers in order.
"""
struct Sequential
    layers::Tuple
end

function (m::Sequential)(x)
    for l in m.layers
        x = l(x)
    end
    return x
end

end # module
