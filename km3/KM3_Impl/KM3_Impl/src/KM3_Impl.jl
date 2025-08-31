module KM3_Impl

# Exports for library users
export Param, value, grad, zero_grads!, start_tape!, stop_tape!, back!
export Dense, Embedding, params, relu, crossentropy
export Adam, step!, train!

# Autodiff
include("autodiff/AutoDiff.jl")
using .AutoDiff

# Layers & activations
include("nn/Layers.jl")
include("nn/Activations.jl")
include("nn/Losses.jl")
using .Layers, .Activations, .Losses

# Optimizers
include("optim/Optimizers.jl")
using .Optimizers

# Training
include("training/Trainer.jl")
using .Trainer

end # module
