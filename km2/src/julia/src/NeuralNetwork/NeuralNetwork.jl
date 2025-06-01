module NeuralNetwork

using ..AutoDiff # Dostęp do TrackedValue, backward!, ops etc.
# Jeśli operacje z AutoDiff.ops nie są eksportowane globalnie, trzeba je zaimportować:
# using ..AutoDiff.OpsAD: relu, matmul, +, * # etc. lub using ..AutoDiff: relu, matmul
# Załóżmy, że są dostępne przez AutoDiff.relu_op, AutoDiff.matmul_op lub przeciążone operatory

# println(" własna biblioteka NeuralNetwork załadowana! 🧠")

# Definicje warstw
include("layers.jl")
export Dense, Chain, ReLU # ReLU jako warstwa/obiekt aktywacji

# Funkcje kosztu
include("loss.jl")
export mse_loss

# Optymalizatory
include("optimizers.jl")
export SGD, update! # update! to funkcja do aktualizacji wag

# Logika treningu
include("train.jl")
export train!

# Funkcja pomocnicza do zbierania parametrów modelu (TrackedValue)
function params(layer) # Generyczna funkcja, specjalizacje w layers.jl
    # Domyślnie warstwa nie ma parametrów
    return []
end

function params(model_or_layer_array::Union{Chain, AbstractArray})
    all_params = TrackedValue[]
    for layer in (isa(model_or_layer_array, Chain) ? model_or_layer_array.layers : model_or_layer_array)
        append!(all_params, params(layer))
    end
    return all_params
end

# Funkcja do zerowania gradientów wszystkich parametrów w modelu
function zero_grad_model!(model_or_layer_collection)
    ps = params(model_or_layer_collection)
    for p in ps
        zero_grad!(p) # Używa zero_grad! z AutoDiff.core
    end
end


end # module NeuralNetwork