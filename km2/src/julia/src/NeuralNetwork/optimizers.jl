# optimizers.jl
using ..AutoDiff # Dostęp do TrackedValue, grad, value

# Abstrakcyjny typ optymalizatora
abstract type Optimizer end

# Prosty Stochastyczny Spadek Gradientu (SGD)
mutable struct SGD <: Optimizer
    lr::Float32      # Współczynnik uczenia
    # Można dodać inne parametry jak momentum, etc.
    # param_groups::Union{Nothing, Vector{Vector{TrackedValue}}} # Dla różnych lr per grupa
    
    SGD(lr=0.01f0) = new(lr)
end

# Funkcja aktualizująca parametry dla danego optymalizatora
# model_params to lista TrackedValue, które są parametrami modelu
function update!(opt::SGD, model_params::Vector{<:TrackedValue})
    for p in model_params
        if grad(p) !== nothing
            # Aktualizacja: param = param - lr * grad(param)
            # p.data .-= opt.lr .* grad(p) # Operacja in-place na danych
            # Ważne: p.data musi być modyfikowalne.
            # Jeśli p.data jest np. statyczną tablicą, to nie zadziała.
            # Dla standardowych tablic Julii to jest OK.
            current_data = value(p) # Pobierz obecne dane
            g = grad(p)
            
            # Sprawdzenie typów dla bezpieczeństwa, jeśli lr i g mogą mieć różne typy
            # typeof(opt.lr .* g) musi być zgodne z typeof(current_data)
            if typeof(current_data) == typeof(opt.lr .* g) || eltype(current_data) == eltype(opt.lr .* g)
                 current_data .-= opt.lr .* g
            else
                 # Próba konwersji lub rzucenie błędu
                 # Dla prostoty, zakładamy zgodność
                 # println("Type mismatch in SGD update: $(typeof(current_data)) vs $(typeof(opt.lr .* g))")
                 current_data .-= convert(eltype(current_data), opt.lr) .* convert.(eltype(current_data), g)
            end

            # p.data = new_data # Jeśli p.data nie jest modyfikowalne in-place
                              # Ale to zerwie połączenie z TrackedValue, jeśli `data` jest immutable.
                              # Dlatego modyfikujemy `p.data` in-place.
        else
            # To może się zdarzyć, jeśli parametr nie wpływa na koszt (np. odłączony od grafu)
            # println("Ostrzeżenie: Parametr $(p.op_name) nie ma gradientu podczas aktualizacji SGD.")
        end
    end
end