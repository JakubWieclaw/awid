# loss.jl
using ..AutoDiff # TrackedValue
using ..AutoDiff.OpsAD # Dostęp do zdefiniowanych operacji jak -, ^, sum_elements, /

# Błąd średniokwadratowy (Mean Squared Error)
function mse_loss(y_pred::TrackedValue, y_true_raw)
    # y_true_raw to surowe dane (np. Float32), nie TrackedValue
    # Możemy je opakować w TrackedValue bez potrzeby gradientu, jeśli operacje tego wymagają
    # lub operacje w ops.jl powinny umieć obsłużyć (TrackedValue - Number)
    
    # y_true = TrackedValue(y_true_raw, op_name=:y_true_const) # Jeśli operacje wymagają dwóch TrackedValue
    # error = y_pred - y_true
    
    # Załóżmy, że operacje w ops.jl (jak '-') potrafią obsłużyć TrackedValue - RawData
    # Jeśli nie, trzeba by y_true_raw opakować.
    # Dla przykładu, załóżmy, że '-' działa z (TrackedValue, Number) i zwraca TrackedValue
    # i że jego _backward_fn_data poprawnie obsługuje to (gradient tylko dla TrackedValue).
    
    # (y_pred - y_true)
    error_val = y_pred - y_true_raw # Zakładając, że ops.- obsługuje (TrackedValue - Number)

    # (y_pred - y_true)^2
    squared_error = error_val ^ 2 # Zakładając, że ops.^ obsługuje (TrackedValue, Number)

    # sum((y_pred - y_true)^2)
    sum_sq_error = sum_elements(squared_error) # Używa sum_elements z AutoDiff.ops

    # N = number of elements or batch size * output_dim
    # Tutaj uproszczenie - dzielimy przez liczbę elementów w y_true_raw
    # Właściwsze byłoby N = batch_size
    N = length(y_true_raw) 
    if N == 0 return TrackedValue(0.0f0; op_name=:empty_loss) end # Zabezpieczenie

    # sum((y_pred - y_true)^2) / N
    loss = sum_sq_error / N # Używa operatora / zdefiniowanego w AutoDiff.ops dla (TrackedValue / Number)
    
    return loss
end