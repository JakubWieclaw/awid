module MyAutoDiffProject

# --- AutoDiff Submodule ---
include("AutoDiff/AutoDiff.jl")
using .AutoDiff
# Eksportuj kluczowe elementy AutoDiff, które będą potrzebne na zewnątrz
export TrackedValue, value, grad, backward!
# Możesz chcieć wyeksportować też przeciążone operatory, jeśli są zdefiniowane w AutoDiff.ops
# export +, -, *, relu # etc.

# --- NeuralNetwork Submodule ---
include("NeuralNetwork/NeuralNetwork.jl")
using .NeuralNetwork
# Eksportuj kluczowe elementy NeuralNetwork
export Dense, Chain, ReLU # Warstwy i aktywacje
export mse_loss           # Funkcje kosztu
export SGD, update!       # Optymalizatory
export train!             # Funkcja treningowa
export params             # Funkcja do zbierania parametrów modelu

end