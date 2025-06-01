import Pkg
Pkg.activate(".")
Pkg.instantiate()

try
    include("../MyAutoDiffProject.jl")
catch e
    @warn "Could not include ../MyAutoDiffProject.jl. Make sure the path is correct."
    rethrow(e)
end

using .MyAutoDiffProject 
using Flux
using Statistics
using Random
using BenchmarkTools

println("Preparing data...")
Random.seed!(1234)
N_samples = 2000
in_features = 15
hidden_features = 10
out_features = 5

X_data = rand(Float32, in_features, N_samples) .* 10 .- 5
Y_true = mapslices(x -> sum(x.^2)/in_features + 0.5f0*x[1], X_data; dims=1)
Y_data = Y_true .+ (rand(Float32, out_features, N_samples) .* 0.2f0 .- 0.1f0)

# --- Definicja Twojej sieci ---
println("Defining MyAutoDiffProject model...")
my_model = MyAutoDiffProject.Chain(
    MyAutoDiffProject.Dense(in_features, hidden_features),
    MyAutoDiffProject.NeuralNetwork.ReLU(),
    MyAutoDiffProject.Dense(hidden_features, out_features)
)
my_loss_fn = MyAutoDiffProject.mse_loss
my_optimizer = MyAutoDiffProject.SGD(0.0005f0)

println("Training MyAutoDiffProject model...")
@time MyAutoDiffProject.train!(my_model, my_loss_fn, my_optimizer, X_data, Y_data, 10, batch_size=32)

println("Defining Flux model...")
flux_model = Flux.Chain(
    Flux.Dense(in_features, hidden_features, relu),
    Flux.Dense(hidden_features, out_features)
)
flux_optimizer_rule = Flux.Descent(0.0005f0) 

println("Training Flux model...")
data_loader = Flux.DataLoader((X_data, Y_data), batchsize=32, shuffle=true)

opt_state = Flux.Optimisers.setup(flux_optimizer_rule, flux_model)

@time for epoch in 1:10
    for (x_batch, y_batch) in data_loader
        grads = Flux.gradient(m -> begin
                                  predictions = m(x_batch)
                                  loss = sum((predictions .- y_batch).^2) / size(x_batch, 2) # MSE
                                  return loss
                              end,
                              flux_model)
        actual_gradients = grads[1]
        Flux.Optimisers.update!(opt_state, flux_model, actual_gradients)
    end
end

println("Comparing predictions...")
Random.seed!(42)
test_indices = rand(1:N_samples, 5)
X_test = X_data[:, test_indices]
Y_test_true = Y_true[:, test_indices]

my_preds = MyAutoDiffProject.value(my_model(X_test))

flux_preds = flux_model(X_test)

println("\nResults:")
for i in 1:5
    println("Example $i:")
    display_in_features = min(2, in_features)
    println("  Input (first $display_in_features features): ", round.(X_test[1:display_in_features, i], digits=3))
    println("  True Value: ", round.(Y_test_true[:, i], digits=3))
    println("  MyModel Prediction: ", round.(my_preds[:, i], digits=3))
    println("  Flux Prediction: ", round.(flux_preds[:, i], digits=3))
end

function calculate_mse(predictions, targets)
    return mean((predictions .- targets).^2)
end

println("\nMSE on test samples:")
println("  MyAutoDiffProject: ", round(calculate_mse(my_preds, Y_test_true), digits=4))
println("  Flux: ", round(calculate_mse(flux_preds, Y_test_true), digits=4))

println("-"^50)
println("Comparison complete.")
