# train.jl
using ..AutoDiff # backward!, zero_grad! (przez NeuralNetwork.zero_grad_model!)
using ..NeuralNetwork # params, zero_grad_model!, Optimizer, update!

# Główna pętla treningowa
function train!(model, loss_fn, opt::Optimizer, X_train_raw, Y_train_raw, epochs::Int; batch_size::Union{Nothing,Int}=nothing)
    # X_train_raw, Y_train_raw to surowe dane (np. macierze Float32)
    # X_train_raw: (liczba_cech, liczba_przykładów)
    # Y_train_raw: (liczba_wyjść, liczba_przykładów)

    num_samples = size(X_train_raw, 2) # Liczba przykładów
    
    if batch_size === nothing
        batch_size = num_samples # Full batch jeśli nie podano
    end
    num_batches = ceil(Int, num_samples / batch_size)

    # println("Rozpoczynanie treningu: epochs=$epochs, batch_size=$batch_size, num_batches_per_epoch=$num_batches")

    model_parameters = NeuralNetwork.params(model) # Pobierz parametry raz

    for epoch in 1:epochs
        epoch_loss = 0.0f0
        
        # Opcjonalne: Mieszanie danych na początku każdej epoki
        # perm = Random.randperm(num_samples)
        # X_shuffled = X_train_raw[:, perm]
        # Y_shuffled = Y_train_raw[:, perm]
        X_shuffled = X_train_raw # Bez mieszania dla uproszczenia
        Y_shuffled = Y_train_raw


        for i in 1:num_batches
            start_idx = (i - 1) * batch_size + 1
            end_idx = min(i * batch_size, num_samples)
            
            X_batch_raw = X_shuffled[:, start_idx:end_idx]
            Y_batch_raw = Y_shuffled[:, start_idx:end_idx]

            # 1. Zerowanie gradientów dla wszystkich parametrów modelu
            NeuralNetwork.zero_grad_model!(model_parameters) # Przekazujemy pre-fetchowane parametry

            # 2. Propagacja w przód (Forward Pass)
            # Model powinien przyjąć surowe dane X_batch_raw i zwrócić TrackedValue
            Y_pred_tracked = model(X_batch_raw)

            # 3. Obliczenie funkcji kosztu
            # loss_fn powinna przyjąć TrackedValue (predykcje) i surowe dane (prawdziwe etykiety)
            # i zwrócić skalarną TrackedValue
            current_loss_tracked = loss_fn(Y_pred_tracked, Y_batch_raw)
            
            batch_loss_value = value(current_loss_tracked) # Pobierz wartość liczbową kosztu
            epoch_loss += batch_loss_value * (end_idx - start_idx + 1) # Ważona suma strat z batchy

            # 4. Propagacja wsteczna (Backward Pass) - obliczenie gradientów
            # Rozpoczynamy od gradientu funkcji kosztu (domyślnie 1.0)
            AutoDiff.backward!(current_loss_tracked)

            # 5. Aktualizacja parametrów modelu za pomocą optymalizatora
            NeuralNetwork.update!(opt, model_parameters)
            
            if i % max(1, div(num_batches, 5)) == 0 || i == num_batches # Loguj co jakiś czas
                #  println("  Epoch [$epoch/$epochs], Batch [$i/$num_batches], Batch Loss: $(round(batch_loss_value, digits=5))")
            end
        end
        
        avg_epoch_loss = epoch_loss / num_samples
        # println("Epoch: $epoch / $epochs, Average Loss: $(round(avg_epoch_loss, digits=7))")
        # println("-"^30)
    end
    # println("Trening zakończony. ✅")
end