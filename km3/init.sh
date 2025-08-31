#!/usr/bin/env bash

# === CONFIG ===
PROJECT_NAME="KM3_Impl"

# === CREATE JULIA PROJECT ===
echo "📦 Creating Julia project: $PROJECT_NAME"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME || exit

# Initialize Julia project with Pkg
julia -e "using Pkg; Pkg.generate(\"$PROJECT_NAME\")"

# Move into src folder created by Pkg.generate
cd $PROJECT_NAME || exit

# === CREATE DIRECTORY STRUCTURE ===
echo "📂 Creating folder structure..."
mkdir -p src/autodiff
mkdir -p src/nn
mkdir -p src/optim
mkdir -p src/utils
mkdir -p src/training
mkdir -p test
mkdir -p examples
mkdir -p docs/src

# === CREATE MAIN MODULE FILE ===
cat <<EOL > src/$PROJECT_NAME.jl
module $PROJECT_NAME

# Re-export submodules
include("autodiff/AutoDiff.jl")
include("nn/Layers.jl")
include("optim/Optimizers.jl")
include("training/Trainer.jl")

end # module
EOL

# === AUTODIFF FILES ===
cat <<EOL > src/autodiff/AutoDiff.jl
module AutoDiff

include("Node.jl")
include("Ops.jl")
include("GradientCheck.jl")

end # module
EOL

touch src/autodiff/Node.jl
touch src/autodiff/Ops.jl
touch src/autodiff/GradientCheck.jl

# === NN FILES ===
cat <<EOL > src/nn/Layers.jl
module Layers

include("Activations.jl")
include("Losses.jl")
include("Models.jl")

end # module
EOL

touch src/nn/Activations.jl
touch src/nn/Losses.jl
touch src/nn/Models.jl

# === OPTIM FILES ===
cat <<EOL > src/optim/Optimizers.jl
module Optimizers

include("Scheduler.jl")

end # module
EOL

touch src/optim/Scheduler.jl

# === UTILS FILES ===
cat <<EOL > src/utils/Initializers.jl
module Initializers
# Xavier, Glorot, etc.
end
EOL

touch src/utils/Profiling.jl
touch src/utils/Serialization.jl

# === TRAINING FILES ===
cat <<EOL > src/training/Trainer.jl
module Trainer

include("DataLoader.jl")
include("Metrics.jl")

end # module
EOL

touch src/training/DataLoader.jl
touch src/training/Metrics.jl

# === TESTS ===
cat <<EOL > test/runtests.jl
using Test
using $PROJECT_NAME

include("test_autodiff.jl")
include("test_layers.jl")
include("test_training.jl")
include("benchmarks.jl")
EOL

touch test/test_autodiff.jl
touch test/test_layers.jl
touch test/test_training.jl
touch test/benchmarks.jl

# === EXAMPLES ===
touch examples/mnist.jl
touch examples/rnn_text.jl
touch examples/compare_flux.jl

# === DOCS ===
cat <<EOL > docs/make.jl
using Documenter
using $PROJECT_NAME

makedocs(
    sitename = "$PROJECT_NAME Documentation",
    modules = [$PROJECT_NAME],
    format = Documenter.HTML()
)
EOL

cat <<EOL > docs/src/index.md
# $PROJECT_NAME Documentation

Welcome to the docs!
EOL

echo "✅ Project $PROJECT_NAME created successfully!"
