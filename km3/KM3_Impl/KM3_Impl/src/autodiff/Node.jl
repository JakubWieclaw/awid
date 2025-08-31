abstract type CalculationGraphNode end

struct ConstantNode <: CalculationGraphNode
    value::Float64
end

struct VariableNode <: CalculationGraphNode
    name::Symbol
end

abstract type Operator end

struct Add <: Operator end
struct Multiply <: Operator end
struct Power <: Operator end
struct Sin <: Operator end
struct Cos <: Operator end
struct Exp <: Operator end
struct Log <: Operator end

struct OperationNode{O<:Operator, N} <: CalculationGraphNode
    operator::O
    input_nodes::NTuple{N, CalculationGraphNode}
end