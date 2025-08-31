include("Node.jl")

import Base: +, *, ^, sin, cos, exp, log

_constant(x::Real) = ConstantNode(Float64(x))
_wrap(x) = x isa CalculationGraphNode ? x : _constant(x)

+(a, b) = OperationNode(Add(), (_wrap(a), _wrap(b)))
*(a, b) = OperationNode(Multiply(), (_wrap(a), _wrap(b)))
^(a, b) = OperationNode(Power(), (_wrap(a), _wrap(b)))

sin(a) = OperationNode(Sin(), (_wrap(a),))
cos(a) = OperationNode(Cos(), (_wrap(a),))
exp(a) = OperationNode(Exp(), (_wrap(a),))
log(a) = OperationNode(Log(), (_wrap(a),))