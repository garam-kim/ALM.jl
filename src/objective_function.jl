using LinearAlgebra

## Objective function and its gradient 

function f(x, y)
   return .5 * norm(x - y)^2
end

function grad(x, y)
   return x - y
end