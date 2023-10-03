using FrankWolfe
using LinearAlgebra
using Polyhedra
import GLPK
import MathOptInterface as MOI

"""
Unit simplex scaled by radius and shifted by center.
"""
struct shiftedUnitSimplex <: FrankWolfe.LinearMinimizationOracle
   radius :: Float64
   center :: Vector{Float64}
end

"""
Unit l2 ball scaled by radius and shifted by center.
"""
struct shiftedL2ball <: FrankWolfe.LinearMinimizationOracle
    radius :: Float64
    center :: Vector{Float64}
end
 


# # Technical replacements for different settings.
function FrankWolfe.compute_extreme_point(lmo::shiftedUnitSimplex, direction; v=nothing, kwargs...)
   idx = argmin(direction)
   if direction[idx] < 0
       return FrankWolfe.ScaledHotVector(lmo.radius, idx, length(direction)) .+ lmo.center
   end
   return FrankWolfe.ScaledHotVector(zero(Float64), idx, length(direction)) .+ lmo.center
end

function FrankWolfe.compute_extreme_point(lmo::shiftedL2ball, direction; v=similar(direction))
   dir_norm = norm(direction, 2)
   n = length(direction)
   # if direction numerically 0
   if dir_norm <= 10eps(float(eltype(direction)))
       @. v = lmo.radius / sqrt(n)
   else
       @. v = -lmo.radius * direction / dir_norm
   end
       
   return v + lmo.center
end



"""
Linear minimization oracle for given polytope.
"""
function PolytopeLMO(p::DefaultPolyhedron)
   # Initialize empty lists to store the halfspaces' coefficients and constants
   ws = [] # Normal vectors of the halfspaces
   cs = [] # Constants of the halfspaces

   # Extract normal vectors and constants from each halfspace in polyhedra
   for halfspace in hrep(p).halfspaces
       push!(ws, halfspace.a)
       push!(cs, halfspace.β)
   end
   n = length(ws[1])
   # Create a GLPK optimizer instance
   optimizer = GLPK.Optimizer()

   # Add variables to the optimizer
   x = MOI.add_variables(optimizer, n);

   # Add constraints to the optimizer based on the halfspaces
   for (w, c) in zip(ws, cs)
       affine = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w, x), 0.0)
       # Add the constraint to the optimizer: w * x <= c
       MOI.add_constraint(optimizer, affine, MOI.LessThan(c))
   end

   # Create a linear minimization oracle (LMO) using the optimizer
   lmo = FrankWolfe.MathOptLMO(optimizer)
   return lmo
end