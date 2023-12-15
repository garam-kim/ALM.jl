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
struct shiftedl2ball <: FrankWolfe.LinearMinimizationOracle
    radius :: Float64
    center :: Vector{Float64}
end

"""
Unit l1 ball scaled by radius and shifted by center.
"""
struct shiftedl1ball <: FrankWolfe.LinearMinimizationOracle
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


function FrankWolfe.compute_extreme_point(lmo::shiftedl2ball, direction; v=similar(direction))
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


function FrankWolfe.compute_extreme_point(lmo::shiftedl1ball, direction; v=similar(direction))
    idx = 0
    v = -one(eltype(direction))
    for i in eachindex(direction)
        if abs(direction[i]) > v
            v = abs(direction[i])
            idx = i
        end
    end
    sign_coeff = sign(direction[idx])
    if sign_coeff == 0.0
        sign_coeff -= 1
    end
    return FrankWolfe.ScaledHotVector(-lmo.radius * sign_coeff, idx, length(direction)) + lmo.center
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