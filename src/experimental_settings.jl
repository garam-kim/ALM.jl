include("../src/feasible_region.jl")
include("../src/plotting.jl")

"""
Experimental settings of each problem.

# Returns
- 'lmo1::LinearMinimizationOracle': respective linear minimization oracle.
- 'lmo2::LinearMinimizationOracle': respective linear minimization oracle.
- 'x0::Vector{Float64}': The starting point.
- 'y0::Vector{Float64}': The starting point.
"""


# Feasible sets are two L2 balls.
function build_two_l2balls_settings(n::Int64, status::String)
   Random.seed!(1025); r1 = rand(1:10)*norm(ones(n));
   Random.seed!(1024); center1 = rand!(Vector{Float32}(undef,n), -10:10);
   Random.seed!(1023); center2 = rand!(Vector{Float32}(undef,n), -10:10);

   if status == "disjoint"
      r2 = norm(center1 - center2) - r1 - norm(center1 - center2)/2
   elseif status == "touch"
      r2 = norm(center1 - center2) - r1 
   else
      r2 = norm(center1 - center2) - r1 + norm(center1 - center2)/2
   end
   @assert r2 >= 0 "negative radius"
   
   lmo1 = shiftedl2ball(r1, center1)
   lmo2 = shiftedl2ball(r2, center2)
   x0 = center1 + r1*unit_vector(n, 1)
   y0 = center2 + r2*unit_vector(n, n);

   return lmo1, lmo2, x0, y0
end

# Feasible sets are the L2 ball and the unit simplex.
function build_l2_unitsimplex_settings(n::Int64, status::String)
   rho = 5
   shift = rho * ones(n)
   
   if status == "disjoint"
      radius = rho/2 * norm(ones(n))
   elseif status == "touch"
      radius = rho * norm(ones(n))
   else
      radius = 1.5*rho * norm(ones(n))
   end
   
   lmo1 = shiftedl2ball(radius, zeros(n))
   lmo2 = shiftedUnitSimplex(rho, shift)

   x0 = unit_vector(n,1)
   y0 = shift + rho/2*(unit_vector(n,2))

   return lmo1, lmo2, x0, y0
end



# Feasible sets are two unit simplexes.
function build_unitsimplexes_settings(n::Int64, rho::Float64)
   shift = ones(n);

   lmo1 = FrankWolfe.UnitSimplexOracle(rho)
   lmo2 = shiftedUnitSimplex(rho/2, shift)

   x0 = unit_vector(n,1)
   y0 = shift + rho/2*(unit_vector(n,1)+unit_vector(n,2))
   
   return lmo1, lmo2, x0, y0
end



# Feasible sets are the L_infty ball and the L1 ball.
function build_poly_settings(flag::String)
   
   radius = 1; center = [1, 3];
   lmo1 = FrankWolfe.LpNormLMO{Inf}(1)
   lmo2 = shiftedl1ball(radius, center)

   if flag == "one_step"
      a = 0; b = 7/8;
   else
      a = 7/8; b = 1/5;
   end
   x0 = [a, 0.]; y0 = [b, 3];
   
   return lmo1, lmo2, x0, y0
end