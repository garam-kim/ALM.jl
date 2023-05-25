include("../all_functions/functions.jl")
include("../all_functions/plotting.jl")

# # This is the case of a polytope and l2 ball.
type = ""
# # Initial setting
# n = 2;  # Dimensions
max_iterations = 1000000;  # Maximum number of iterations
step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell"=>""))

num_vertices = 7;

# Radius for the Lp-norm
r = 4; 
dimensions = 3
for n in dimensions
   # Generate random vertices for polytope within the range [a,b]
   polytope_vertices = vertices(n, 2, 5, num_vertices)
   
   # Create polytope using the vertex representation
   p = polyhedron(vrep(polytope_vertices))
   # removevredundancy!(p)


   # Compute the Linear Minimization Oracle (LMO) for polytope
   lmo = PolytopeLMO(p)

   # LMO based on the L2 (Euclidean) norm
   l2 = FrankWolfe.LpNormLMO{Float64,2}(r)
   println(polytope_vertices)
   # # eg: the case that polytope and ball contact/ disjoint
   # push!(polytope_vertices, zeros(n))
   # c = rand!(Vector{Float32}(undef,n),-10:-1)
   # r = Float64(norm(c)) - 1
   # @assert r > 0
   # l2 = shiftedL2ball(r,c)

   # Compute initial points
   x0 = center_of_mass(p) * 0.9
   y0 = FrankWolfe.compute_extreme_point(l2, rand!(Vector{Float32}(undef,n)))

   # Perform alternative linear minimization optimization with the given initial points, LMOs, and parameters
   data = Dict() # Initialize data
   data = run_experiment(x0, y0, lmo, l2, max_iterations, f, step_size)

   members = Dict("type" => type, "arg1" => p, "arg2" => l2)

   primal_gap_plotter(data, members)
end
