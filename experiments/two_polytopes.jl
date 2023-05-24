include("../all_functions/functions.jl")
include("../all_functions/plotting.jl")

# # This is the case that two polyhedra.
type = "polytopes"
n = 5 # Dimensions
max_iterations = 100000  # Maximum number of iterations
step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell"=>""))


num_vertices1 = 15  # Number of vertices for polytope 1
num_vertices2 = 10  # Number of vertices for polytope 2


# Generate random vertices for polytope 1, 2 within the range [a, b]
polytope_vertices1 = vertices(n, 0, 30, num_vertices1)
polytope_vertices2 = vertices(n, -10, 0, num_vertices2)

# # eg: the case that two balls contact 
# common_vertex = rand!(Vector{Float32}(undef,n))
# push!(polytope_vertices1, common_vertex)
# push!(polytope_vertices2, common_vertex)

# Create polytope 1 and 2 using the vertex representation
p1 = polyhedron(vrep(polytope_vertices1))
p2 = polyhedron(vrep(polytope_vertices2))
# removevredundancy!(p1)
# removevredundancy!(p2)

# Compute the Linear Minimization Oracle for polytope 1 and 2
lmo1 = PolytopeLMO(p1)
lmo2 = PolytopeLMO(p2)

# Compute initial points (interior)
x0 = FrankWolfe.compute_extreme_point(lmo1, center_of_mass(p1) * 1.1)
y0 = FrankWolfe.compute_extreme_point(lmo2, center_of_mass(p2) * 0.9)

# Perform the experiments
data = Dict() # Initialize data
data = run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size)

members = Dict("type" => type, "arg1" => p1, "arg2" => p2)
# membership_oracle(members)

# println("convergence rate (log): ", (log10(primal[argmin(primal)])-log10(primal[argmax(primal)]))/(log10(max_iterations)))


# Plot the min primal gaps
primal_gap_plotter(data, members)



# polytopes_plotter(xt, yt, p1, p2, primal, members)




