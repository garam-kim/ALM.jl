include("../src/feasible_region.jl")
include("../src/experiments_auxiliary_functions.jl")
include("../src/alternative_linear_minimization.jl")
include("../src/auxiliary_functions.jl")
include("../src/objective_function.jl")
include("../src/plotting.jl")

# # Initial setting
type = "polytopes"
max_iterations = 10^5+200  # Maximum number of iterations
step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell"=>""))
# step_size = [Dict("step_type" => "line_search", "ell" => "")]

# dimensions
n = 4
num_vertices1 = n * 4  # Number of vertices for polytope 1
num_vertices2 = n * 3  # Number of vertices for polytope 2

# Generate random vertices for polytope 1, 2 within the range [a, b]
polytope_vertices1 = vertices(n, -5, 0, num_vertices1)
polytope_vertices2 = vertices(n, 0, 4, num_vertices2)

# Create polytope 1 and 2 using the vertex representation
p1 = polyhedron(vrep(polytope_vertices1))
p2 = polyhedron(vrep(polytope_vertices2))
# removevredundancy!(p1)
# removevredundancy!(p2)
println("complete: building polyhedrons")


# Compute the Linear Minimization Oracle for polytope 1 and 2
lmo1 = PolytopeLMO(p1)
lmo2 = PolytopeLMO(p2)


# Compute initial points (interior)
x0 = center_of_mass(p1) 
y0 = center_of_mass(p2)


# Perform the experiments
data = Dict() # Initialize data
data = run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size)
println("complete: running experiments")

members = Dict("type" => type, "arg1" => p1, "arg2" => p2)
xstar, ystar = find_star(data)

   
# feasibility check 
check(data, p1, 1); check(data, p2, 2);

is_intersect(xstar, ystar)
print("P: "); find_location(xstar, p1)
print("Q: "); find_location(ystar, p2)


# find_optimal_sets(data, p1, p2)

# # Plot the min primal gaps
primal_gap_plotter(data, members)