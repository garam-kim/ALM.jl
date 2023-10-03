include("../src/feasible_region.jl")
include("../src/experiments_auxiliary_functions.jl")
include("../src/alternative_linear_minimization.jl")
include("../src/auxiliary_functions.jl")
include("../src/objective_function.jl")
include("../src/plotting.jl")

# # Initial setting
type = "ball and polytope"
max_iterations = 10^5+200;  # Maximum number of iterations
step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell"=>""))
 
n = 5;  # Dimensions
num_vertices = n*2;

# Generate random vertices for polytope within the range [a,b]
polytope_vertices = vertices(n, 0, 8, num_vertices).*.5

# Create polytope using the vertex representation
p = polyhedron(vrep(polytope_vertices))
# removevredundancy!(p)

# Compute the Linear Minimization Oracle (LMO) for polytope
lmo = PolytopeLMO(p)

# LMO based on the L2 (Euclidean) norm (intersect)
r = 1.5; c = zeros(n);
l2 = shiftedL2ball(r,c)


# Compute initial points
x0 = zeros(n)
y0 = center_of_mass(p) * 0.9

# Perform the experiments
data = Dict() # Initialize data
data = run_experiment(x0, y0, l2, lmo, max_iterations, f, step_size)
println("complete: running experiments")

members = Dict("type" => type, "arg1" => l2, "arg2" => p)
xstar, ystar = find_star(data)

# feasibility check
check(data, l2, 1); check(data, p, 2);

is_intersect(xstar, ystar)
print("P: "); find_location(xstar, l2)
print("Q: "); find_location(ystar, p)


# # Plot the min primal gaps
primal_gap_plotter(data, members)
