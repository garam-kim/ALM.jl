include("../src/feasible_region.jl")
include("../src/experiments_auxiliary_functions.jl")
include("../src/alternative_linear_minimization.jl")
include("../src/auxiliary_functions.jl")
include("../src/objective_function.jl")
include("../src/plotting.jl")

# # Initial setting
type = "balls"
max_iterations = 10^5+200  # Maximum number of iterations
step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell" => ""))

n = 50

r1 = rand(1:5)
center1 = rand!(Vector{Float32}(undef,n), -5:0)
center2 = rand!(Vector{Float32}(undef,n), 0:10)

# touch
# r2 = norm(center1 - center2) - r1 
# intersect
# r2 = norm(center1 - center2) - r1 + norm(center1 - center2)/2
# disjoint
r2 = norm(center1 - center2) - r1 - 1
@assert r2 >= 0 "negative radius"


# Create LMOs for the shifted L2 balls
lmo1 = shiftedL2ball(r1, center1)
lmo2 = shiftedL2ball(r2, center2)


# Compute initial points
x0 = FrankWolfe.compute_extreme_point(lmo1, lmo1.center + rand(n) * lmo1.radius)
y0 = FrankWolfe.compute_extreme_point(lmo2, lmo2.center - rand(n) * lmo2.radius)

# Perform the experiments
data = Dict() # Initialize data
data = run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size)

members = Dict("type" => type, "arg1" => lmo1, "arg2" => lmo2)
xstar, ystar = find_star(data);

# feasibility check
check(data, lmo1, 1); check(data, lmo2, 2)

is_intersect(xstar, ystar)
print("P: "); find_location(xstar, lmo1)
print("Q: "); find_location(ystar, lmo2)

# # Plot the min primal gaps
primal_gap_plotter(data, members)