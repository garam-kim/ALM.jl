# # Initial setting
type = "balls"
n = 2 # Number of dimensions
max_iterations = 1e5  # Maximum number of iterations

step_size = [Dict("step_type" => "open_loop", "ell" => ell) for ell in [2, 4, 6]]
push!(step_size, Dict("step_type" => "line_search", "ell"=>""))

# Radius for the Lp-norm
r1 = rand(1:10) 
center1 = rand!(Vector{Float32}(undef,n),-1:5)
r2 = rand(1:6)
center2 = rand!(Vector{Float32}(undef,n),-5:10)

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
# println("convergence rate (log): ", (log10(primal[argmin(primal)])-log10(primal[argmax(primal)]))/(log10(max_iterations)))

# Plot the min primal gaps
primal_gap_plotter(data, members)

