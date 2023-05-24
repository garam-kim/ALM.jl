# # Initial setting
type = "balls"
n = 6 # Number of dimensions
max_iterations = 1e5  # Maximum number of iterations

# Radius for the Lp-norm
r1 = rand(1:10) 
center1 = rand!(Vector{Float32}(undef,n),-3:5)
r2 = rand(1:6)
center2 = rand!(Vector{Float32}(undef,n),-5:10)

# Create LMOs for the shifted L2 balls
lmo1 = shiftedL2ball(r1, center1)
lmo2 = shiftedL2ball(r2, center2)


# Compute initial points
x0 = FrankWolfe.compute_extreme_point(lmo1, lmo1.center + rand(n) * lmo1.radius)
y0 = FrankWolfe.compute_extreme_point(lmo2, lmo2.center - rand(n) * lmo2.radius)

# Perform alternative linear minimization optimization with the given initial points, LMOs, and parameters
xt, yt, loss = Alternative_Frank_Wolfe(x0, y0, lmo1, lmo2, max_iterations, f)

# Find the optimal points
xstar, ystar = find_optimal(xt, yt, loss)

# Evaluate the primal function over iterations
primal = evaluate_primal(max_iterations, xt, yt, xstar, ystar)
members = Dict("type" => type, "arg1" => lmo1, "arg2" => lmo2)
# membership_oracle(members)
println("convergence rate (log): ", (log10(primal[argmin(primal)])-log10(primal[argmax(primal)]))/(log10(max_iterations)))

# Plot the L2 balls (only n = 2) and min primal gap
balls_plotter(xt, yt, lmo1, lmo2, primal, members)