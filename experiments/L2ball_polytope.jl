# # This is the case of a polytope and l2 ball.
type = "tbd"
# # Initial setting
n = 2;  # Dimensions
max_iterations = 1000000;  # Maximum number of iterations
num_vertices = 7;

# Radius for the Lp-norm
r = 2; 

# Generate random vertices for polytope within the range [a,b]
polytope_vertices = vertices(n, 0, 8, num_vertices)

# Create polytope using the vertex representation
p = polyhedron(vrep(polytope_vertices))
removevredundancy!(p)


# Compute the Linear Minimization Oracle (LMO) for polytope
lmo = PolytopeLMO(p)

# LMO based on the L2 (Euclidean) norm
l2 = FrankWolfe.LpNormLMO{Float64,2}(r)
# l2 = shiftedL2ball(r,center)

# Compute initial points
x0 = FrankWolfe.compute_extreme_point(lmo, center_of_mass(p) * 0.5)
y0 = FrankWolfe.compute_extreme_point(l2, rand!(Vector{Float32}(undef,n)))

# Perform alternative linear minimization optimization with the given initial points, LMOs, and parameters
xt, yt, loss = Alternative_Frank_Wolfe(x0, y0, lmo, l2, max_iterations, f)
members = Dict("type" => type, "arg1" => p, "arg2" => l2)
# membership_oracle(members)


xstar, ystar = find_optimal(xt, yt, loss)

primal = evaluate_primal(max_iterations, xt, yt, xstar, ystar)
println("convergence rate (log): ", (log10(primal[argmin(primal)])-log10(primal[argmax(primal)]))/(log10(max_iterations)))

# Plot the results, including the loss values, polytope, L2 ball radius, and xT
l2_polytope_plotter(xt, yt, l2, p, primal, members)