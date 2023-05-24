
# # This is the case that two polyhedra.
type = "polytopes"
n = 4 # Dimensions
max_iterations = 100000  # Maximum number of iterations
num_vertices1 = 20  # Number of vertices for polytope 1
num_vertices2 = 10  # Number of vertices for polytope 2


# Generate random vertices for polytope 1, 2 within the range [a, b]
polytope_vertices1 = vertices(n, -10, 30, num_vertices1)
polytope_vertices2 = vertices(n, -10, 10, num_vertices2)

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

# Perform alternative linear minimization optimization with the given initial points and parameters
xt, yt, loss = Alternative_Linear_Minimization(x0, y0, lmo1, lmo2, max_iterations, f)
members = Dict("type" => type, "arg1" => p1, "arg2" => p2)
# membership_oracle(members)

xstar, ystar = find_optimal(xt, yt, loss)
primal = evaluate_primal(max_iterations,xt,yt,xstar,ystar)

println("convergence rate (log): ", (log10(primal[argmin(primal)])-log10(primal[argmax(primal)]))/(log10(max_iterations)))

# Plot the results, including the loss values, polyhedra, and xT
polytopes_plotter(xt, yt, p1, p2, primal, members)




