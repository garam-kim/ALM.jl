import GLPK
import MathOptInterface as MOI

using Plots
using Polyhedra
using Random
using FrankWolfe
using LinearAlgebra
using LaTeXStrings

# Generate a collection of random vertices
function vertices(n, a, b, num_vertices)
    return([rand!(Vector{Float32}(undef,n),a:b) for _ in 1:num_vertices])
end

function PolytopeLMO(p)
    # Initialize empty lists to store the halfspaces' coefficients and constants
    ws = [] # Normal vectors of the halfspaces
    cs = [] # Constants of the halfspaces

    # Extract normal vectors and constants from each halfspace in polyhedra
    for halfspace in hrep(p).halfspaces
        push!(ws, halfspace.a)
        push!(cs, halfspace.β)
    end
    n = length(ws[1])
    # Create a GLPK optimizer instance
    optimizer = GLPK.Optimizer()

    # Add variables to the optimizer
    x = MOI.add_variables(optimizer, n);

    # Add constraints to the optimizer based on the halfspaces
    for (w, c) in zip(ws, cs)
        affine = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(w, x), 0.0)
        # Add the constraint to the optimizer: w * x <= c
        MOI.add_constraint(optimizer, affine, MOI.LessThan(c))
    end

    # Create a linear minimization oracle (LMO) using the optimizer
    lmo = FrankWolfe.MathOptLMO(optimizer)
    return lmo
end




function Alternative_Frank_Wolfe(x0, y0, lmo1, lmo2, max_iterations, f, step)
    
    # Initialize empty lists to store the loss, x points, and y points
    loss = Float64[]
    xt = []; yt = []
    x = copy(x0); y = copy(y0)
    
    if step["step_type"] == "open_loop"
        ell = step["ell"]

        for t = 0:max_iterations - 1
            # Store the current loss, x and y points
            push!(loss, f(x,y)) 
            push!(xt, x); push!(yt, y)

            # Compute the extreme point u using the lmo1 oracle and update x
            u = FrankWolfe.compute_extreme_point(lmo1, x - y)
            x += ell/(t + ell) * (u - x)

            # Compute the extreme point v using the lmo2 oracle and update y
            v = FrankWolfe.compute_extreme_point(lmo2, y - x)
            y += ell/(t + ell) * (v - y)
        end
    elseif step["step_type"] == "line_search"
        for t = 0:max_iterations - 1
            # Store the current loss, x and y points
            push!(loss, f(x,y)) 
            push!(xt, x); push!(yt, y)

            # Compute the extreme point u using the lmo1 oracle and update x
            u = FrankWolfe.compute_extreme_point(lmo1, x - y)
            eta_x = dot(x - y, x - u)/dot(u - x, u - x)
            x += eta_x * (u - x)

            # Compute the extreme point v using the lmo2 oracle and update y
            v = FrankWolfe.compute_extreme_point(lmo2, y - x)
            eta_y = dot(y - x, y - v)/dot(y - v, y - v)
            y += eta_y * (v - y)
        end
    end
    return xt, yt, loss
end

function run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size)
    data = []
    for step in step_size
        label = step["step_type"] * string(step["ell"])
        init = Dict()
        init["step_type"] = step["step_type"]
        init["ell"] = step["ell"]
        init["xt"], init["yt"], init["loss"] = Alternative_Frank_Wolfe(x0, y0, lmo1, lmo2, max_iterations, f, step)
        push!(data, init)
    end
    return data
end


function f(x, y)
    return norm(x - y)^2
end

function find_optimal(xt, yt, loss)
    star = argmin(loss)
    return xt[star] , yt[star]
end




function evaluate_primal(xt, yt, xstar, ystar)
    primal = Float64[]
    max_iterations = length(xt)
    for t in 1:Int(max_iterations - 1000)
        current_primal = f(xt[t], yt[t]) - f(xstar, ystar)
        push!(primal, current_primal)
    end
    deleteat!(primal, primal .== 0.0)

    if length(findall(primal.== 0.0)) >= 10
        idx = findfirst(primal.== 0.0)
        return primal[1:idx - 1]
    else
        return primal
    end
end

struct shiftedL2ball <: FrankWolfe.LinearMinimizationOracle
    radius :: Float64
    center :: Vector{Float64}
end

function FrankWolfe.compute_extreme_point(lmo::shiftedL2ball, direction; v=similar(direction))
    dir_norm = norm(direction, 2)
    n = length(direction)
    # if direction numerically 0
    if dir_norm <= 10eps(float(eltype(direction)))
        @. v = lmo.radius / sqrt(n)
    else
        @. v = -lmo.radius * direction / dir_norm
    end
    return v + lmo.center
end

function membership_oracle(members)
    type = members["type"]
    
    if type == "balls"
        lmo1 = members["arg1"]; lmo2 = members["arg2"]
        distance = norm(lmo1.center - lmo2.center)
        if distance == lmo1.radius + lmo2.radius
            loc = "contact"
            println("Two balls: contact at an one point")
        elseif distance < lmo1.radius + lmo2.radius
            loc = "intersect"
            println("Two balls: intersect")
        else
            loc = "disjoint"
            println("Two balls: disjoint")
        end

    elseif type == "polytopes"
        p1 = members["arg1"]; p2 = members["arg2"]
        points_intersect = npoints(intersect(p1, p2))
        if points_intersect >= 1
            loc = "intersect"
            println("Two polytopes: intersect")
        else
            loc = "disjoint"
            println("Two polytopes: disjoint")
        end

    else
        loc = ""
        println("TBD...")
    end
    return loc
end
