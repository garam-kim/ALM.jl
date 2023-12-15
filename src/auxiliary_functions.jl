using Polyhedra
using Random
using FrankWolfe
using LinearAlgebra
using DelimitedFiles
include("../src/feasible_region.jl")

# Generate a collection of random vertices
function vertices(n, a, b, num_vertices)
    """ Generates polytope vertices randomly within a given interval [a, b]

    Args:
        n: integer 
            dimension
        a: Float64
            the begining of the interval
        b: Float64
            the end of the interval
        num_vertices: integer
            the number of vertices to be generated

    Returns:
        array of the vertices: Vector{Vector{Float64}}
    """
    return([rand!(Vector{Float32}(undef,n),a:b) for _ in 1:num_vertices])
end

function unit_vector(n, i)
    """ Generates i-th unit vector 
    Args:
        n: integer
            dimension
        i: integer

    Returns:
        v: Vector{Float64}
            unit vector
    """
    v = zeros(n); v[i] = 1
    return v
end

function mean(arr)
    """ Average of the elements in the array """
    return sum(arr)/length(arr)
end
 



function find_star(data)
    """Find a numerical optimum that gives the lowest function value among all data. """
    xstar = data[1]["xt"][end]
    ystar = data[1]["yt"][end]
    for i in 2:length(data)
        x_candid = data[i]["xt"][end]
        y_candid = data[i]["yt"][end]
        if f(x_candid, y_candid) < f(xstar, ystar)
            xstar = x_candid
            ystar = y_candid
        end
    end
    return xstar, ystar
end

function evaluate_primal(xt, yt, xstar, ystar, max_iterations)
    """
    From loss, computes primal gaps.
    """

    primal = Float64[]
    T = length(xt)
    max_iterations = Int(10^floor(log10(max_iterations)))
    t = 1; current_primal = Inf
    while t <= min(T, Int(max_iterations)) && current_primal >= 1e-10
        current_primal = f(xt[t], yt[t]) - f(xstar, ystar)
        push!(primal, current_primal)
        t += 1
    end
    idx = findall(x->x.<=1e-10 && x.>-1e-6, primal)
    primal[idx] .= 1e-10 
    # if length(findall(primal[idx[1]+1:end] == 1e-10)) > 5 deleteat!(primal, idx) end
    @assert sum(primal.<0) == 0 "primal should be positive"
    return primal
end


function is_intersect(xstar::Vector{Float64}, ystar::Vector{Float64})
    """
    When two optimums are close enough, decide that the sets intersect. Otherwise, disjoint.
    """
    if norm(xstar.-ystar) < 1e-4 println("intersect") else println("disjoint") end
end

function is_intersect(p1::DefaultPolyhedron, p2::DefaultPolyhedron)
    """
    When two optimums are close enough, decide that the sets intersect. Otherwise, disjoint.
    """
    if npoints(intersect(p1,p2))!=0 println("intersect") else println("disjoint") end
end


function find_location(star::Vector{Float64}, polytope::DefaultPolyhedron)
    """
    Find the location of a point in a polytope.
    """
    if length(findall(x -> abs(norm(star .- x)) < 1e-7, collect(points(polytope)))) != 0
       println("a vertex")
    elseif length(findall(i -> abs(dot(hrep(polytope).halfspaces[i].a, star) - hrep(polytope).halfspaces[i].β) < 1e-7, collect(1:nhalfspaces(hrep(polytope))))) >= 1
       println("in the relative interior of a halfspace")
    else
       println("in the interior")
    end
end



function find_location(star, lmo::shiftedl2ball)
    """
    Find the location of a point in a ball.
    """
    if lmo.radius - norm(star.-lmo.center) > 1e-5
        println("the optimum is in the interior")
    elseif abs(lmo.radius - norm(star.-lmo.center)) <= 1e-5
        println("the optimum is on the border")
    else
        println("the optimum is in the exterior")
    end
end

function find_location(star, lmo::FrankWolfe.UnitSimplexOracle)
    """
    Find the location of a point in a ball.
    """
    # if lmo.radius - norm(star.-lmo.center) > 1e-5
    #     println("the optimum is in the interior")
    # elseif abs(lmo.radius - norm(star.-lmo.center)) <= 1e-5
    #     println("the optimum is on the border")
    # else
    #     println("the optimum is in the exterior")
    # end
end



function check(data, polytope::DefaultPolyhedron, flag::Int)
    """
    Feasibility check, whether all iterates are in a given polytope.
    """
    if flag == 1 entry = "xt" else entry = "yt" end
    @assert [in.(data[i][entry][2:end], polytope) == ones(length(data[i][entry][2:end])) for i in eachindex(data)] ==ones(length(data)) "feasiblity error"
end

function check(data, lmo::shiftedl2ball, flag::Int)
    """
    Feasibility check, whether all iterates are in a given ball.
    """
    if flag == 1 entry = "xt" else entry = "yt" end
    @assert [mean(norm.(data[i][entry][2:end].-[lmo.center]) .- lmo.radius .< 10e-5) == 1 for i in eachindex(data)] ==ones(length(data)) "feasiblity error"
end

function check(data, lmo::shiftedUnitSimplex, flag::Int)
    """
    Feasibility check, whether all iterates are in a given ball.
    """
    if flag == 1 entry = "xt" else entry = "yt" end
    @assert [sum(sum.(data[i][entry][1:end].-[lmo.center]).-lmo.radius) <  10e-5 for i in eachindex(data)]==ones(length(data)) "feasiblity error"
end


function check(data, lmo::FrankWolfe.UnitSimplexOracle, flag::Int)
    """
    Feasibility check, whether all iterates are in a given ball.
    """
    if flag == 1 entry = "xt" else entry = "yt" end
    @assert [sum(sum.(data[i][entry][1:end]).-lmo.right_side) <  10e-5 for i in eachindex(data)]==ones(length(data)) "feasiblity error"
end




#### All function below are not used so far. 

# FW algorithm for the inexact projection
function frank_wolfe(initial_point, point_you_want_to_get_close, lmo,iter)
    x = copy(initial_point); t=0; dual_gap = Inf
    while t <= iter && dual_gap >= max(1e-10, eps(float(typeof(dual_gap))))
        eta = 2/(t+2)
        gradient = x - point_you_want_to_get_close
        u = FrankWolfe.compute_extreme_point(lmo, gradient)
        dual_gap = dot(x, gradient) - dot(u, gradient)
        x += eta * (u - x)
        t+= 1
    end
    return x
end
 
# x0: initial_point, y0: pt that you want to get close  
function Projection(x0, y0, lmo1, lmo2, max_iteration, inner_iter)
    x = copy(x0); y = copy(y0); t = 0; 
    loss=[]; xt = []; yt = [];
    while t <= max_iteration 
       push!(loss, f(x,y))
       push!(xt, x); push!(yt,y);
       x = frank_wolfe(x, y, lmo1, inner_iter)
       y = frank_wolfe(y, x, lmo2, inner_iter)
       t += 1
    end
    return xt,yt,loss
end


function projection_shifted_l2ball(y, r, c)
    val = max(r, norm(y-c))
    return (r/val).* (y-c) .+ c
end

function projection_standardSimplex(y, r)
    if sum(y) == r return y end
    v, vv, rho = [y[1]], [], y[1] - r
    N = length(y)
    for n in 2:N
        yn = y[n]
        if yn > rho
            rho += (yn - rho) / (length(v) + 1)
            if rho > yn - r
                push!(v, yn)
            else
                append!(vv, v)
                v = [yn]
                rho = yn - r
            end
        end
    end

    if length(vv) > 0
        for w in vv
            if w > rho
                push!(v, w)
                rho += (w - rho) / length(v)
            end
        end
    end

    l, flag = length(v), 1
    while flag == 1
        for w in v
            if w <= rho
                deleteat!(v, findall(v .== w))
                rho += (rho - w) / length(v)
            end
        end
        if length(v) != l
            l, flag = length(v), 1
        else
            flag = 0
        end
    end

    return max.(y .- rho, 0)
end

function find_optimal_face(star, polytope::DefaultPolyhedron)
    faces = [];
    entry = findall(i -> abs(dot(hrep(polytope).halfspaces[i].a, star) - hrep(polytope).halfspaces[i].β) < 1e-7, collect(1:nhalfspaces(hrep(polytope))))
    if length(entry) >= 1
        for i in eachindex(entry)
            push!(faces, hrep(polytope).halfspaces[i])
        end
        return faces
    end
end

function find_parallel_faces(face1, face2)
    for i in eachindex(face1)
        for j in eachindex(face2)
            if abs(dot(face1[i].a,face2[j].a)/(norm(face1[i].a)*norm(face2[j].a))) > 1-1e-7
                println("i = ",i,", j = ",j)
                println(face1[i].a./face2[j].a)
                println(face1[i])
                println(face2[j])
            end
        end
    end
end


function find_optimal_sets(data, lmo1, lmo2)
    """
    Decide the location of the last iterates from all data.
    """
    for i in eachindex(data)
        println("* "*data[i]["step_type"]*"_"*string(data[i]["ell"]))
        find_location(data[i]["xt"][end], lmo1)
        find_location(data[i]["yt"][end], lmo2)
    end
end