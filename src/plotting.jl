import GLPK
using Plots
using Polyhedra
using Random
using FrankWolfe
using LinearAlgebra
using LaTeXStrings


# # Plot
function only_min(primal)
    """
    Replaces every entry in every list in data by the minimum up to said entry.

    Args:
        new_data: list
    """
    new_data = Float64[]
    push!(new_data, primal[1])
    for t in eachindex(primal[2:end])
        entry = primal[t]
        if entry < new_data[end]
            push!(new_data, entry)
        else
            push!(new_data, new_data[end])
        end
    end
    if length(findall(new_data .== new_data[end])) >= 500
        idx = findfirst(new_data .== new_data[end])
        new_data = new_data[1:idx]
    end
    if length(new_data) < 100 push!(new_data,1e-10) end
    return new_data
end

function determine_y_lims(data)
    """Determines how large the y-axis has to be given the data."""

    max_val = 1e-16
    min_val = 1e+16
    for entry in data
        primal = entry["primal"]
        min_val = max(min(primal[argmin(primal)] / 10, min_val), 1e-10)
        max_val = max(primal[argmax(primal)] * 10, max_val)
    end
    return min_val, max_val
end


function primal_gap_plotter(data, members)
    """Plots the primal gaps of different step-sizes and can save it under "./experiments/figures" + file_name + ".png"."""

    g = plot(legend=:topright, xtickfontsize = 10, ytickfontsize = 10, ylabelfontsize =15, linewidth = 2)
    colors = [:blue, :orange, :green, :red]
    n = length(data[1]["xt"][1])
    max_iterations = Int(10^floor(log10(length(data[1]["xt"]))))
    xstar, ystar = find_star(data) 

    for (idx, current_data) in enumerate(data)
        if current_data["step_type"] == "line_search"
            current_label = current_data["step_type"]
        else
            current_label = current_data["step_type"]*", "*L"\ell="*string(current_data["ell"])
        end
        xt = current_data["xt"]
        yt = current_data["yt"]
       
        primal = evaluate_primal(xt, yt, xstar, ystar, max_iterations)
        data[idx]["primal"] = primal
        g = plot!(only_min(primal), xscale=:log10, yscale=:log10, linestyle=:dot, color = colors[idx], label = current_label, linewidth = 5, legendfontsize = 12, framestyle = :box);
    end
    g = xlabel!("number of iterations")
    g = ylabel!("min"*L"_{i} h_i")
    g = ylims!(determine_y_lims(data))
    g = yticks!([10.0^(3k) for k in Int(floor(log10(determine_y_lims(data)[1]))):Int(floor(log10(determine_y_lims(data)[2])))])
    g = xticks!([10^k for k in 1:Int(floor(log10(max_iterations)))])
    plot(g)

    # if members["type"] == "balls"
    #     address = "./experiments/figures/two_balls/"
    # elseif members["type"] == "polytopes"
    #     address = "./experiments/figures/two_polytopes/"
    # else
    #     address = "./experiments/figures/ball_polytope/"
    # end
    # filename = "_"*string(n)*"d"
    # savefig(address * filename * ".png")
end


function polytope_plotter(p::DefaultPolyhedron, star)
    """Plots polytope and an optimum."""

    g = plot!(p, color = "blue", alpha = 0.2);
    g = scatter!([star[1]], [star[2]]);
    plot!(g)
end


function ball_plotter(lmo::shiftedL2ball, star)
    """Plots ball and an optimum."""

    # Generate an array of angles for plotting the circle
    theta = range(0, 2π, length = 100)

    # Calculate the x and y coordinates of the circle's points
    xcoord =  lmo.radius .* cos.(theta) .+ lmo.center[1]
    ycoord =  lmo.radius .* sin.(theta) .+ lmo.center[2]

    # Create a plot g2 for the polytopes and optimal points
    g = plot!(xcoord, ycoord, color="red", alpha=0.2, aspect_ratio=:equal, seriestype=:shape)
    g = scatter!([star[1]], [star[2]]);
    plot!(g)
end

