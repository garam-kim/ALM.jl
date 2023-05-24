import GLPK

using Plots
using Polyhedra
using Random
using FrankWolfe
using LinearAlgebra
using LaTeXStrings


# # Plot
function only_min(primal)
    new_data = Float64[]
    push!(new_data, primal[1])
    for t in 2:length(primal)
        entry = primal[t]
        if entry < new_data[end]
            push!(new_data, entry)
        else
            push!(new_data, new_data[end])
        end
    end
    if length(findall(new_data .== new_data[end])) >= 100
        idx = findfirst(new_data .== new_data[end])
        new_data = new_data[1:idx]
    end
    return new_data
end


function primal_gap_plotter(data, members)
    g = plot(legend=:topright,xtickfontsize = 10, ytickfontsize = 10, ylabelfontsize =15, linewidth = 2)
    colors = [:blue, :orange, :green, :red]
    n = length(data[1]["xt"][1])
    # for (idx, current_data) in enumerate(data)
    for (idx, current_data) in enumerate(data)
        if current_data["step_type"] == "line_search"
            current_label = current_data["step_type"]
        else
            current_label = current_data["step_type"]*", "*L"\ell="*string(current_data["ell"])
        end

        xt = current_data["xt"]
        yt = current_data["yt"]
        loss = current_data["loss"]
        xstar, ystar = find_optimal(xt, yt, loss) 
        primal = evaluate_primal(xt, yt, xstar, ystar)
        data[idx]["primal"] = primal
        g = plot!(only_min(primal), xscale=:log10, yscale=:log10, linestyle=:dot, color = colors[idx], label = current_label,linewidth = 5);
        println("current: ", current_label)
    end
    g = xlabel!("number of iterations")
    g = ylabel!("min"*L"_{i} h_i")
    plot(g)

    if members["type"] == "balls"
        address = "./experiments/figures/two_balls/"
    elseif members["type"] == "polytopes"
        address = "./experiments/figures/two_polytopes/"
    else
        address = "./experiments/figures/ball_polytope/"
    end
    filename = membership_oracle(members)*"_"*string(n)*"d"
    savefig(address * filename * ".png")
end


# # Auxiliary functions for plot

function polytopes_plotter(xt::Vector{Any}, yt::Vector{Any}, p1::DefaultPolyhedron, p2::DefaultPolyhedron, primal::Vector{Float64}, members)
    n = length(xt[1])
    # Create a plot g1 for the primal values
    # marker=:hex,
    g1 = plot(only_min(primal), xscale=:log10, yscale=:log10, linestyle=:dot, color =:green);
    g1 = plot!(legend=false, xtickfontsize = 10, ytickfontsize = 10, ylabelfontsize =15, linewidth = 2)
    g1 = xlabel!("number of iterations")
    g1 = ylabel!("min"*L"_{i} h_i")

    if n == 2
        labels = ["x*","y*"]
        colors = [:blue, :orange]
        idx = length(primal)

        # Create a plot g2 for the polytopes and optimal points
        g2 = plot(p1, color="blue", alpha=0.2, label="polytope 1");
        g2 = plot!(p2, color="red", alpha=0.2, label="polytope 2");
        g2 = scatter!([xt[idx][1]], [xt[idx][2]], color=colors[1], label=labels[1], legend=true);
        g2 = scatter!([yt[idx][1]], [yt[idx][2]], color=colors[2], label=labels[2]);
        g2 = plot!(legend =:topleft, xlabel = "x", ylabel = "y", yguidefontrotation = -90)
        plot(g1, g2)
    else
        plot(g1)
    end
    filename = membership_oracle(members)*"_"*string(n)*"d"
    savefig("./experiments/figures/two_polytopes/"*filename*".png")
end


function l2_polytope_plotter(xt, yt, lmo, p, primal, type)
    n = length(xt[1])
    # Create a plot g1 for the loss values
    g1 = plot(only_min(primal), xscale=:log10, yscale=:log10, linestyle=:dot, color =:green);
    g1 = plot!(legend=false, xtickfontsize = 10, ytickfontsize = 10, ylabelfontsize =15, linewidth = 2)
    g1 = xlabel!("number of iterations")
    g1 = ylabel!("min"*L"_{i} h_i")    
    if n == 2
        # Generate an array of angles for plotting the circle
        theta = range(0, 2π, length = 100)
        # Calculate the x and y coordinates of the circle's points
        xcoord =  lmo.right_hand_side .* cos.(theta)
        ycoord =  lmo.right_hand_side .* sin.(theta)

        labels = ["x*","y*"]
        colors = [:blue, :orange]
        idx = length(primal)

        # Create a plot g2 for the polytopes and optimal points
        g2 = plot(p, color="blue", alpha=0.2,label="polytope 1");
        g2 = plot!(xcoord, ycoord, color="red", alpha=0.2, aspect_ratio=:equal, seriestype=:shape, label=L"\ell^1 ball")
        g2 = scatter!([xt[idx][1]], [xt[idx][2]], color=colors[1], label=labels[1], legend=true);
        g2 = scatter!([yt[idx][1]], [yt[idx][2]], color=colors[2], label=labels[2]);
        g2 = plot!(legend =:topleft, xlabel = "x", ylabel = "y", yguidefontrotation = -90)
        plot(g1, g2)
    else
        plot(g1)
    end

    filename = membership_oracle(members)*"_"*string(n)*"d"
    savefig("./experiments/figures/ball_polytope/"*filename*".png")
end

function balls_plotter(xt, yt, lmo1, lmo2, primal, members, step_size)
    n = length(xt[1])
    # Create a plot g1 for the loss values
    g1 = plot(only_min(primal), xscale=:log10, yscale=:log10, linestyle=:dot, color =:green);
    g1 = plot!(legend=false, xtickfontsize = 10, ytickfontsize = 10, ylabelfontsize =15, linewidth = 2)
    g1 = xlabel!("number of iterations")
    g1 = ylabel!("min"*L"_{i} h_i")
    if n == 2
        # Generate an array of angles for plotting the circle
        theta = range(0, 2π, length = 100)
        # Calculate the x and y coordinates of the circle's points
        xcoord1 =  lmo1.center[1] .+ lmo1.radius .* cos.(theta)
        ycoord1 =  lmo1.center[2] .+ lmo1.radius .* sin.(theta)
        xcoord2 =  lmo2.center[1] .+ lmo2.radius .* cos.(theta)
        ycoord2 =  lmo2.center[2] .+ lmo2.radius .* sin.(theta)
        labels = ["x*","y*"]
        colors = [:blue, :orange]
        idx = length(primal)

        # Create a plot g2 for the polytopes and optimal points
        g2 = plot(xcoord1, ycoord1, color="red", alpha=0.2, aspect_ratio=:equal, seriestype=:shape, label=L"\ell^1 ball")
        g2 = plot!(xcoord2, ycoord2, color="blue", alpha=0.2, aspect_ratio=:equal, seriestype=:shape, label=L"\ell^1 ball")
        g2 = scatter!([xt[idx][1]], [xt[idx][2]], color=colors[1], label=labels[1]);
        g2 = scatter!([yt[idx][1]], [yt[idx][2]], color=colors[2], label=labels[2]);
        g2 = plot!(legend =:topleft, xlabel = "x", ylabel = "y", yguidefontrotation = -90)
        plot(g1, g2)
    else
        plot(g1)
    end

    if step_size["step_type"] == "line_search"
        filename = membership_oracle(members)*"_line_search_"*string(n)*"d"
        savefig("./experiments/figures/two_balls/"*filename*".png")
    else
        ell = step_size["ell"]
        filename = membership_oracle(members)*"_open_loop("*string(ell)*")_"*string(n)*"d"
        savefig("./experiments/figures/two_balls/"*filename*".png")
    end

end

