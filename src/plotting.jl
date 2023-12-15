using LinearAlgebra
using LaTeXStrings
using CairoMakie
using Interpolations
using Plots

# # Plot
function only_min(primal::AbstractVector)
    """
    Replaces every entry in every list in data by the minimum up to said entry.

    Args:
        new_data: list
    """
    new_data = Float64[]
    push!(new_data, primal[1])
    for t in eachindex(primal)
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
    return new_data[2:end]
end


function determine_y_lims(data)
    """Determines how large the y-axis has to be given the data."""

    max_val = 1e-16
    min_val = 1e+16
    for entry in data
        primal = entry["primal"]
        min_val = max(min(primal[argmin(primal)] / 10, min_val), 1e-9)
        max_val = max(primal[argmax(primal)] * 10, max_val)
    end
    return min_val, max_val
end


function get_markers(primal::AbstractVector, n_makers::Int64)
    """Returns the x and y coordinates of the markers that will be plotted on the graph. This takes into account the scaling of the axes."""
    primal = only_min(primal)
    itp = LinearInterpolation(log10.(1:length(primal)),log10.(primal))
    xcoor = exp10.(range(log10(1), log10(length(primal)), n_makers))
    ycoor = exp10.(itp(log10.(xcoor)))
    return xcoor, ycoor
end

function plotting_function(data)
    """A function that manages all the parameters required for generating a plot."""
    colors = [:magenta, :blue, :limegreen, :goldenrod1]
    markers = [:rect, :pentagon, :dtriangle, :circle, :diamond]
    fig = Figure(resolution = (600, 500))

    Axis(fig[1, 1], 
        xscale = log10, yscale = log10, 
        xlabel = "Number of iterations", ylabel = L"$\min_i \;h_i$",

        xticks = [10^k for k in 1:Int(floor(log10(max_iterations)))],
        xtickformat = x -> [L"10^{%$x}" for x in 1:Int(floor(log10(max_iterations)))],

        yticks = [10.0^(3k) for k in -3:1],
        ytickformat = x -> [L"10^{%$(3x)}" for x in -3:1],

        xlabelfont = "serif-roman", ylabelfont = "serif-roman",
        xlabelsize = 28, ylabelsize = 28,
        xticklabelsize = 30, yticklabelsize = 30
    )

    xstar, ystar = find_star(data) 

    marker_num = 10
    for (idx, current_data) in enumerate(data)
        xt = current_data["xt"]
        yt = current_data["yt"]
        primal = evaluate_primal(xt, yt, xstar, ystar, max_iterations)
        data[idx]["primal"] = primal 

        if current_data["step_type"] == "open-loop" 
            t = string(current_data["ell"]) 
            labels = L"open-loop, $\ell$ = %$(t)"
        else 
            labels = L"line-search $$"
        end

        xcoor, ycoor = get_markers(primal, marker_num)
        scatterlines!(xcoor, ycoor, markersize = 16, color =(colors[idx],0.001),marker=markers[idx],markercolor = colors[idx])
        scatterlines!(xcoor[1], ycoor[1], markersize = 16,color =colors[idx],marker=markers[idx],markercolor = colors[idx], label = labels)
        lines!(only_min(primal),linewidth = 2, color = colors[idx])
    end 
    axislegend(labelsize = 23, framecolor =:grey60, bgcolor=(:white, 0.4), position = :rt)
    # savefig("strcvx_intersect.png")
    return fig
end


function plotting_ablation_function(data, xstar::AbstractVector, ystar::AbstractVector)
    """A function that manages all the parameters required for generating a plot."""

    colors = [:orangered, :blue3, :mediumseagreen, :goldenrod1]
    markers = [:pentagon, :utriangle, :circle, :diamond]

    fig = Figure(resolution = (600, 500))

    ax = Axis(fig[1, 1], 
        xscale = log10, yscale = log10, 
        xlabel = "Number of iterations", ylabel = L"$\min_i \;h_i$",

        xticks = [10^k for k in 1:Int(floor(log10(max_iterations)))],
        xtickformat = x -> [L"10^{%$x}" for x in 1:Int(floor(log10(max_iterations)))],

        xlabelfont = "serif-roman", ylabelfont = "serif-roman",
        xlabelsize = 28, ylabelsize = 28,
        xticklabelsize = 30, yticklabelsize = 30
    )


    marker_num = 10
    for (idx, current_data) in enumerate(data)
        xt = current_data["xt"]
        yt = current_data["yt"]
        primal = evaluate_primal(xt, yt, xstar, ystar, max_iterations)
        data[idx]["primal"] = primal 

        t = string(current_data["nu"])
        labels = L"$\nu$ = %$(t)"

        xcoor, ycoor = get_markers(primal, marker_num)

        scatterlines!(xcoor, ycoor, markersize = 16, color =(colors[idx], 0.001),marker=markers[idx],markercolor = colors[idx])
        scatterlines!(xcoor[1], ycoor[1], markersize = 16,color =colors[idx],marker=markers[idx],markercolor = colors[idx], label = labels)
        lines!(only_min(primal),linewidth = 2, color = colors[idx])
    end 
    if log10(determine_y_lims(data)[1]) < -9
        ax.yticks = [10.0^(3k) for k in -3:1]
        ax.ytickformat = x -> [L"10^{%$(3x)}" for x in -3:1]
    else
        ax.yticks = [10.0^(2k+1) for k in -5:1]
        ax.ytickformat = x -> [L"10^{%$(2x+1)}" for x in -5:1]
    end
    axislegend(labelsize = 23, framecolor =:grey60, bgcolor=(:white, 0.4), position = :rt)
    # savefig("strcvx_intersect.png")
    return fig
end



function plot_finite_convergence(data, flag::String)
    """A function generating a 2d plot with trajectories of two polytopes."""

    v1 = [[1.,1],[-1.,1],[-1,-1],[1,-1.]]
    v2 = [[1.,0],[-1,0.],[0,1.],[0,-1.]] .+ [[1.,3]]
 
    p1 = polyhedron(vrep(v1))
    p2 = polyhedron(vrep(v2))
    xstar, ystar = find_star(data);
 
    Plots.plot(size=(300, 400), framestyle = :box)
    Plots.plot!(p1, color=:blue, alpha=0.15); Plots.plot!(p2, color=:red, alpha=0.15)
    Plots.xlims!(-1.5, 2.5); Plots.ylims!(-1.5, 4.5)
    xt = data[1]["xt"]; x0 = xt[1]; y0 = data[1]["yt"][1];
    xxt = reduce(hcat, xt)[1,:]; yxt = reduce(hcat, xt)[2,:];
    if flag != "one_step"
       for i in 1:3
          Plots.plot!(xxt[i:i+1],yxt[i:i+1],arrow=true,color=:blue,linewidth=2)
       end
       Plots.plot!([xxt[4],xstar[1]],[yxt[4],xstar[2]],arrow=true,color=:blue,linewidth=2)
       Plots.scatter!(xxt[1:4],yxt[1:4], color=:black)
       Plots.annotate!(xxt[1:4],yxt[1:4].+.4, [L"\mathbf{x_0}",L"\mathbf{x_1}",L"\mathbf{x_2}",L"\mathbf{x_3}"], :top)
    else 
       Plots.plot!([x0[1],xstar[1]],[x0[2],xstar[2]],arrow=true,color=:blue,linewidth=2)
    end
    
    Plots.scatter!([x0[1]],[x0[2]], color=:black)
    Plots.annotate!([x0[1]],[x0[2]].+.4, [L"\mathbf{x_0}"], :top)

    Plots.scatter!([y0[1]],[y0[2]], color=:black)
    Plots.annotate!([y0[1]+.45],[y0[2]], [L"\mathbf{y_0}"], :right)
    
    Plots.scatter!([xstar[1]],[xstar[2]], color=:blue)
    Plots.annotate!([xstar[1]+.2],[xstar[2]+.4], L"\mathbf{x}*", :top)

    Plots.scatter!([ystar[1]],[ystar[2]], color=:red)
    Plots.annotate!([ystar[1]+.2],[ystar[2]+.5], L"\mathbf{y}*", :top)

    Plots.plot!([y0[1],ystar[1]], [y0[2],ystar[2]], arrow=true, color=:red, linewidth=2)
 end