
function run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size)
    """
    Collect all data into a dictionary.


    Args:
    x0: Vector{Float64}
        initial point of the region (P, belongs to lmo1).
    y0: Vector{Float64}
        initial point of the region (Q, belongs to lmo2).
    lmo1: 
        linear minimization oracle of the region P.
    lmo2: 
        linear minimization oracle of the region Q.
    max_iterations: Int
        the maximum number of the iterations.
    f:
        objective_function, 1/2 ||x-y||^2_2.
    step_size: Dict
        A dictionary containing the information about the step type. The dictionary can have the following arguments:
            "step type": Choose from "open-loop", "line-search".
        Additional Arguments:
            For "open-loop", provide integer values for the keys "ell" that affect the step type as follows: ell / (iteration + ell).

    Returns:
    data: Vector{Dict}
        An array of dictionaries containing the information of ALM. Each elements of the array is the output of ALM with a step-size.
        For example:
            data = [[Dict{"vt"=>[..],"ut"=>[...], "primal"=>[...], "step_type"=>"open_loop", ...}], ... ]
    """
    data = []
    for step in step_size
        label = step["step_type"] *"_"*string(step["ell"])
        println("now: ", label)
        init = Dict()
        init["step_type"] = step["step_type"]
        init["ell"] = step["ell"]
        init["xt"], init["yt"], init["loss"], init["ut"], init["vt"] = Alternating_linear_minimizations(x0, y0, lmo1, lmo2, max_iterations, f, step)
        push!(data, init)
        println("done")
    end
    return data
end