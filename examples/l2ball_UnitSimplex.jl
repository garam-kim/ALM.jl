include("../src/experiments_auxiliary_functions.jl")
include("../src/experimental_settings.jl")


##############################
# Experiments.
##############################


# Initial setting
n = 100;  # Dimensions.
max_iterations = 10^5+200; 

step_size = [Dict("step_type" => "open-loop", "ell" => ell) for ell in [2, 5, 8]]
push!(step_size, Dict("step_type" => "line-search", "ell"=>""))

nu = [1, 0.8, 0.5, 0.2]; step = Dict("step_type" => "open-loop", "ell" => 2);



# Compares the performance of different step-size rules and approximation errors for the relation between two sets.
for loc in ["disjoint", "touch"]
   lmo1, lmo2, x0, y0 = build_l2_unitsimplex_settings(n, loc);
   filename = "l2ball_UnitSimplex_"*string(loc)*".png";

   data = run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size);
   xstar, ystar = find_star(data);

   fig = plotting_function(data)
   CairoMakie.save("../figures/"*filename, fig)

   data_ablation = run_ablation(x0, y0, lmo1, lmo2, max_iterations, f, step, nu);
   fig = plotting_ablation_function(data_ablation, xstar, ystar)
   CairoMakie.save("../figures_ablation/"*filename, fig)
end
