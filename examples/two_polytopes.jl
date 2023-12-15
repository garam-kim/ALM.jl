include("../src/experiments_auxiliary_functions.jl")
include("../src/experimental_settings.jl")


##############################
# Experiments
##############################


# Initial setting
n = 100;  # Dimensions
max_iterations = 10^5+200; 

step_size = [Dict("step_type" => "open-loop", "ell" => ell) for ell in [2]]



# Compares the number of iterates required to converge for different initializations.
for flag in ["one_step", "finite_step"]
   lmo1, lmo2, x0, y0 = build_poly_settings(flag)
   data = run_experiment(x0, y0, lmo1, lmo2, max_iterations, f, step_size);
   fig = plot_finite_convergence(data, flag)
   filename = flag * ".png";
   Plots.savefig(fig, "../figures/"*filename)
end