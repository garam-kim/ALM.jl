using FrankWolfe
using LinearAlgebra

function Alternating_linear_minimizations(x0, y0, lmo1, lmo2, max_iterations, f, step)
   """Performs alternating linear minimizations.

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
      step: dict
         A dictionary containing the information about the step type. The dictionary can have the following arguments:
            "step type": Choose from "open-loop", "line-search".
         Additional Arguments:
            For "open-loop", provide integer values for the keys "ell" that affect the step type as follows: ell / (iteration + ell).

   Returns:
      xt: list
         Returns a list containing the iterate of x in P at each iteration.
      yt: list
         Returns a list containing the iterate of y in Q at each iteration.
      loss: list
         Returns a list containing the loss at each iteration.
      ut: list
         Returns a list containing the frank-wolfe vertex u in P at each iteration.
      vt: list
         Returns a list containing the frank-wolfe vertex v in Q at each iteration.
   """

   # Initialize empty lists to store the loss, x points, and y points
   loss = Float64[];
   xt = []; yt = []; ut = []; vt = [];

   x = copy(x0); y = copy(y0);

   if step["step_type"] == "open_loop"
      ell = step["ell"]
      t = 0; dual_gap = [Inf, Inf]; 
      while t <= max_iterations && sum(dual_gap) >= max(1e-10, eps(float(typeof(sum(dual_gap)))))

         # Store the current loss, x and y points
         push!(loss, f(x,y)); push!(xt, x); push!(yt, y); 
         eta = ell/(t + ell);

         u = FrankWolfe.compute_extreme_point(lmo1, x - y);
         dual_gap[1] = dot(x, x - y) - dot(u, x - y);
         if dual_gap[1] > 1e-10 x += eta * (u - x) end

         v = FrankWolfe.compute_extreme_point(lmo2, y - x);
         dual_gap[2] = dot(y, y - x) - dot(v, y - x);
         if dual_gap[2] > 1e-10 y += eta * (v - y) end

         push!(ut, u); push!(vt, v);
         t+= 1
      end
   elseif step["step_type"] == "line_search"
      t = 0; dual_gap = [Inf, Inf]; 
      while t <= max_iterations && sum(dual_gap) >= max(1e-10, eps(float(typeof(sum(dual_gap)))))
         # Store the current loss, x and y points
         push!(loss, f(x,y)); push!(xt, x); push!(yt, y); 

         u = FrankWolfe.compute_extreme_point(lmo1, x - y); 
         dual_gap[1] = dot(x, x - y) - dot(u, x - y);
         if dual_gap[1] > 1e-10
            eta_P = min(max(dot(x - y, x - u)/dot(u - x, u - x), 0), 1);
            x += eta_P * (u - x);
         end

         v = FrankWolfe.compute_extreme_point(lmo2, y - x);
         dual_gap[2] = dot(y, y - x) - dot(v, y - x);
         if dual_gap[2] > 1e-10 
            eta_Q = min(max(dot(y - x, y - v)/dot(v-y, v-y), 0), 1);
            y += eta_Q * (v - y);
         end
         push!(ut, u); push!(vt, v);
         t+= 1
      end
   end
   return xt, yt, loss, ut, vt
end