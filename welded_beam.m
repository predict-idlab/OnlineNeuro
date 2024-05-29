function [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = welded_beam(plot_bool)
    %Based on Wedled beam problem
    % Goal is to demonstrate the use of non linear constraints.
    % https://nl.mathworks.com/help/gads/multiobjective-optimization-welded-beam.html
    % 4 design varibles.
    % 2 objectives to be optimized (Fabrication cost and )

    n_features = ['x1','x2','x3','x4'];
    n_targets = ['y1', 'y2'];

    fun_name = "welded_beam"

    %eval_fun = @(x) (x(2) + 2*sin(x(1)*2*pi) - (2*x(1))^2)^2 + (2.1 - x(1)*cos(x(1)*pi))^2;
    upper_bound = [5, 10, 10, 5];
    lower_bound = [0.125, 0.1, 0.1, 0.125];
    % Plot the function surface 
    if plot_bool
        disp("AxonSim can't display a plot as the response surface is unknown");
    end
    eval_fun = @objval;
    % TODO check how to pass constraints to Python end?
    % Easiest way would be to specify this in a shared file, but it may
    % break the Matlab/Python explicity problem handling/optimizer

end

function [Cineq, Ceq] = nonlcon(x1, x2, x3, x4)
    sigma = 5.04e5 ./ (x3.^2 .* x4);
    P_c = 64746.022*(1 - 0.028236*x3).*x3.*x4.^3;
    tp = 6e3./sqrt(2)./(x1.*x2);
    tpp = 6e3./sqrt(2) .* (14+0.5*x2).*sqrt(0.25*(x2.^2 + (x1 + x3).^2)) ./ (x1.*x2.*(x2.^2 / 12 + 0.25*(x1 + x3).^2));
    tau = sqrt(tp.^2 + tpp.^2 + (x2.*tp.*tpp)./sqrt(0.25*(x2.^2 + (x1 + x3).^2)));
    Cineq = [tau - 13600, sigma - 3e4, 6e3 - P_c];
    Ceq = [];
end
    
function F = objval(x1, x2, x3, x4)
    f1 = 1.10471*x1.^2.*x2 + 0.04811*x3.*x4.*(14.0+x2);
    f2 = 2.1952./(x3.^3 .* x4);
    
    F = [f1, f2];
end

function z = pickindex(x,k)
    z = objval(x); % evaluate both objectives
    z = z(k); % return objective k
end
