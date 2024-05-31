function [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = rosenbrock_problem(plot_bool)
    
    n_features = ['x0'; 'x1'];
    n_targets = ['y'];
    % Define the objective function (Rosenbrock function)

    fun_name = "rosenbruck"
    eval_fun = @(x) (x(2) + 2*sin(x(1)*2*pi) - (2*x(1))^2)^2 + (2.1 - x(1)*cos(x(1)*pi))^2;
    upper_bound = [1, 1];
    lower_bound = [-1, -1];
    % Plot the function surface
    if plot_bool
        % Define the range for plotting
        x1_range = linspace(-1, 1, 100);
        x2_range = linspace(-1, 1, 100);
        
        % Generate a grid of points for plotting
        [X1, X2] = meshgrid(x1_range, x2_range);
        Z = zeros(size(X1));
        
        % Compute function values at each grid point
        for i = 1:size(X1, 1)
            for j = 1:size(X1, 2)
                Z(i, j) = eval_fun([X1(i, j), X2(i, j)]);
            end
        end
    

        figure;
        surf(X1, X2, Z);
        xlabel('x1');
        ylabel('x2');
        zlabel('f(x)');
        title('Rosenbrock Function');
        drawnow;
    end

end