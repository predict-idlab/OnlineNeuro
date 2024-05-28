function [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = circle_problem(plot_bool, radius, noise, center)
    
    if nargin < 4
        center = [0, 0];
    end
    if nargin < 3
        noise = 0;
    end
    if nargin < 2
        radius = 0.5;
    end
    if nargin < 1
        plot_bool=false;
    end
    
    fun_name = "circle";
    n_features = ['x0','x1'];
    n_targets = ['y'];

    upper_bound = [1, 1];
    lower_bound = [-1, -1];

    % Define the objective function (Rosenbrock function)
    eval_fun = @(x) calculateCircle(x, radius, noise, center);


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
        title('Circle Function');
        drawnow;
    end

end

function result = calculateCircle(x, radius, noise, center)
    x_centered = x - center;
    radii = sqrt(sum(x_centered.^2, 2));
    if noise > 0
        result = double(radii - radius > 0);
    else
        noise_vector = noise * randn(size(radii));
        result = double(radii - radius + noise_vector > 0);
    end
end