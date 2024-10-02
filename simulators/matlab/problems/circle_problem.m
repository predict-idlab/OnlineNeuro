function [eval_fun, features, n_targets] = circle_problem(varargin)
   % Create an input parser object
    p = inputParser;
    
    % Define the parameters and their default values
    addOptional(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    addOptional(p, 'radius', 0.5, @(x) isnumeric(x) && x > 0);
    addOptional(p, 'noise', 0, @(x) isnumeric(x) && x >= 0);
    addOptional(p, 'center', [0, 0], @(x) isnumeric(x) && numel(x) == 2);
    
    % Parse the input arguments
    parse(p, varargin{:});
    
    % Retrieve values after parsing
    plot_bool = p.Results.plot_bool;
    radius = p.Results.radius;
    noise = p.Results.noise;
    center = p.Results.center;

    n_targets = ['y'];
    features = struct('x0',[-1, 1], 'x1',[-1, 1])

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