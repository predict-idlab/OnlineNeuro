function [eval_fun, features, n_targets] = circle_problem(varargin)
   % A function that defines a (noisy) circle classification problem.
   % It accepts name-value pairs for configuration.

   % Create an input parser object
    p = inputParser;
    p.KeepUnmatched = true;

    % Define all parameters with their default values and validation functions
    addParameter(p, 'plot', false, @(x) islogical(x) || isscalar(x));
    addParameter(p, 'x0', [-1, 1], @(x) isnumeric(x) && numel(x)==2);
    addParameter(p, 'x1', [-1, 1], @(x) isnumeric(x) && numel(x)==2);
    addParameter(p, 'radius', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'noise', 0.0, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'center', [0, 0], @(x) isnumeric(x) && numel(x)==2);

    % Parse the input arguments and fill in any missing ones with defaults.
    parse(p, varargin{:});

    % Retrieve values after parsing
    params = p.Results;

    n_targets = 'y';
    features = struct('x0', params.x0, 'x1', params.x1);

    % Define the evaluation function handle using the parsed parameters
    eval_fun = @(x) calculateCircle(x, params.radius, params.noise, params.center);

    % Plot the function surface if requested
    if params.plot
        % Define the range for plotting
        x1_range = linspace(params.x0(1), params.x0(2), 100);
        x2_range = linspace(params.x1(1), params.x1(2), 100);

        % Generate a grid of points
        [X1, X2] = meshgrid(x1_range, x2_range);
        Z = zeros(size(X1));

        % Compute function values at each grid point
        for i = 1:size(X1, 1)
            for j = 1:size(X1, 2)
                Z(i, j) = eval_fun([X1(i, j), X2(i, j)]);
            end
        end

        figure;
        % Use contourf for a 2D classification problem view
        contourf(X1, X2, Z, [0.5 0.5]);
        colormap([0.8 0.8 1; 1 0.8 0.8]); % Blue for inside, Red for outside
        hold on;

        % Plot the "true" circle boundary
        theta = linspace(0, 2*pi, 200);
        xc = params.center(1) + params.radius * cos(theta);
        yc = params.center(2) + params.radius * sin(theta);
        plot(xc, yc, 'k--', 'LineWidth', 2);

        axis equal;
        xlabel('x0');
        ylabel('x1');
        title(sprintf('Circle Problem (Radius=%.2f, Noise=%.2f)', params.radius, params.noise));
        legend('Classification Regions', 'True Boundary');
        grid on;
        drawnow;
    end
end

function result = calculateCircle(x, radius, noise, center)
    if isstruct(x)
        x = [x(:).x0, x(:).x1];
    end
    x_centered = x - center;
    radii = sqrt(sum(x_centered.^2, 2));
    if noise > 0
        noise_vector = noise * randn(size(radii));
        result = double(radii + noise_vector > radius);
    else
        result = double(radii > radius);

    end
end
