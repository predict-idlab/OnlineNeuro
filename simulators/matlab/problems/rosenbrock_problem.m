function [eval_fun, features, n_targets] = rosenbrock_problem(varargin)
    % Define the objective function (Rosenbrock function)
    % 1. Create and configure the input parser
    p = inputParser;
    p.KeepUnmatched = true;

    % Define all parameters with their default values and validation functions
    addParameter(p, 'plot', false, @(x) islogical(x) || isscalar(x));
    addParameter(p, 'a', 1, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'b', 100, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'x0', [-2, 2], @(x) isnumeric(x) && numel(x)==2);
    addParameter(p, 'x1', [-1, 3], @(x) isnumeric(x) && numel(x)==2);

    % 2. Parse the input arguments
    % This will validate inputs and fill in any missing ones with defaults.
    parse(p, varargin{:});
    params = p.Results;

    plot_bool = params.plot;
    n_targets = 'y';

    features = struct('x0', params.x0, 'x1', params.x1);
    eval_fun =  @(x) rosenbrock_fun(x, params.a, params.b);

    % Plot the function surface
    if plot_bool
        % Define the range for plotting
        x1_range = linspace(params.x0(1), params.x0(2), 100);
        x2_range = linspace(params.x1(1), params.x1(2), 100);

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
        title(sprintf('Rosenbrock Function (a=%.1f, b=%.1f)', params.a, params.b));
        drawnow;
    end

end

function result = rosenbrock_fun(x, a, b)
    if isstruct(x)
        x = [x.x0, x.x1];
    end
    result = (a - x(:,1)).^2 + b * (x(:,2) - x(:,1).^2).^2;

end
