function [eval_fun, features, n_targets] = circle_problem(varargin)
   % Create an input parser object
    p = inputParser;

    % Define the parameters and their default values
    addOptional(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    addOptional(p, 'problem_setting', false, @(x) isstruct(x));

    % Parse the input arguments
    parse(p, varargin{:});

    % Retrieve values after parsing
    plot_bool = p.Results.plot;
    problem_setting = p.Results.problem_setting;

    n_targets = 'y';
    %Default values

    x0=[-1,1];
    x1=[-1,1];
    radius=0.5;
    noise=0.1;
    center= [0,0];

    missing_vars = [];
    spec_vars = {'x0','x1','radius','noise','center'};

    if problem_setting
        for i = 1:length(spec_vars)
            if ~isfield(problem_setting, spec_vars{i})
                missing_vars = [missing_vars, spec_vars{i}];
            end
        end
        if ~isempty(missing_vars)
            warning(['The following variables are missing: ', strjoin(missing_vars, ', '), '. Using default values.']);
        end
        if isfield(problem_setting, 'x0')
            x0 = problem_setting.x0;
        end
        if isfield(problem_setting, 'x1')
            x1 = problem_setting.x1;
        end
        if isfield(problem_setting, 'radius')
            radius = problem_setting.radius;
        end
        if isfield(problem_setting, 'noise')
            noise = problem_setting.noise;
        end
        if isfield(problem_setting, 'center')
            center = problem_setting.center;
        end
    else
        warning('No problem configuration was passed, using default values');
    end

    features = struct('x0', x0, 'x1', x1);

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
