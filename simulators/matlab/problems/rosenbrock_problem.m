function [eval_fun, features, n_targets] = rosenbrock_problem(varargin)
    % Define the objective function (Modified Rosenbrock function)

    p = inputParser;

    % Define the parameters and their default values
    addOptional(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    addOptional(p, 'problem_setting', false, @(x) isstruct(x));

    % Parse the input arguments
    parse(p, varargin{:});
    plot_bool = p.Results.plot;
    problem_setting = p.Results.problem_setting;

    n_targets = 'y';
    %Default values
    a = 1;
    b = 100;
    x0 = [-5, 5];
    x1 = [-5, 5];

    missing_vars = [];
    spec_vars = {'a','b','x0','x1'};

    if problem_setting
       for i = 1:length(spec_vars)
           if ~isfield(problem_setting, spec_vars{i})
               missing_vars = [missing_vars, spec_vars{i}];
           end
       end
       if ~isempty(missing_vars)
           warning(['The following variables are missing: ', strjoin(missing_vars, ', '), '. Using default values.']);
       end

       if isfield(problem_setting, 'a')
           a = problem_setting.a;
       end
       if isfield(problem_setting, 'b')
           b = problem_setting.b;
       end
       if isfield(problem_setting, 'x0')
           x0 = problem_setting.x0;
       end
       if isfield(problem_setting, 'x1')
           x0 = problem_setting.x1;
       end
    else
        warning('No problem configuration was passed, using default values');
    end
    features = struct('x0',x0,'x1',x1);
    eval_fun =  @(x) rosenbrock_fun(x, a, b);

    % Plot the function surface
    if plot_bool
        % Define the range for plotting
        x1_range = linspace(-5, 5, 100);
        x2_range = linspace(-5, 5, 100);

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

function result = rosenbrock_fun(x, a, b)
    if isstruct(x)
        x = [x.x0, x.x1];
    end
    result = (a - x(:,1)).^2 + b * (x(:,2) - x(:,1).^2).^2;

end
