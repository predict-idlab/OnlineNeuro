function [eval_fun, features, n_targets] = rosenbrock_problem(varargin)
    % Define the objective function (Modified Rosenbrock function)

    p = inputParser;

    % Define the parameters and their default values
    addOptional(p, 'plot', false, @(x) islogical(x) || isnumeric(x));

    % Parse the input arguments
    parse(p, varargin{:});
    plot_bool = p.Results.plot_bool;

    n_targets = ['y'];

    %a =1 , b=100
    eval_fun =  @(x) (1 - x(:,1)).^2 + 100 * (x(:,2) - x(:,1).^2).^2;

    features = struct('x0',[-5, 10],'x1',[-5, 10]);

    % Plot the function surface
    if plot_bool
        % Define the range for plotting
        x1_range = linspace(-5, 10, 100);
        x2_range = linspace(-5, 10, 100);
        
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