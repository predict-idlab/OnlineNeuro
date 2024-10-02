function [eval_fun, features, n_targets] = multiobjective_problem(varargin)
    % Veldhuizen and Lamont multiobjective
    p = inputParser;
    addOptional(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});
    plot_bool = p.Results.plot_bool;

    n_targets = ['y0'; 'y1'];

    % Define the objective function (Rosenbrock function)
    eval_fun = @(x)vlmop2(x);

    features = struct('x0',[-2,2],'x1',[-2,2]);
    % Plot the function surface 
    if plot_bool
        % Define the range for plotting
        x1_range = linspace(features.x0(1), features.x0(2), 100);
        x2_range = linspace(features.x1(1), features.x1(2), 100);

        % Generate a grid of points for plotting
        [X1, X2] = meshgrid(x1_range, x2_range);
        Z0 = zeros(size(X1));
        Z1 = zeros(size(X1));
        % Compute function values at each grid point
        %
        tiledlayout(2,1)

        for i = 1:size(X1, 1)
            for j = 1:size(X1, 2)
                [Z0(i, j), Z1(i,j)] = eval_fun([X1(i, j), X2(i, j)]);
            end
        end

        t = tiledlayout(1,2)
        nexttile

        surf(X1, X2, Z0);
        xlabel('x1');
        ylabel('x2');
        zlabel('Objective one');
        nexttile

        surf(X1, X2, Z1);
        xlabel('x1');
        ylabel('x2');
        zlabel('Objective two');

        title(t,'VL Multiobjective');
        set(gcf,'Position',[100 100 500 500])

        drawnow;
    end

end

function [y0, y1] = vlmop2(x)
%Veldhuizen and Lamont multiobjective
    transl = 1 / sqrt(2);
    part1 = (x(1) - transl).^2 + (x(2) - transl).^2;
    part2 = (x(1) + transl).^2 + (x(2) + transl).^2;
    y0 = 1 - exp(-1 * part1);
    y1 = 1 - exp(-1 * part2);
    
end