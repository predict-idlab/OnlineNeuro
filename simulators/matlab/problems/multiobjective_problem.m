function [eval_fun, features, n_targets] = multiobjective_problem(varargin)
    % A function defining the Veldhuizen and Lamont (VLMOP2) multiobjective problem.
    % It accepts name-value pairs for configuration.

    % 1. Create and configure the input parser
    p = inputParser;

    % Define all parameters with their default values and validation functions
    addParameter(p, 'plot', false, @(x) islogical(x) || isscalar(x));
    addParameter(p, 'x0', [-2, 2], @(x) isnumeric(x) && numel(x)==2);
    addParameter(p, 'x1', [-2, 2], @(x) isnumeric(x) && numel(x)==2);

    % 2. Parse the input arguments
    % This will validate inputs and fill in any missing ones with defaults.
    parse(p, varargin{:});

    % 3. Retrieve all parameters from the parser's Results struct
    params = p.Results;

    % (Optional but helpful) Display the final parameters used
    disp('Multiobjective Problem (VLMOP2) configured with parameters:');
    disp(params);

    % --- Main Function Logic ---

    n_targets = {'y0'; 'y1'}; % Using cell array is more standard for string lists

    % The evaluation function itself doesn't depend on parameters here
    eval_fun = @(x) vlmop2(x);

    % Define the feature space based on the parsed parameters
    features = struct('x0', params.x0, 'x1', params.x1);

    % Plot the function surfaces if requested
    if params.plot
        % Define the range for plotting from the parsed parameters
        x0_range = linspace(params.x0(1), params.x0(2), 50); % 50 points is often enough
        x1_range = linspace(params.x1(1), params.x1(2), 50);

        % Generate a grid of points for plotting
        [X1, X2] = meshgrid(x0_range, x1_range);

        % --- Vectorized Calculation (much faster than loops) ---
        grid_points = [X1(:), X2(:)];      % Create a N-by-2 matrix of all points
        Y_all = vlmop2(grid_points);       % Call the function ONCE on all points
        Z0 = reshape(Y_all(:,1), size(X1)); % Reshape the first objective
        Z1 = reshape(Y_all(:,2), size(X1)); % Reshape the second objective
        % --- End of Vectorized Calculation ---

        figure;
        t = tiledlayout(1, 2, 'TileSpacing', 'compact');
        title(t, 'VLMOP2 Objectives');

        % Plot for the first objective
        nexttile;
        surf(X1, X2, Z0);
        xlabel('x0');
        ylabel('x1');
        zlabel('Objective 1 (y0)');
        title('Objective 1');
        colorbar;

        % Plot for the second objective
        nexttile;
        surf(X1, X2, Z1);
        xlabel('x0');
        ylabel('x1');
        zlabel('Objective 2 (y1)');
        title('Objective 2');
        colorbar;

        set(gcf, 'Position', [100 100 800 400]); % Adjust figure size
        drawnow;
    end
end

function y = vlmop2(x)
    % Veldhuizen and Lamont multiobjective problem (VLMOP2).
    % This function is already vectorized to handle multiple input rows.
    if isstruct(x)
        x = [x(:).x0, x(:).x1];
    end

    % Ensure x is a matrix for the following operations
    if isvector(x)
        x = x(:)'; % Force to be a row vector if a single point is passed
    end

    transl = 1 / sqrt(2);
    part1 = (x(:,1) - transl).^2 + (x(:,2) - transl).^2;
    part2 = (x(:,1) + transl).^2 + (x(:,2) + transl).^2;

    y0 = 1 - exp(-1 * part1);
    y1 = 1 - exp(-1 * part2);

    y = [y0, y1];
end
