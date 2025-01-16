function [eval_fun, eval_dict] = axon_problem(problem_setting, varargin)
    % Create an input parser object
    p = inputParser;
    p.KeepUnmatched = true;  % Optional, to allow unspecified name-value pairs
    p.PartialMatching = false;  % To avoid ambiguities in argument names

    % Define the parameters and their default values
    addRequired(p, 'problem_setting', @(x) ischar(x) || isstring(x) || isstruct(x));
    addParameter(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'path_config', 'config.json', @(x) ischar(x) || isstring(x) || isstruct(x));
    % Parse the input arguments
    parse(p, problem_setting, varargin{:});

    % Retrieve values after parsing
    plot = p.Results.plot;
    problem_setting = p.Results.problem_setting;
    path_setting = p.Results.path_config;
    
    % If problem_setting is a string, interpret it as a JSON file path and load the config
    if ischar(problem_setting) || isstring(problem_setting)
        try
            display("Loading problem configuration")
            experiment_params = jsondecode(fileread(problem_setting));
        catch
            error("Configuration file could not be loaded.");
        end
    elseif isstruct(problem_setting)% If problem_setting is already a struct, use it directly
        display("Valid configuration, proceeding")
        experiment_params = problem_setting;
    else
        error('Invalid type for problem_setting. It must be either a string (filename) or a struct.');
    end

    if ischar(path_setting) || isstring(path_setting)
        try
            path_params = jsondecode(fileread(path_setting));
        catch
            error("Configuration file could not be loaded.");
        end
    elseif isstruct(path_setting)% If path_setting is already a struct, use it directly
        path_params = path_setting;
    else
        error('Invalid type for path_setting. It must be either a string (filename) or a struct.');
    end
    %Adding axonsim and all its folders to path
    addpath(genpath(path_params.path_config.axonsim_path))

    % Plot the function surface
    if plot
        disp("AxonSim can't display a plot as the response surface is unknown");
    end
    eval_fun = @axonsim_call;
    
    fields = fieldnames(experiment_params);
    eval_dict = struct();

    %This is just to allow the code to be debugged. In general num of
    %electrodes needs to be specified!
    array_size = experiment_params.('num_electrodes');

    if isstring(array_size)
        array_size = str2double(array_size);
    end

    for i = 1:length(fields)
        % Fixed and included in the experiment_params
        if isfield(experiment_params.(fields{i}), 'value')
            eval_dict.(fields{i}) = experiment_params.(fields{i}).value;
            % If it doesn't contain a value then is not part of the problem requirements i.e. booleans passed from the GUI
            %eval_dict.(fields{i}) = value;
        end
    end
end
