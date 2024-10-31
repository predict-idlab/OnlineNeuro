function [eval_fun, eval_dict] = axon_problem(problem_setting, varargin)
    % Create an input parser object
    p = inputParser;
    p.KeepUnmatched = true;  % Optional, to allow unspecified name-value pairs
    p.PartialMatching = false;  % To avoid ambiguities in argument names

    % Define the parameters and their default values
    addRequired(p, 'problem_setting', @(x) ischar(x) || isstring(x) || isstruct(x));

    addParameter(p, 'plot', false, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'default_setting', 'config.json', @(x) ischar(x) || isstring(x) || isstruct(x));
    addParameter(p, 'path_config', 'config.json', @(x) ischar(x) || isstring(x) || isstruct(x));
    % Parse the input arguments
    parse(p, problem_setting, varargin{:});

    % Retrieve values after parsing
    plot = p.Results.plot;
    problem_setting = p.Results.problem_setting;
    default_setting = p.Results.default_setting;
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

    if ischar(default_setting) || isstring(default_setting)
        try
            default_params = jsondecode(fileread(default_setting));
        catch
            warning("Configuration file could not be loaded, using default ('config_experiments/axonsim_template.json').");
            default_params = jsondecode(fileread('config_experiments/axonsim_template.json'));
        end
    elseif isstruct(default_setting)
        default_params = default_setting;
    else
        error('Invalid type for default_setting. It must be either a string (filename) or a struct.');
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
    
    fields_default = fieldnames(default_params);
    fields_experiment = fieldnames(experiment_params);
    fields = unique([fields_default; fields_experiment]);

    eval_dict = struct();

    %This is just to allow the code to be debugged. In general num of
    %electrodes needs to be specified!
    if isfield(experiment_params,'num_electrodes')
        array_size = experiment_params.('num_electrodes').value;
    else
        array_size = default_params.('num_electrodes').value;
    end
    
    if isstring(array_size)
        array_size = str2double(array_size);
    end

    for i = 1:length(fields)
        %Populating inputs of Axonsim, if field not in dict, then use
        %default (axonsim.template).  Attention that some default values
        %may not suit the expected experiment. Have a look at the template
        %to make sure is the desired experiment (This will be shown in the
        %GUI later, but can't be seen if running from terminal)
        if isfield(experiment_params, fields{i})
            % Fixed and included in the experiment_params
            if isfield(experiment_params.(fields{i}), 'value')
                eval_dict.(fields{i}) = experiment_params.(fields{i}).value;
            else 
                %Placeholders. Non fixed values are replaced later.
                if isfield(experiment_params, fields{i})
                    %type = experiment_params.(fields{i}).type;
                    value = experiment_params.(fields{i}).min_value;
                elseif isfield(default_params, fields{i})
                    %type = default_params.(fields{i}).type;
                    value = default_params.(fields{i}).min_value;
                else
                    error("Parameters not specified for variable %s",fields{i})
                end
                
                eval_dict.(fields{i}) = value;

                % if endsWith(type, 'mult')
                %     default_array = double.empty(0, array_size);
                %     for j = 1:array_size
                %         default_array{j} = value;
                %     end
                %     eval_dict.(fields{i}) = default_array;
                % elseif endsWith(type, 'complex')
                %     default_array = strings(array_size,1);
                %     for j = 1:array_size
                %         default_array{j} = value;
                %     end
                %     eval_dict.(fields{i}) = default_array;
                % else
                % end
            end
        else
            % Load defaults
            if default_params.(fields{i}).optimizable
                error("Optimizable params don't have default ranges")
            else
                % array (populate with default values)
                if endsWith(default_params.(fields{i}).type, 'mult')
                    default_array = double.empty(0, array_size);
                    for j = 1:array_size
                        default_array{j} = default_params.(fields{i}).value;
                    end
                    eval_dict.(fields{i}) = default_array;
                % Not an array, single value
                else
                    eval_dict.(fields{i}) = default_params.(fields{i}).value;
                end

            end
        end
    end
end
