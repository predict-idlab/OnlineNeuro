function [eval_fun, eval_dict] = axon_problem(experiment_params, varargin)
    % Sets up an evaluation function for an AxonSim problem.
    %
    % Inputs:
    %   experiment_params: A struct containing all parameters for the axon simulation.
    %   varargin: Optional name-value pairs:
    %     'plot'        - (logical) A flag for plotting (currently unused). Default: false.
    %     'path_config' - (string|struct) Path to path_config.json or a pre-loaded struct. Default: 'config.json'.

    % 1. Validate the required input
    if ~isstruct(experiment_params)
        error('The first argument "experiment_params" must be a struct containing the problem configuration.');
    end

    % 2 Create an input parser object
    p = inputParser;
    addParameter(p, 'plot', false, @(x) islogical(x) || isscalar(x));
    addParameter(p, 'path_config', 'config.json', @(x) ischar(x) || isstring(x) || isstruct(x));
    parse(p, varargin{:});
    % Store parsed optional arguments in a single struct
    opts = p.Results;

    % 3. Load path configuration from its source (file or struct)
    try
        path_params = load_config_from_source(opts.path_config);
    catch ME
        error('Could not load path configuration: %s', ME.message);
    end

    % 4. Set up the environment
    % Adding axonsim and all its folders to path
    if isfield(path_params, 'path_config') && isfield(path_params.path_config, 'axonsim_path')
        addpath(genpath(path_params.path_config.axonsim_path));
    else
        warning('axonsim_path not found in path configuration. Ensure it is already in the MATLAB path.');
    end

    % --- Main Function Logic ---

    % The evaluation function handle is static
    eval_fun = @axonsim_call;

    % 5. Transform the input parameter struct into the flat format needed by the simulation
    % This isolates the complex transformation logic into a helper function.
    eval_dict = flatten_param_struct(experiment_params);

    % Post-processing: Handle specific data type conversions if necessary
    if isfield(eval_dict, 'num_electrodes') && (isstring(eval_dict.num_electrodes) || ischar(eval_dict.num_electrodes))
        eval_dict.num_electrodes = str2double(eval_dict.num_electrodes);
    end

    % Handle the 'plot' option
    if opts.plot
        disp("Note: AxonSim problem type does not currently support response surface plotting.");
    end
end

% --- Helper Functions ---
function flat_struct = flatten_param_struct(s)
    % Transforms a nested parameter struct into a flat one.
    % It checks if a field is a struct containing a '.value' field, and if so,
    % it "unpacks" it. Otherwise, it copies the value directly.
    flat_struct = struct();
    fields = fieldnames(s);

    for i = 1:length(fields)
        fieldName = fields{i};
        fieldValue = s.(fieldName);

        % Check if the field is a struct and has a 'value' subfield
        if isstruct(fieldValue) && isfield(fieldValue, 'value')
            flat_struct.(fieldName) = fieldValue.value;
        else
            % Otherwise, just copy the value as is
            flat_struct.(fieldName) = fieldValue;
        end
    end
end

function config_struct = load_config_from_source(source)
    % Loads a configuration from a source, which can be a struct or a file path.
    if isstruct(source)
        config_struct = source; % It's already a struct, just return it
        return;
    elseif ischar(source) || isstring(source)
        if ~exist(source, 'file')
            error('Configuration file does not exist: %s', source);
        end
        config_struct = jsondecode(fileread(source));
    else
        error('Invalid configuration source type. Must be a struct or a file path string.');
    end
end
