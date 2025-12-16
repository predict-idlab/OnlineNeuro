% file: main.m

% IMPORTANT: Matlab comes with its on C++ libraries which are used Python.
% If using linux, hide those (or delete) as explained in
% https://nl.mathworks.com/matlabcentral/answers/1907290-how-to-manually-select-the-libstdc-library-to-use-to-resolve-a-version-glibcxx_-not-found

function main(json_data)
    format long g
    %pyenv('ExecutionMode','InProcess');

    root_path = fullfile(pwd, '../../');
    addpath(genpath(pwd))

    %display(json_data)
    data = jsondecode(json_data);

    if ~isfield(data, 'problem_config')
        error('No problem configuration provided');
    end
    if ~isfield(data, 'connection_config')
        error('No connection configuration provided');
    end

    problem_config = data.problem_config;

    if isstruct(problem_config) && isfield(problem_config, 'config_file')
        config_file = problem_config.config_file;
        % Ensure it's a string or char array
        if isstring(config_file) || ischar(config_file)
            file_path = fullfile(root_path, config_file);
            % Check if the file actually exists
            if exist(file_path, 'file')
                problem_config = jsondecode(fileread(file_path));
                problem_config.('config_file') = config_file; % Retain the config_file field
            else
                error('Config file specified in problem_config.config_file does not exist: %s', file_path);
            end
        end
    end
    display(problem_config.('experiment_parameters'))
    problem_name = problem_config.('experiment_parameters').('problem_name');
    problem_type = problem_config.('experiment_parameters').('problem_type');
    connection_config = data.('connection_config');

    max_retries = 5; % Number of times to try connecting
    retry_pause_duration = 2.0; % Seconds to wait between retries
    connection_successful = false;

    fprintf("Verifying port is open \n");
    cmd = sprintf('fuser %d/tcp', connection_config.port);
    system(cmd);

    for attempt = 1:max_retries
        tcpipClient = [];

        try
            fprintf('MATLAB: Attempt %d/%d to connect and handshake with Python server at %s:%d...\n', ...
                attempt, max_retries, connection_config.ip, connection_config.port);

            % Step 1: Connect
            tcpipClient = tcpclient(connection_config.ip, connection_config.port, ...
                "Timeout", connection_config.Timeout, "ConnectTimeout", connection_config.ConnectTimeout, ...
                 "EnableTransferDelay", false);

            fprintf('      -> Connection established.\n');
            handshake(tcpipClient)

            connection_successful = true;
            break; % Exit the loop ONLY if both connection and handshake succeed

        catch ME
            fprintf('MATLAB: Attempt %d failed. Reason: %s\n', attempt, ME.message);
            if attempt < max_retries
                fprintf('      -> Retrying in %.1f seconds...\n', retry_pause_duration);
                pause(retry_pause_duration);
            else
                fprintf(2, 'MATLAB: All attempts failed. The process will now terminate.\n');
                rethrow(ME);
            end
        end
    end

    if ~connection_successful
        error('main:SetupFailed', 'Failed to connect and handshake with Python server after %d attempts.', max_retries);
    end

    disp("Selected problem")
    disp(problem_name)
    % Prepare call function based on problem_config
    try
        fprintf('MATLAB: Waiting for initial fixed features...\n');
        receivedData = receiveMessage(tcpipClient);

        if ~isstruct(receivedData) || ~isfield(receivedData, 'Fixed_features')
            error('main:SyncError', 'Did not receive expected initial message from Python server.');
        end
        fieldnames_fixed = fieldnames(receivedData.("Fixed_features"));

        % Send acknowledgement back to Python
        fprintf('MATLAB: Sending acknowledgement to Python...\n');
        ack_message.status = 'ready';
        sendMessage(tcpipClient, ack_message);
        fprintf('MATLAB: Acknowledgement sent. Ready for query points.\n');

    catch ME
        fprintf(2, 'AN ERROR OCCURRED WHILE RECIEVING FIXED FEATURE NAMES:\n');
        disp(ME.getReport());
        return;
    end
    % 1. Get the parameters struct from the loaded config
    exp_params = problem_config.experiment_parameters;

    % 2. Convert the struct to a cell array for varargin
    paramNames = fieldnames(exp_params);
    transformed_params = struct();

    for k = 1:length(paramNames)
        name = paramNames{k};
        value = exp_params.(name); % Access the current field value

        if isstruct(value)
            % This handles fields like x0 and x1 which are nested structs

            % Ensure the required fields exist (error handling highly recommended)
            if isfield(value, 'min_value') && isfield(value, 'max_value')
                % Extract [min, max] and assign it to the new structure
                transformed_params.(name) = [value.min_value, value.max_value];
            else
                warning('Nested structure field %s missing min/max_value.', name);
                % Fallback: Assign the structure itself (may cause failure later)
                transformed_params.(name) = value;
            end
        else
            % This handles simple fields like 'a', 'b', or 'plot'
            transformed_params.(name) = value;
        end
    end
    paramNames = fieldnames(transformed_params);
    paramValues = struct2cell(transformed_params);
    argsCell = [paramNames'; paramValues'];
    argsCell = argsCell(:)';
    display(argsCell)
    switch problem_name
        case 'circle_classification'
            [eval_fun, eval_dict] = circle_problem(argsCell{:});
        case 'matlab_rose_regression'
            [eval_fun, eval_dict] = rosenbrock_problem(argsCell{:});
        case 'vlmop2'
            %TODO fix this
            [eval_fun, eval_dict] = multiobjective_problem(argsCell{:});
        case {'axonsim_single', 'axonsim_double', 'axonsim_regression','axon_threshold','axonsim_nerve_block'}
            [eval_fun, eval_dict] = axon_problem(exp_params,'plot', false);
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [eval_fun, eval_dict] = rosenbrock_problem(argsCell{:});
    end

    fieldnames_trainable = fieldnames(eval_dict);
    missingFields = setdiff(fieldnames_fixed, fieldnames_trainable);

    for i = 1:numel(missingFields)
        eval_dict.(missingFields{i}) = receivedData.("Fixed_features").(missingFields{i});
    end

    receivedData = receiveMessage(tcpipClient);
    query_points = receivedData.query_points;

    num_points = length(query_points);
    fvalues = cell(1,length(query_points));

    if isfield(receivedData, 'save_path')
        save_path = receivedData.save_path;
    else
        currentFolder = pwd;
        save_path = sprintf('../../results/simulations/%s/full_mats', problem_name);
        save_path = fullfile(currentFolder, save_path);
    end

    %TODO adjust operators as needed.
    for i = 1:num_points
        fname = sprintf('%s/simulation_%d_%s.mat', save_path, i, datestr(now,'mm-dd-yyyy HH-MM-SS'));

        if isstruct(query_points)
            qp = query_points(i);
        else
            qp = query_points{i};
        end
        switch problem_name
            case {'axonsim','axonsim_regression'}
                fvalues{i} = fun_wrapper(fname, eval_fun, qp, eval_dict);
            case {'axonsim_threshold','axonsim_nerve_block'}
                fvalues{i} = fun_wrapper(fname, eval_fun, qp, eval_dict, 'nerve_block');
            otherwise
                fvalues{i} = fun_wrapper(fname, eval_fun, qp);
        end
    end
    first_loop = i;
    %display("Sending data to Python ...")
    sendMessage(tcpipClient, fvalues);

    % Read values sent from Python
    fprintf("Beginning loop now \n")

    if isfield(receivedData, 'terminate_flag')
        terminateFlag = receivedData.terminate_flag;
    else
        terminateFlag = false;
    end

    try
        while ~terminateFlag
            % Receive data from Python
            receivedData = receiveMessage(tcpipClient);

            % Check if termination signal received from Python
            if isfield(receivedData, 'terminate_flag')
                terminateFlag = receivedData.terminate_flag;
            end
            if ~isfield(receivedData, "query_points")
                error('MissingFieldError:FieldNotFound', ...
                    'The field "query_points" is required but is missing.');
            end
            query_points = receivedData.query_points; %TODO Reshape if needed (batch_sampling may need this)

            num_points = length(query_points);

            if terminateFlag
                if isfield(receivedData, 'terminate_flag')
                    msg = receivedData.message;
                else
                    msg = "";
                end
                fprintf('Termination signal received from Python with message %s \n Saving data ... \n Closing connection...', msg);

            else
                fprintf("requested query points \n")
                fvalues = cell(num_points, 1);
                for i = 1:num_points
                    fname = sprintf('%s/simulation_%d_%s.mat', save_path, i+first_loop, datestr(now,'mm-dd-yyyy HH-MM-SS'));

                    if isstruct(query_points)
                        qp = query_points(i);
                    else
                        qp = query_points{i};
                    end
                    switch problem_name
                        case {'axonsim','axonsim_regression'}
                            fvalues{i} = fun_wrapper(fname, eval_fun, qp, eval_dict);
                        case {'axonsim_threshold','axonsim_nerve_block'}
                            fvalues{i} = fun_wrapper(fname, eval_fun, qp, eval_dict, 'nerve_block');
                        otherwise
                            fvalues{i} = fun_wrapper(fname, eval_fun, qp);
                    end
                end
                sendMessage(tcpipClient, fvalues);
            end
        end
    catch ME
        if strcmp(ME.identifier, 'MissingFieldError:FieldNotFound')
            warning('Exiting loop because of missing query.');
        else
            rethrow(ME); % let unknown errors propagate
        end
    end
    % Close connection
    delete(tcpipClient);
    clear tcpipClient;

end
