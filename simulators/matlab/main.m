function main(json_data)
    format long g
    pyenv('ExecutionMode','InProcess');

    root_path = fullfile(pwd, '../../');
    addpath(genpath(pwd))

    %display(json_data)
    data = jsondecode(json_data);
    %display(data.('problem_config'))
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
    display(problem_config.('experimentParameters'))
    problem_name = problem_config.('experimentParameters').('problem_name');
    problem_type = problem_config.('experimentParameters').('problem_type');
    connection_config = data.('connection_config');

    fprintf("Verifying port is open \n");
    % IMPORTANT: Matlab comes with its on C++ libraries which are used Python.
    % If using linux, hide those (or delete) as explained in
    % https://nl.mathworks.com/matlabcentral/answers/1907290-how-to-manually-select-the-libstdc-library-to-use-to-resolve-a-version-glibcxx_-not-found
    cmd = sprintf('fuser %d/tcp', connection_config.port);
    system(cmd);

    tcpipClient = tcpclient(connection_config.ip, connection_config.port, ...
        "Timeout",connection_config.Timeout, "ConnectTimeout",connection_config.ConnectTimeout, "EnableTransferDelay",false);

    SizeLimit = connection_config.SizeLimit;
    pause(1);

    % Send handshake
    dataToSend = struct('message', 'Hello from MATLAB','dummyNumber', 123);
    jsonData = jsonencode(dataToSend);
    write(tcpipClient, jsonData, 'char');
    fprintf("Data sent \n")
    disp('Waiting for data...');
    %end
    pause(1);
    receivedData = readClientData(tcpipClient);
    % End handshake, connection works.

    display("Selected problem")
    display(problem_name)
    % Prepare call function based on problem_config

    switch problem_name
        case 'circle_classification'
            [eval_fun, eval_dict] = circle_problem('plot', true);
        case 'rose_regression'
            [eval_fun, eval_dict] = rosenbrock_problem('plot', true);
        case 'vlmop2'
            %TODO fix this
            [eval_fun, eval_dict] = multiobjective_problem('plot', true);
        case {'axonsim_single', 'axonsim_double', 'axonsim_regression','axon_threshold','axonsim_nerve_block'}
            [eval_fun, eval_dict] = axon_problem(problem_config.('experimentParameters'),'plot', false);
            n_features = fieldnames(eval_dict);
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [eval_fun, eval_dict] = rosenbrock_problem('plot',true);
    end

    receivedData = readClientData(tcpipClient);
    fieldnames_fixed = fieldnames(receivedData.("Fixed_features"));
    fieldnames_trainable = fieldnames(eval_dict);

    missingFields = setdiff(fieldnames_fixed, fieldnames_trainable);

    for i = 1:numel(missingFields)
        eval_dict.(missingFields{i}) = receivedData.("Fixed_features").(missingFields{i});
    end

    receivedData = readClientData(tcpipClient);
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
    sendData(fvalues, tcpipClient, SizeLimit);

    % Read values sent from Python
    fprintf("Beginning loop now \n")

    if isfield(receivedData, 'terminate_flag')
        terminateFlag = receivedData.terminate_flag;
    else
        terminateFlag = false;
    end


    while ~terminateFlag
        % Receive data from Python
        receivedData = readClientData(tcpipClient);
        query_points = receivedData.query_points; %TODO Reshape if needed (batch_sampling may need this)

        num_points = length(query_points);
        % Check if termination signal received from Python

        if isfield(receivedData, 'terminate_flag')
            terminateFlag = receivedData.terminate_flag;
        else
            terminateFlag = false;
        end
        if terminateFlag
            fprintf('Termination signal received from Python \n Saving data ... \n Closing connection...');
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
            sendData(fvalues, tcpipClient, SizeLimit);
        end
    end
    % Close connection
    delete(tcpipClient);
    clear tcpipClient;

end
