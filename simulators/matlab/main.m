function main(json_data)
    format long g
    pyenv('ExecutionMode','InProcess');
    
    root_path = fullfile(pwd, '../../');
    addpath(genpath(pwd))

    display(json_data)
    data = jsondecode(json_data);
    display(data.('problem_config'))
    problem_name = data.('problem_name');
    problem_type = data.('problem_type');
    connection_config = data.('connection_config');
    %Detect if path or configuration was provided
    problem_config = data.('problem_config');
    if isstring(problem_config)
        file_path = fullfile(root_path, problem_config);
        problem_config = jsondecode(fileread(file_path));
    end

    fprintf("Verifying port is open \n");
    % IMPORTANT
    % Matlab comes with its on C++ libraries which are used Python.
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
    receivedData = readData(tcpipClient);
    display("1")
    display(receivedData)
    % End handshake, connection works.

    display("Selected problem")
    display(problem_name)
    % Prepare call function based on problem_config 

    switch problem_name
        case 'circle_classification'
            [eval_fun, eval_dict] = circle_problem('plot',true);
        case 'rose_regression'
            [eval_fun, eval_dict] = rosenbrock_problem('plot',true);
        case 'vlmop2'
            %TODO fix this
            [eval_fun, eval_dict] = multiobjective_problem('plot',true);
        case {'axonsim_single', 'axonsim_double', 'axonsim_regression','axon_threshold','axonsim_nerve_block'}
            [eval_fun, eval_dict] = axon_problem(problem_config,'plot', false);
            n_features = fieldnames(eval_dict);
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [eval_fun, eval_dict] = rosenbrock_problem('plot',true);
    end

    %    exp_summary = struct('features', eval_dict, ...
    %                        'constraints', '');

    receivedData = readData(tcpipClient);
    fieldnames_fixed = fieldnames(receivedData.("Fixed_features"));
    fieldnames_trainable = fieldnames(eval_dict);

    missingFields = setdiff(fieldnames_fixed, fieldnames_trainable);

    for i = 1:numel(missingFields)
        eval_dict.(missingFields{i}) = receivedData.("Fixed_features").(missingFields{i});
    end

    receivedData = readData(tcpipClient);
    query_points = receivedData.query_points;

    num_points = length(query_points);
    fvalues = cell(1,length(query_points));

    %TODO adjust operators as needed.
    for i = 1:num_points
        if isstruct(query_points)
            qp = query_points(i);
        else
            qp = query_points{i};
        end
        switch problem_name
            case {'axonsim','axonsim_single','axonsim_double','axonsim_regression'}
                fvalues{i} = fun_wrapper(eval_fun, qp, eval_dict);
            case {'axonsim_threshold','axonsim_nerve_block'}
                fvalues{i} = fun_wrapper(eval_fun, qp, eval_dict, 'nerve_block');
            otherwise
                fvalues{i} = fun_wrapper(eval_fun, qp);
        end
    end
    
    %Sending first (large batch)
    if nargin == 0
        quit()
    end

    display("Sending data to Python ...")
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
        receivedData = readData(tcpipClient);
        qp = receivedData.query_points; %TODO Reshape if needed (batch_sampling may need this)
        
        num_points = length(qp);
        % Check if termination signal received from Python

        terminateFlag = receivedData.terminate_flag;

        if terminateFlag
            fprintf('Termination signal received from Python \n Saving data ... \n Closing connection...');
        else
            fprintf("requested query points \n")
            fvalues = cell(num_points, 1);
            switch problem_name
                case {'axonsim','axonsim_single','axonsim_double','axonsim_regression'}
                    for i = 1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, query_points(i), eval_dict);
                    end
                %TODO, define experiment setup for nerve block tests                    
                case {'axonsim_threshold','axonsim_nerve_block'}
                    for i =1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, query_points(i), eval_dict, 'nerve_block');
                    end

                otherwise
                    for i = 1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, query_points(i));
                    end
            end
            sendData(fvalues, tcpipClient, SizeLimit);
        end
    end
    % Close connection
    delete(tcpipClient);
    clear tcpipClient;

end

