function main(json_data)
    format long g
    pyenv('ExecutionMode','InProcess');
    if nargin==0
        %A default routine here for debugging purposes.
        %In practice main is not executed without recieveing a json_data
        addpath('../../')
        data = jsondecode(fileread('config.json'));
        display(data)
        connection_config = data.('connection_config');
        problem_config = jsondecode(fileread('../../config_experiments/experiment_axonsim_double.json'));
        default_config = jsondecode(fileread('../../config_experiments/experiment_axonsim_double.json'));
        problem_name = 'axonsim_double';

    else
        display(json_data)
        root_path = fullfile(pwd, '../../');        
        data = jsondecode(json_data);
        display(data)
        problem_name = data.('problem_name');
        problem_type = data.('problem_type');
        connection_config = data.('connection_config');
        %Detect if path or configuration was provided
        if isfield(data.('problem_config'), 'config_path')
            %Improve this so it can work with absolute paths and not just
            %relatives
            file_path = fullfile(root_path, data.problem_config.config_path);
            problem_config = jsondecode(fileread(file_path));
        else
            problem_config = data.('problem_config');
        end
        default_config = jsondecode(fileread('../../config_experiments/axonsim_template.json'));
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
    display(receivedData)
    % End handshake, connection works.

    % First package sent to Python contains the details of the simulation.
    % Information such as feature map (name -> [min_bound, max_bound]),
    % number of objectives (1), constraints should be passed here.
    % It could also be loaded as shared parameters like config.json,
    % however, it may make sense for later stages, that it is defined from
    % the simulator side and passed to Python.
    % TODO decide on the logic if using Streamlit

    switch problem_name
        case 'circle_classification'
            [eval_fun, features, n_targets] = circle_problem('plot',true);
        case 'rose_regression'
            [eval_fun, features, n_targets] = rosenbrock_problem('plot',true);
        case 'vlmop2'
            %TODO fix this
            [eval_fun, features, n_targets] = multiobjective_problem('plot',true);
        case {'axonsim_single', 'axonsim_double', 'axonsim_regression','axon_threshold','axonsim_nerve_block'}
            [eval_fun, features, n_targets, eval_dict] = axon_problem(problem_config,'plot', false, 'default_setting', default_config);
            n_features = fieldnames(features);
            display(n_features)
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [eval_fun, features, n_targets] = rosenbrock_problem('plot',true);
    end
    
    exp_summary = struct('features',features,'n_targets', {n_targets}, ...
                        'constraints', '');
    
    jsonData = jsonencode(exp_summary);
    write(tcpipClient, jsonData, 'char');
    fprintf("Data sent \n") 
    
    %First batch of query points
    receivedData = readData(tcpipClient);
    display(receivedData)
    
    query_points = receivedData.query_points;
    num_points = length(query_points);
    
    num_targets = length(n_targets);
    fvalues = cell(length(query_points), 1);

    %TODO adjust operators as needed.
    switch problem_name
        case {'axonsim','axonsim_single','axonsim_double'}
            for i =1:num_points
                fvalues{i} = fun_wrapper(eval_fun, query_points(i,:), num_targets, n_features, eval_dict);
            end
        case {'axonsim_threshold','axonsim_nerve_block'}
            for i =1:num_points
                fvalues{i} = fun_wrapper(eval_fun, query_points(i,:), num_targets, n_features, eval_dict, 'nerve_block');
            end
        otherwise
            for i =1:num_points
                fvalues{i} = fun_wrapper(eval_fun, query_points(i,:), num_targets);
            end
    end
    
    %Sending first (large batch)
    dataToSend = struct('init_response', fvalues);
    jsonData = jsonencode(dataToSend);
    
    size_lim = 1024;

    if length(jsonData)>size_lim 
        fprintf("breaking down message")
        pckgs = floor(length(jsonData)/size_lim) + 1;
        %dataToSend.tot_pckgs = pckgs;
        %jsonData = jsonencode(dataToSend);
        row_per_pckg = ceil(length(fvalues)/pckgs);

        for i=1:pckgs
            start_ix = (i-1)*row_per_pckg + 1;
            end_ix = min(i*row_per_pckg , length(fvalues));
            chunk = struct();
            chunk.('data') = dataToSend(start_ix:end_ix);
            chunk.('tot_pckgs') = pckgs;
            jsonChunk = jsonencode(chunk);
            write(tcpipClient, jsonChunk, 'char');
        end
    else
        write(tcpipClient, jsonData, 'char');
    end

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
        num_points = length(qp)/length(fieldnames(features));
        % Check if termination signal received from Python
        terminateFlag = receivedData.terminate_flag;

        if terminateFlag
            fprintf('Termination signal received from Python \n Saving data ... \n Closing connection...');
        else
            fprintf("requested query points \n")
            fvalues = cell(num_points, 1);
            switch problem_name
                case {'axonsim','axonsim_single','axonsim_double'}
                    for i = 1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, qp(i,:), num_targets, n_features, eval_dict);
                    end
                %TODO, define experiment setup for nerve block tests                    
                case {'axonsim_threshold','axonsim_nerve_block'}
                    for i =1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, qp(i,:), num_targets, n_features, eval_dict, 'nerve_block');
                    end

                otherwise
                    for i = 1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, qp(i,:), num_targets);
                    end
            end
            dataToSend = struct('observations', fvalues);
            sendData = jsonencode(dataToSend);
            write(tcpipClient, sendData,"char");
        end
    end
    % Close connection
    delete(tcpipClient);
    clear tcpipClient;

end






