% MATLAB code to send data via TCP/IP
function main(varargin)
    pyenv('ExecutionMode','InProcess')

    if nargin==0
        matlab_initiates = true; %default value
    else
        matlab_initiates = false;
    end

    format long g
    str = fileread('config.json'); % dedicated for reading files as text 
    data = jsondecode(str); % Using the jsondecode function to parse JSON from string 
    
    fprintf("Verifying port is open \n");    

    if matlab_initiates
        % TODO PENDING. Python execution from within Matlab is a whole mess due to 
        % not finding libraries such as libstdc++.so.6.
        % For the time being the process can only be started from within
        % Python!!!
        if ispc
            status = dos("python server_side.py");
        else
            status = unix("python3 server_side.py");
        end
        fprintf(sprintf("running python command %d \n", status));
        cmd = sprintf("fuser %d/tcp >/dev/null 2>&1 && echo 1 || echo 0", data.port);
        fprintf("Process status %s waiting ... \n", status)

        while true
            [status, output] = system(cmd);
            if status == 0 && strcmp(strtrim(output), '1')
                fprintf("Server is up and running.\n");
                break;
            else
                fprintf("Waiting for server to start...\n");
                pause(3); % Wait for 1 second before checking again
            end
        end

    end
    
    cmd = sprintf('fuser %d/tcp', data.port);
    system(cmd);

    %server = tcpserver(data.ip,data.port);
    tcpipClient = tcpclient(data.ip, data.port, ...
        "Timeout",data.Timeout, "ConnectTimeout",data.ConnectTimeout, "EnableTransferDelay",false);

    pause(1);

    % Send handshake
    dataToSend = struct('message', 'Hello from MATLAB','dummyNumber', 123);
    jsonData = jsonencode(dataToSend);
    write(tcpipClient, jsonData, 'char');
    fprintf("Data sent \n") 
    disp('Waiting for data...');
    
    pause(1);
    receivedData = readData(tcpipClient);
    display(receivedData)
    
    % End handshake, connection works.

    % First package contains the logic of the simulation.
    % Details such as experiment name, feature dimensionality,
    % feature names, number of objectives (1), constraints should be passed here.
    % It could also be loaded as shared parameters like config.json,
    % however, it may make sense for later stages, that it is defined from
    % the Matlab side and passed to Python

    %Defining problem, getting N first samples
    switch data.problem
        case 'circle'
            [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = circle_problem(true);
        case 'rose'
            [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = rosenbrock_problem(true);
        case {'axon_single','axon_double','axon_threshold'}
            [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets, eval_dict] = axon_problem(false, data.problem);
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets] = rosenbrock_problem(true);
    end
    
    display(eval_fun)
    % TODO adjust function_name
    exp_summary = struct('name', fun_name, ...
                        'n_features',{n_features},'n_targets', n_targets, ...
                        'lower_bound', lower_bound, ...
                        'upper_bound', upper_bound, ...
                        'constraints', '');

    jsonData = jsonencode(exp_summary);
    write(tcpipClient, jsonData, 'char');
    fprintf("Data sent \n") 
    
    %First batch of query points
    receivedData = readData(tcpipClient);
    display(receivedData)
    
    query_points = receivedData.query_points;
    num_points = length(query_points);

    fvalues = zeros(length(query_points), 1);
    
    switch data.problem
        case {'axon','axon_single','axon_double'}
            for i =1:num_points
                fvalues(i) = fun_wrapper(eval_fun, query_points(i,:), n_features, eval_dict);
            end
        case {'axon_threshold'}
            for i =1:num_points
                fvalues(i) = fun_wrapper(eval_fun, query_points(i,:), n_features, eval_dict,'threshold');
            end
            
        otherwise
            for i =1:num_points
                fvalues(i) = fun_wrapper(eval_fun, query_points(i,:));
            end
    end
    
    %Sending first (large batch) Usually a GP is initialized with a
    % a few samples
    dataToSend = struct('init_response', fvalues);
    jsonData = jsonencode(dataToSend);
    
    size_lim = 1024;
    if length(jsonData)>size_lim 
        pckgs = floor(length(jsonData)/1024) + 1;
        sprintf("breaking down message")
        %dataToSend.tot_pckgs = pckgs;
        %jsonData = jsonencode(dataToSend);
        row_per_pckg = floor(length(random_inputs)/pckgs);

        for i=1:pckgs
            start_ix = (i-1)*row_per_pckg + 1;
            end_ix = min(i*row_per_pckg , length(random_inputs));

            chunk = struct();
            fields = fieldnames(dataToSend);
            for j = 1:length(fields)
                if ndims(dataToSend.(fields{j}))==1 | size(dataToSend.(fields{j}),1) == 1
                    chunk.(fields{j}) = dataToSend.(fields{j})(start_ix:end_ix);
                else
                    chunk.(fields{j}) = dataToSend.(fields{j})(start_ix:end_ix,:);
                end
            end
            chunk.tot_pckgs = pckgs;
            jsonChunk = jsonencode(chunk);
            write(tcpipClient, jsonChunk, 'char');
        end
    else
        write(tcpipClient, jsonData, 'char');
    end

    % Read values sent from Python 
    fprintf("Beginning loop now \n")

    terminateFlag = false;
    
    while ~terminateFlag
        % Receive data from Python
        receivedData = readData(tcpipClient);

        % Check if termination signal received from Python
        terminateFlag = receivedData.terminate_flag;

        if terminateFlag
            fprintf('Termination signal received from Python \n Saving data ... \n Closing connection...');
        else
            fprintf("requested query points \n")
            num_points = size(receivedData.query_points,1);
            fvalues = zeros(num_points);
            switch data.problem
                case {'axon','axon_single','axon_double'}
                    for i = 1:num_points
                        fvalues(i) = fun_wrapper(eval_fun, receivedData.query_points(i,:), n_features, eval_dict);
                    end
                case {'axon_threshold'}
                    for i =1:num_points
                        fvalues(i) = fun_wrapper(eval_fun, receivedData.query_points(i,:), n_features, eval_dict,'threshold');
                    end
                otherwise
                    for i = 1:num_points
                        fvalues(i) = fun_wrapper(eval_fun, receivedData.query_points(i,:));
                    end
            end
            dataToSend = struct('observations', fvalues);
            sendData = jsonencode(dataToSend);
            write(tcpipClient, sendData,"char");
        end
    end
    % Close connection
    fclose(tcpipClient);

end

function decodedData = readData(obj, ~)
    % Read data from the TCP/IP connection
    data = readline(obj);  % Read all available bytes

    % Decode the received data (assuming it's JSON)
   decodedData = jsondecode(data);
    
end


function y = fun_wrapper(fun, qp, feat_names, feat_struct, operator, chanel)
    % operator defines how to post-process the AP, need to discuss what's
    % meaningful to extract.
    if nargin < 2
        error('Fun Wrapper requires at least two input arguments.');
    end

    % Toy problems
    if nargin == 3
        y = fun(qp);
    else

        % Default values
        if nargin < 5
            operator = 'default';
        end

        if nargin < 6
            chanel = 8;
        end

        for i=1:length(feat_names)
            feat_struct.(string(feat_names(i))) = qp(i);
        end

        y = fun(feat_struct);
        % Operator applied to only a node 
        % TODO, extend what else to do here. 
        switch operator
            case 'min_global'
                % Pass the entire signal (models AP over time)
                y = y.Yp(:,chanel);

            case 'min_max'
                y_min = min(y.Yp(:,chanel));
                y_max = max(y.Yp(:,chanel));
                y = -(y_max - y_min);

            case 'threshold'
                y_max = max(y.Yp(:,chanel));
                if y_max>0.0
                    y = 1;
                else
                    y = 0;
                end

            otherwise
                % min_max (default)
                y_min = min(y.Yp(:,chanel));
                y_max = max(y.Yp(:,chanel));
                y = -(y_max - y_min);
        end
    
    end

end
