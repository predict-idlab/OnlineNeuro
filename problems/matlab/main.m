% MATLAB code to send data via TCP/IP
function main(json_data)
    pyenv('ExecutionMode','InProcess')
    format long g
    data = jsondecode(json_data);

    fprintf("Verifying port is open \n");
    % IMPORTANT
    % Matlab comes with its on C++ libraries which are used Python.
    % If using linux, hide those (or delete) as explained in
    % https://nl.mathworks.com/matlabcentral/answers/1907290-how-to-manually-select-the-libstdc-library-to-use-to-resolve-a-version-glibcxx_-not-found

    cmd = sprintf('fuser %d/tcp', data.port);
    system(cmd);

    tcpipClient = tcpclient(data.ip, data.port, ...
        "Timeout",data.Timeout, "ConnectTimeout",data.ConnectTimeout, "EnableTransferDelay",false);

    pause(1);

    % Send handshake
    %if matlab_initiates
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
    % Information such as experiment name, feature map (name -> [min_bound, max_bound]),
    % number of objectives (1), constraints should be passed here.
    % It could also be loaded as shared parameters like config.json,
    % however, it may make sense for later stages, that it is defined from
    % the simulator side and passed to Python.
    % TODO decide on the logic if using Streamlit

    switch data.problem
        case 'circle'
            [fun_name, eval_fun, features, n_targets] = circle_problem(true);
        case 'rose'
            [fun_name, eval_fun, features, n_targets] = rosenbrock_problem(true);
        case 'vlmop2'
            %TODO fix this
            [fun_name, eval_fun, features, n_targets] = multiobjective_problem(true);
        case {'axon_single', 'axon_double', 'axon_threshold','nerve_block'}
            [fun_name, eval_fun, features, n_targets, eval_dict] = axon_problem(false, data.problem);
            n_features = fieldnames(features);
        otherwise
            warning('Unexpected problem type, defaulting to Rosenbrock')
            [fun_name, eval_fun, features, n_targets] = rosenbrock_problem(true);
    end
    
    display(eval_fun)
    display(features)

    exp_summary = struct('name', fun_name, ...
                        'features',features,'n_targets', {n_targets}, ...
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
    
    switch data.problem
        case {'axon','axon_single','axon_double'}
            for i =1:num_points
                fvalues{i} = fun_wrapper(eval_fun, query_points(i,:), num_targets, n_features, eval_dict);
            end
        case {'axon_threshold','nerve_block'}
            for i =1:num_points
                fvalues{i} = fun_wrapper(eval_fun, query_points(i,:), num_targets, n_features, eval_dict, data.problem);
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
            switch data.problem
                case {'axon','axon_single','axon_double'}
                    for i = 1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, qp(i,:), num_targets, n_features, eval_dict);
                    end
                %TODO, define experiment setup for nerve block tests                    
                case {'axon_threshold','nerve_block'}
                    for i =1:num_points
                        fvalues{i} = fun_wrapper(eval_fun, qp(i,:), num_targets, n_features, eval_dict, data.problem);
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

function decodedData = readData(obj, ~)
    % Read data from the TCP/IP connection
    data = readline(obj);  % Read all available bytes

    % Decode the received data (assuming it's JSON)
  decodedData = jsondecode(data);
end


function y = fun_wrapper(fun, qp, num_targets, feat_names, feat_struct, operator, channel, full_response)
    % TODO. handle putting multi-input vectors in the correct struct 
    if nargin < 2
        error('Fun Wrapper requires at least two input arguments.');
    end

    if nargin == 2
        num_targets = 1;
    end
    % Toy problems
    if nargin <= 3
        if num_targets == 1
            y = fun(qp);
        elseif num_targets ==2
            [y0, y1] = fun(qp);
            y = [y0; y1];
        else
            error("More than 2 targets not implemented!")
        end

    else
        % Default values
        if nargin <5
            error("If using feat_names, a feat_struct is required")
        end

        if nargin < 6
            operator = 'default';
        end

        if nargin < 7
            channel = 8;
        end

        if nargin < 8
            full_response = false;
        end

        for i=1:length(feat_names)
            feat_struct.(string(feat_names(i))) = qp(i);
        end

        feat_struct = preprocess_struct(feat_struct);
        y = fun(feat_struct);

        % Operator applied to only a node 
        % TODO, extend what else to do here. 
        switch operator
            case 'min_global'
                % Pass the entire signal (models AP over time)
                y = y.Yp(:,channel);

            case 'min_max'
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                y = -(y_max - y_min);

            case 'threshold'
                y_max = max(y.Yp(:,channel));
                if y_max>0.0
                    y = 1;
                else
                    y = 0;
                end
            case 'nerve_block' %Seen as classification atm
                %TODO save the entire pulses.
                ch_1 = y.Yp(:,1); % Verify this?
                ch_2 = y.Yp(:,18);
                
                ch_1_min = min(ch_1);
                ch_1_max = max(ch_1);
                ch_2_min = min(ch_2);
                ch_2_max = max(ch_2);

                % figure(1)
                % plot(ch_1)
                % hold on
                % plot(ch_2)
                % hold off
                % fig = gca;
                % save_figure_counter(fig, './figures/nerve_block/caps/', 'caps_first_last')
                % 
                % figure(2)
                % plot(y.Yp(:,1:18)+ (-9:8)*40)
                % fig = gca;
                % save_figure_counter(fig, './figures/nerve_block/caps/', 'caps_all')

                if ((ch_1_max - ch_1_min) > 80) && (ch_1_max > 0)
                    ap_1 = 1;
                else
                    ap_1 = 0;
                end
                
                if ((ch_2_max - ch_2_min) > 80) && (ch_2_max > 0)
                    ap_2 = 1;
                else
                    ap_2 = 0;
                end
                
                if ((ap_1 == 1) & (ap_2 == 0)) | ((ap_1 == 0) & (ap_2 == 1))
                    y = 1;

                else
                    y = 0;
                end
            otherwise
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                y = -(y_max - y_min);
        end
    
    end

end

