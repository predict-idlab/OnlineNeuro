function out_struct = fun_wrapper(fun, qp,  feat_struct, operator, channel, full_response)
    % TODO. handle putting multi-input vectors in the correct struct
    if nargin < 2
        error('Fun Wrapper requires at least two input arguments.');
    end

    % Toy problems
    if nargin <= 2
        display(qp)
        response = fun(qp);
        out_struct = struct('observations', response);

    elseif nargin <3
            error(" A feature structure needs to be provided")
    else
        if nargin < 4
            operator = 'default';
        end

        if nargin < 5
            channel = 8;
        end

        if nargin < 6
            full_response = false;
        end

        fieldNames = fieldnames(qp);
        for i=1:length(fieldNames)
            fN = fieldNames{i};
            feat_struct.(fN) = qp.(fN);
        end

        feat_struct = preprocess_struct(feat_struct);
        display(feat_struct)
        y = fun(feat_struct);
        out_struct = struct();

        % Operator applied to only a node
        % TODO, extend what else to do here.
        switch operator
            case 'min_global'
                % Pass the entire signal (i.e. modelling AP over time)
                response = y.Yp(:,channel);
                out_struct.('observations') = response;

            case 'min_max'
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                response = -(y_max - y_min);
                out_struct.('observations') = response;

            case 'threshold'
                y_max = max(y.Yp(:,channel));
                if y_max>0.0
                    response = 1;
                else
                    response = 0;
                end
                out_struct.('observations') = response;

            case 'nerve_block'
                %Saves the entire pulses for later verification.
                % Verify this?
                threshold_ap = -20;
                % We will assume that leftmost electrode is the generating
                % one, and rightmos is the blocking one. A block could
                % occur in the other direction, but we ignore that scenario
                % (not valid event).
                data = y.Yp;
                max_time = max(data, [], 1);
                b0 = max_time(1) > threshold_ap;    % First point exceeds threshold
                b1 = max_time(end) < threshold_ap;  % Last point stays below threshold
                effective_block = b0 && b1;

                if effective_block
                    response=1;
                else
                    response=0;
                end

                out_struct.('full_observations') = y.fname;
                out_struct.('observations') = response;


            otherwise
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                response = -(y_max - y_min);
                out_struct.('observations') = response;

        end

    end

end
