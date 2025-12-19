function out_struct = fun_wrapper(fname, fun, qp, feat_struct, operator, channel)
    if nargin < 3
        error('Fun Wrapper requires at least three input arguments.');
    end

    % Toy problems
    if nargin <= 3
        display(qp)
        response = fun(qp);
        out_struct = struct('observations', response);

    elseif nargin < 4
            error(" A feature structure needs to be provided")
    else
        if nargin < 5
            operator = 'default';
        end

        if nargin < 6
            channel = 8;
        end

        fieldNames = fieldnames(qp);
        for i=1:length(fieldNames)
            fN = fieldNames{i};
            feat_struct.(fN) = qp.(fN);
        end

        feat_struct = preprocess_struct(feat_struct);
        y = fun(feat_struct);

        out_struct = struct();
        %Saves the entire pulses for later verification.
        out_struct.('full_observations') = fname;
        save(fname, 'y');

        % TODO, extend what else to do here.
        switch operator
            % Notice some operators applied to only a node
            case 'min_global'
                % Pass the entire signal (i.e. modelling AP over time)
                out_struct.('observations') = y.Yp(:,channel);

            case 'min_max'
                out_struct.('observations') = max(y.Yp(:,channel)) - min(y.Yp(:,channel));

            case 'threshold'
                y_max = max(y.Yp(:,channel));
                if y_max>0.0
                    response = 1;
                else
                    response = 0;
                end
                out_struct.('observations') = response;

            case 'nerve_block'
                threshold_ap = -20;
                % We will assume that leftmost electrode is the generating
                % one, and rightmost is the blocking one.
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
                out_struct.('observations') = response;

            otherwise % Returning the range
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                response = -(y_max - y_min);
                out_struct.('observations') = response;

        end
    end

end
