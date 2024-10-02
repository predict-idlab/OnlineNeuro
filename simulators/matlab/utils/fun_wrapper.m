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