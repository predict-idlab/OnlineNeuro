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
        display(qp)
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
        
        y = fun(feat_struct);
        out_struct = struct();

        % Operator applied to only a node 
        % TODO, extend what else to do here. 
        switch operator
            case 'min_global'
                % Pass the entire signal (models AP over time)
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
                %Seen as classification atm
                %TODO save the entire pulses.
                % Verify this?
                threshold_ap = 15;
                ncols = size(y.Yp,2);
                up_to = floor(ncols /2);

                ch_1 = y.Yp(:,1:up_to); % we try to detect an AP here
                ch_2 = y.Yp(:,up_to+1:ncols); % We try to detect a block here
                
                ap_generated = any(ch_1(:)>threshold_ap);
                effective_block = ap_generated && any(ch_2(:) < threshold_ap);
                if effective_block
                    response=1;
                else
                    response=0;
                end
                out_struct.('full_observations') = y.fname;
                out_struct.('observations') = response;

                % figure(1)
                % plot(ch_1)
                % hold on
                % plot(ch_2)
                % hold off
                % fig = gca;
                % save_figure_counter(fig, './figures/nerve_block/caps/', 'caps_first_last')
                % 
                % figure(1)
                % ncols = size(y.Yp,2);
                % plot(y.Yp + (1:ncols)*40)
                % fig = gca;
                % save_figure_counter(fig, '../../figures/nerve_block/caps/', 'caps_all')

                % if ((ch_1_max - ch_1_min) > 80) && (ch_1_max > 0)
                %     ap_1 = 1;
                % else
                %     ap_1 = 0;
                % end

                % if ((ch_2_max - ch_2_min) > 80) && (ch_2_max > 0)
                %     ap_2 = 1;
                % else
                %     ap_2 = 0;
                % end
                % 
                % if ((ap_1 == 1) & (ap_2 == 0)) | ((ap_1 == 0) & (ap_2 == 1))
                %     y = 1;
                % 
                % else
                %     y = 0;
                % end
                
            otherwise
                y_min = min(y.Yp(:,channel));
                y_max = max(y.Yp(:,channel));
                response = -(y_max - y_min);
                out_struct.('observations') = response;

        end
    
    end

end