function [fun_name, eval_fun, features, n_targets, eval_dict] = axon_problem(plot_bool, problem_setting)
    if nargin < 2
        error('axon_problem requires at two input arguments.');
    end
    
    config_file = ("./config.json");
    exp_file = jsondecode(fileread(config_file));
    
    addpath(genpath(exp_file.axonsim_path))
    
    %problem_setting
    exp_file = sprintf('./config_experiments/experiment_%s.json',problem_setting);
    exp_file = fileread(exp_file);
    default_file  = sprintf('./config_experiments/parameter_spec_%s.json', problem_setting);
    default_file = fileread(default_file);
    
    experiment_params = jsondecode(exp_file);
    default_params = jsondecode(default_file);
    
    % Plot the function surface
    if plot_bool
        disp("AxonSim can't display a plot as the response surface is unknown");
    end
    fun_name = problem_setting;
    eval_fun = @axonsim_call;
    
    fields = fieldnames(default_params);
    eval_dict = struct();
    
    for i = 1:length(fields)
        %Making sure that if variables are going to be optimized, and have multiple electrodes
        % they have the same dimensions.
        %This allows setting different min/max values for each electrode.
    
        if isfield(experiment_params, fields{i})
            if experiment_params.(fields{i}).('optimizable')
                assert(isequal(size(experiment_params.(fields{i}).value, 1), ...
                    size(experiment_params.(fields{i}).min_value, 1), ...
                    size(experiment_params.(fields{i}).max_value, 1)))
            end
            
            array_size = size(experiment_params.(fields{i}).value, 1);
            if array_size>1
                for j=1:array_size
                    field_num = strcat(fields{i},"_",string(j));
                    eval_dict.(field_num) = experiment_params.(fields{i}).value(j);
                end
            else
                eval_dict.(fields{i}) = experiment_params.(fields{i}).('value');
            end
        else
            if default_params.(fields{i}).('optimizable')
                assert(isequal(size(default_params.(fields{i}).default, 1), ...
                    size(default_params.(fields{i}).min_value, 1), ...
                    size(default_params.(fields{i}).max_value, 1)))
            end
            array_size = size(default_params.(fields{i}).value, 1);
            if array_size>1
                for j=1:array_size
                    field_num = strcat(fields{i},"_",string(j));
                    eval_dict.(field_num) = default_params.(fields{i}).value(j);
                end
            else
                eval_dict.(fields{i}) = default_params.(fields{i}).('default');
            end
        end
    end
    
    upper_bound = [];
    lower_bound = [];
    n_features = {};
    
    for i = 1:length(fields)
        if isfield(experiment_params, fields{i})
            if experiment_params.(fields{i}).('optimizable')
                % If balanced charges, then
                if isfield(experiment_params.(fields{i}), 'min_value')
                    min_value = experiment_params.(fields{i}).('min_value');
                else
                    min_value = default_params.(fields{i}).('min_value');
                end
    
                %It would be strange that min was specified but no max, but
                %out of completeness
                if isfield(experiment_params.(fields{i}),'max_value')
                    max_value = experiment_params.(fields{i}).('max_value');
                else
                    max_value = default_params.(fields{i}).('max_value');
                end
    
                min_array_size = size(min_value, 1);
                max_array_size = size(max_value, 1);
                assert(min_array_size==max_array_size);
    
                array_size = min_array_size;
                delete_vector = [];
    
                if array_size>1
                    if any(strcmp(fields{i},["I", "pulse_dur"])) & experiment_params.("balance_charge").("value")
                        n_features = [n_features; fields{i}];
                    else
                        for j=1:array_size
                            %If Min and Max are identical, it can't be optimized.
                            if min_value(j)~=max_value(j)
                                keep_bool = true;
                                n_features = [n_features; strcat(fields{i},"_",string(j))];
                            else
                                keep_bool = false;
                            end
                            delete_vector = [delete_vector; keep_bool];
                        end
                    end
                else
                    n_features = [n_features; fields{i}];
                end
                % Converting it to vectors in case size_array>1
                %if array_size>1
                %    min_value = ones(1,array_size)*min_value;
                %    max_value = ones(1,array_size)*max_value;
                %end
    
                if any(strcmp(fields{i},["I", "pulse_dur"])) & experiment_params.("balance_charge").("value")
                    min_value = abs(min_value);
                    max_value = abs(max_value);
                    warning_bool = false;
                    if size(min_value, 1)>1
                        warning_bool = true;
                        min_value = min(min_value);
                    end
                    if size(max_value, 1)>1
                        warning_bool = true;
                        max_value= max(max_value);
                    end
                    if warning_bool
                        warning("Total current set to be mirrored and a single value to be tuned (instead of two separate ones)")
                    end
                end
    
                if any(~delete_vector)
                    min_value = min_value(boolean(delete_vector));
                    max_value = max_value(boolean(delete_vector));
                end

                lower_bound = [lower_bound; min_value];
                upper_bound = [upper_bound; max_value];
    
            end
        end % Non specified variables cannot be tuned and are fixed.
    end
       
    features = struct();
    for i = 1:length(n_features)
        features.(n_features{i}) = [lower_bound(i), upper_bound(i)];
    end
    n_targets = 1;
    display(features)
end
