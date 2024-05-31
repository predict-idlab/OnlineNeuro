function [fun_name, eval_fun, upper_bound, lower_bound, n_features, n_targets, eval_dict] = axon_problem(plot_bool, problem_setting)
    if nargin < 2
        error('axon_problem requires at two input arguments.');
    end
    %TODO ~ This needs to be specified somewhere so that it can load from
    %any machine 
    config_file = ("./config.json");
    exp_file = fileread(config_file);

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
    fun_name = 'AxonSim';
    eval_fun = @axonsim_call;

    fields = fieldnames(default_params);
    eval_dict = struct();

    for i = 1:length(fields)
        if isfield(experiment_params, fields{i})
            eval_dict.(fields{i}) = experiment_params.(fields{i}).('value');
        else
            eval_dict.(fields{i}) = default_struct.(fields{i}).('default');
        end
    end
    
    upper_bound = [];
    lower_bound = [];
    n_features = {};

    for i = 1:length(fields)
        if isfield(experiment_params, fields{i})
            if experiment_params.(fields{i}).('optimizable')
                n_features = [n_features, fields{i}];
                if isfield(experiment_params.(fields{i}), 'min_value')
                    lower_bound = [lower_bound, experiment_params.(fields{i}).('min_value')];
                else
                    lower_bound = [lower_bound, default_params.(fields{i}).('min_value')];
                end
                %It would be strange that min was specified but no max, but
                %out of completeness
                if isfield(experiment_params.(fields{i}),'max_value')
                    upper_bound = [upper_bound, experiment_params.(fields{i}).('max_value')];
                else
                    upper_bound = [upper_bound, default_params.(fields{i}).('max_value')];
                end                
            end
        end % Non specified variables cannot be tuned and are fixed.
    end    

    % For the time being, although post-processing is still needed and
    % seems that it will be handled outside?
    n_targets = 1;

end

