
function val = getFieldEither(s, field1, field2)
%GETFIELDEITHER  Get value from struct using one of two possible field names.
%
%   val = GETFIELDEITHER(s, field1, field2)
%
%   Returns the value of s.(field1) if it exists, otherwise s.(field2).
%   Throws an error if neither field is present.
%
%   Example:
%       exp_params = problem_config.experiment_parameters;
%       problem_name = getFieldEither(exp_params, 'problem_name', 'name');
%       problem_type = getFieldEither(exp_params, 'problem_type', 'type');
    if isfield(s, field1)
        val = s.(field1);
    elseif isfield(s, field2)
        val = s.(field2);
    else
        error('Neither "%s" nor "%s" found.', field1, field2);
    end
end
