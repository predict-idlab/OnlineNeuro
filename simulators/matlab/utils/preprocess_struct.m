function feat_struct = preprocess_struct(feat_struct)
    display(feat_struct)
    funTypes = feat_struct.fun_type;
    if ischar(funTypes) || isstring(funTypes)
        funTypes = {funTypes}; % Convert to a cell array
        feat_struct.('fun_type') = funTypes;
    end

    num_pulses = numel(funTypes);

    feat_struct.custom_fun = strings(num_pulses,1);
    feat_struct.I = zeros(num_pulses, 1);
    feat_struct.pulse_dur = zeros(num_pulses, 1);
    % TODO Update this later, if needed. Currently we only use single electrodes
    feat_struct.e_type = repmat("single", num_pulses, 1);

    j = 1;
    fields_to_remove = {};

    for i = 1:num_pulses

        currentType = funTypes(i);
        % NOTE: We use (i-1) to create field names to match Python's 0-based indexing.
        py_index = i - 1;

        if ismember(currentType, {'single_pulse','double_pulse'})
           feat_struct.custom_fun(i) = currentType;

           % Read and store I
           if num_pulses == 1
               field_name_I = 'pulse_parameters_I';
               field_name_dur = 'pulse_parameters_pulse_dur';
           else
               field_name_I = sprintf('pulse_parameters_%d_I', py_index);
               field_name_dur = sprintf('pulse_parameters_%d_pulse_dur', py_index);
           end
           feat_struct.I(i) = feat_struct.(field_name_I);
           feat_struct.pulse_dur(i) = feat_struct.(field_name_dur);

           fields_to_remove{end+1} = field_name_I; % Mark for removal
           fields_to_remove{end+1} = field_name_dur; % Mark for removal

        elseif ismember(currentType, {'pulse_ramp'})
            functionName = sprintf('temp_ramp_%s(t)',num2str(j));
           if num_pulses == 1
                delay_name = 'pulse_parameters_delay';
                amplitude_name = 'pulse_parameters_amplitude';
                pulse_width_name = 'pulse_parameters_pulse_width';
                interphase_gap_name = 'pulse_parameters_interphase_gap';
                decay_width_name = 'pulse_parameters_decay_width';
                k_name = 'pulse_parameters_k';
                ramp_width_name = 'pulse_parameters_ramp_width';
           else
                delay_name = sprintf('pulse_parameters_%d_delay',py_index);
                amplitude_name = sprintf('pulse_parameters_%d_amplitude',py_index);
                pulse_width_name = sprintf('pulse_parameters_%d_pulse_width',py_index);
                interphase_gap_name = sprintf('pulse_parameters_%d_interphase_gap',py_index);
                decay_width_name = sprintf('pulse_parameters_%d_decay_width',py_index);
                k_name = sprintf('pulse_parameters_%d_k',py_index);
                ramp_width_name = sprintf('pulse_parameters_%d_ramp_width',py_index);
           end

            write_ramp_fun(feat_struct.(delay_name), ...
                feat_struct.(amplitude_name), ...
                feat_struct.(pulse_width_name), ...
                feat_struct.(interphase_gap_name), ...
                feat_struct.(decay_width_name), ...
                feat_struct.(k_name), ...
                feat_struct.(ramp_width_name), ...
                num2str(j))

            feat_struct.custom_fun(i) = functionName;
            feat_struct.I(i) = 1; % Writing current as 1, actual amplitude is generated in the function
            feat_struct.pulse_dur(i) = 0; % It is not used

            fields_to_remove{end+1} = delay_name;
            fields_to_remove{end+1} = amplitude_name;
            fields_to_remove{end+1} = pulse_width_name;
            fields_to_remove{end+1} = interphase_gap_name;
            fields_to_remove{end+1} = decay_width_name;
            fields_to_remove{end+1} = k_name;
            fields_to_remove{end+1} = ramp_width_name;

            j = j+1;

        else
            display("Current type '%s' is not implemented.\n", currentType);
            error("Not implemented Error")
        end
    end
    % --- Post-processing ---
    if isfield(feat_struct, 'e_offset') && feat_struct.e_offset ~= 0
        % Using explicit 2D indexing for clarity. Assumes e_pos is Nx2 matrix.
        for i = 1:size(feat_struct.e_pos, 1)
            % Shift the x-coordinate (first column)
            feat_struct.e_pos(i, 1) = feat_struct.e_pos(i, 1) + feat_struct.e_offset;
        end
    end

    if ~isempty(fields_to_remove)
        feat_struct = rmfield(feat_struct, fields_to_remove);
    end
end
