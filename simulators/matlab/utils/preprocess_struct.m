function feat_struct = preprocess_struct(feat_struct)
    funTypes = feat_struct.fun_type;
    j = 1;
    k = 1;

    feat_struct.custom_fun = strings(numel(funTypes),1);
    %original_currents = feat_struct.I;
    %original_pulse_dur = feat_struct.pulse_dur;
    feat_struct.I = zeros(numel(funTypes), 1);
    feat_struct.pulse_dur = zeros(numel(funTypes), 1);
    % TODO Update this later Currently only using single electrodes
    feat_struct.e_type = repmat("single", numel(funTypes), 1);

    for i = 1:numel(funTypes)
        currentType = feat_struct.fun_type(i);
        if ismember(currentType, {'single_pulse','double_pulse'})
           feat_struct.custom_fun(i) = currentType;
           field_name = sprintf('pulses_%d_I',i);
           feat_struct.I(i) = feat_struct.(field_name);
           field_name = sprintf('pulses_%d_pulse_dur',i);
           feat_struct.pulse_dur(i) = feat_struct.(field_name);
           % field_name = sprintf('pulses_%d_frq',i);
           % feat_struct.frq(i) = feat_struct.(field_name);
           k = k+1;

        elseif ismember(currentType, {'pulse_ramp'})
            functionName = sprintf('temp_ramp_%s(t)',num2str(j));

            delay_name = sprintf('pulses_%d_delay',i);
            amplitude_name = sprintf('pulses_%d_amplitude',i);
            pulse_width_name = sprintf('pulses_%d_pulse_width',i);
            interphase_gap_name = sprintf('pulses_%d_interphase_gap',i);
            decay_width_name = sprintf('pulses_%d_decay_width',i);
            k_name = sprintf('pulses_%d_k',i);
            ramp_width_name = sprintf('pulses_%d_ramp_width',i);

            write_ramp_fun(feat_struct.(delay_name), ...
                feat_struct.(amplitude_name), ...
                feat_struct.(pulse_width_name), ...
                feat_struct.(interphase_gap_name), ...
                feat_struct.(decay_width_name), ...
                feat_struct.(k_name), ...
                feat_struct.(ramp_width_name), ...
                num2str(j))

            feat_struct.custom_fun(i) = functionName;
            feat_struct.I(i) = 1;%Writing current as 1, actual amplitude is generated in the function
            feat_struct.pulse_dur(i) = 0;%It is not used
            % feat_struct.frq(i) = 1; %It is not used

            j = j+1;

        else
            error("Not implemented Error")
        end
    end

    if feat_struct.('e_offset') ~= 0
        for i =1:numel(funTypes)
            feat_struct.('e_pos')(2*i-1) = feat_struct.('e_pos')(2*i-1) + feat_struct.('e_offset');
        end
    end

    %new_struct.("I") = new_struct.("I")*[1, -1];
    %new_struct.('e_pos')(1) = new_struct.('e_pos')(1) + new_struct('electrode_shift');

end
