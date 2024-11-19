function feat_struct = preprocess_struct(feat_struct)
    % TODO this could be done in the loading of defaults rather than
    % here
    funTypes = feat_struct.fun_type;
    j = 1;
    k = 1;
    
    feat_struct.custom_fun = strings(numel(funTypes),1);
    original_currents = feat_struct.I;
    original_pulse_dur = feat_struct.pulse_dur;
    feat_struct.I = zeros(numel(funTypes), 1);
    feat_struct.pulse_dur = zeros(numel(funTypes), 1);

    for i = 1:numel(funTypes)
        currentType = feat_struct.fun_type(i);
        if ismember(currentType, {'single_pulse','double_pulse'})
           feat_struct.custom_fun(i) = currentType;
           feat_struct.I(i) = original_currents(k);
           feat_struct.pulse_dur(i) = original_pulse_dur(k);
           k = k+1;

        elseif ismember(currentType, {'pulse_ramp'})
            functionName = sprintf('temp_ramp_%s(t)',num2str(j));
            %functionKey = sprintf('custom_params_%s',i);
            write_ramp_fun(feat_struct.delay(j), ...
                feat_struct.amplitude(j), ...
                feat_struct.pulse_width(j), ...
                feat_struct.interphase_gap(j), ...
                feat_struct.decay_width(j), ...
                feat_struct.k(j), ...
                feat_struct.ramp_width(j), ...
                num2str(j))
            feat_struct.custom_fun(i) = functionName;
            feat_struct.I(i) = 1;%Writing current as 1, actual amplitude is generated in the function
            feat_struct.pulse_dur(i) = 0;%It is not used
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