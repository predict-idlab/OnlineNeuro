function new_struct = preprocess_struct(old_struct)
    %Current convention makes features from axonsim take the name
    %%{name}_%{number} format so that they can be correctly handled from
    %%Pythons end. This function reconverts these features into the correct
    %%array.
    % First find features that have the pattern, then concatenate them in
    % the correct order.

    % Initialize the new structure
    new_struct = struct();
    
    % Get all field names of the input struct
    field_names = fieldnames(old_struct);
    
    % Use a container to group fields with numbers
    grouped_fields = containers.Map('KeyType', 'char', 'ValueType', 'any');
    grouped_indices = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:length(field_names)
        fn = field_names{i};
        
        % Use regex to extract the base name and number if present
        tokens = regexp(fn, '^(.*)_(\d+)$', 'tokens');

        if ~isempty(tokens)
            base_name = tokens{1}{1};
            number = str2double(tokens{1}{2});

            % Add this field's value to the appropriate group
            if isKey(grouped_fields, base_name)
                grouped_fields(base_name) = [grouped_fields(base_name); old_struct.(fn)];
                grouped_indices(base_name) = [grouped_indices(base_name), number];
            else
                grouped_fields(base_name) = old_struct.(fn);
                grouped_indices(base_name) = number;

            end
        else
            % Copy the field to the new struct if it doesn't match the pattern
            new_struct.(fn) = old_struct.(fn);
        end
    end
    
    % Add the grouped fields to the new struct
    keys = grouped_fields.keys();
    for i = 1:length(keys)
        base_name = keys{i};
        [sorted_indices, sort_order] = sort(grouped_indices(base_name));
        sorted_values = grouped_fields(base_name);
        sorted_values = sorted_values(sort_order);
        new_struct.(base_name) = [sorted_values];
        %new_struct.(base_name) = grouped_fields(keys{i});
    end

    %Convert electrode pulses to multiple array
    num_pulses = size(old_struct.('fun_type'),1);
    if all(strcmp(old_struct.('fun_type'),'single pulse'))
        if num_pulses==2
            new_struct.("I") = new_struct.("I")*[1,-1];
        elseif num_pulses ==3
            new_struct.("I") = new_struct.("I")*[-0.5, 1, -0.5];
        end

    end
    % Adjust electrode shift
    if isfield('electrode_shift',new_struct)
        new_struct.('e_pos')(1) = new_struct.('e_pos')(1) + new_struct('electrode_shift');
        new_struct.('e_pos')(3) = new_struct.('e_pos')(1) + new_struct('electrode_shift');
    end


end