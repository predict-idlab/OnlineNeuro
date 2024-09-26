function save_figure_counter(fig, folder, baseFilename)
    % Initialize counter
    counter = 1;
    fileExtension=".png"';
    % Construct the full filename
    fileNameWithCounter =strjoin([folder, baseFilename, "_", sprintf('%02d', counter), fileExtension], "")
    %fullFileName = fullfile(folder, fileNameWithCounter)

    % Check if the file exists and increment the counter if it does
    while isfile(fileNameWithCounter)
        % Update the filename with the new counter
        counter = counter + 1;
        fileNameWithCounter = strjoin([folder, baseFilename, "_", sprintf('%02d', counter), fileExtension], "");
        %fullFileName = fullfile(folder, fileNameWithCounter)
    end

    % Save the figure
    %    exportgraphics(fig, fullPath, 'Resolution', 300);
    exportgraphics(fig, fileNameWithCounter,  'Resolution', 300);
end