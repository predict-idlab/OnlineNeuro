function decodedJson = readClientData(tcpClientObj)
% readClientData Reads a newline-terminated JSON string from a tcpclient object
% and decodes it.
%
% Syntax:
%   decodedJson = readClientData(tcpClientObj)
%
% Inputs:
%   tcpClientObj - A connected tcpclient object.
%
% Outputs:
%   decodedJson  - The decoded JSON data as a MATLAB struct or array.
%
% Throws:
%   An error with a specific identifier if reading fails, data is empty/missing,
%   or JSON decoding fails.

    % if ~(isa(tcpClientObj, 'matlab.net.tcpclient') && tcpClientObj.Connected)
    %     error('readClientData:InputError', 'Input must be a connected tcpclient object.');
    % end

    % fprintf('DEBUG: Class of tcpClientObj.Timeout is: %s\n', class(tcpClientObj.Timeout));
    % fprintf('DEBUG: Value of tcpClientObj.Timeout is: \n');
    % disp(tcpClientObj.Timeout);

    fprintf('MATLAB readClientData: Attempting to read line (client configured timeout: %.2f s)...\n', tcpClientObj.Timeout);

    try
        % readline reads until a newline character or the client's Timeout is reached.
        % It returns a string scalar.
        rawDataLine = readline(tcpClientObj);

        % --- DEBUG ---
        if ismissing(rawDataLine)
            fprintf('MATLAB readClientData: readline returned a MISSING string.\n');
        elseif strlength(rawDataLine) == 0
            fprintf('MATLAB readClientData: readline returned an EMPTY string ("").\n');
        else
            fprintf('MATLAB readClientData: Received from readline (class: %s, length: %d): "%s"\n', ...
                class(rawDataLine), strlength(rawDataLine), rawDataLine);
        end
        % --- END DEBUG ---

    catch ME_read
        % Catch errors specifically from the readline operation
        fprintf('MATLAB readClientData: ERROR during readline operation: %s\n', ME_read.message);
        if strcmp(ME_read.identifier, 'MATLAB:networklib:tcpclient:readline:operationTimeout')
             error('readClientData:ReadTimeout', ...
                 'Timeout (%.2f s) occurred in readline while waiting for data from Python. Check Python sender and client timeout setting.', tcpClientObj.Timeout);
        else
            % For other readline errors (e.g., connection closed unexpectedly)
            rethrow(ME_read);
        end
    end

    % Check if readline returned a missing string or an empty string
    if ismissing(rawDataLine) || strlength(rawDataLine) == 0
        errorMessage = sprintf(['readline returned %s. This usually means the operation timed out (client timeout: %.2f s) ', ...
                               'before a complete line (terminated by newline) was received from Python, or Python sent an empty line.'], ...
                                 ternary(ismissing(rawDataLine), 'a MISSING string', 'an EMPTY string ("")'), ...
                                 tcpClientObj.Timeout);
        fprintf('MATLAB readClientData: %s\n', errorMessage);
        error('readClientData:EmptyOrMissingDataReceived', '%s', errorMessage);
    end

    % Convert the received string to a character vector and trim whitespace
    % (strtrim removes leading/trailing whitespace, including newlines if any were present)
    jsonString = strtrim(char(rawDataLine));

    % After trimming, it's possible the string becomes empty if it only contained whitespace
    if isempty(jsonString)
        fprintf('MATLAB readClientData: Data after strtrim is empty. Original from readline was (class: %s): "%s"\n', class(rawDataLine), rawDataLine);
        error('readClientData:WhitespaceOnlyDataReceived', ...
              'Received only whitespace from Python. Original from readline: "%s"', rawDataLine);
    end

    try
        decodedJson = jsondecode(jsonString);
        fprintf('MATLAB readClientData: JSON decoding successful.\n');
    catch ME_decode
        fprintf('MATLAB readClientData: ERROR during jsondecode.\n');
        fprintf('MATLAB readClientData:   MATLAB error message: %s\n', ME_decode.message);
        fprintf('MATLAB readClientData:   Problematic JSON string was: "%s"\n', jsonString);

        % For very difficult cases, save the problematic string to a file
        problem_data_filename = sprintf('matlab_jsondecode_error_data_%s.mat', ...
                                        datestr(now, 'yyyy-mm-dd_HHMMSS'));
        try
            save(problem_data_filename, 'jsonString', 'rawDataLine');
            fprintf('MATLAB readClientData:   Problematic string and original readline output saved to %s\n', problem_data_filename);
        catch ME_save
            fprintf('MATLAB readClientData:   Failed to save problematic string: %s\n', ME_save.message);
        end

        % Create a new exception with more context
        newExc = MException('readClientData:JSONDecodeError', ...
            'Failed to decode JSON string. Error: "%s". Attempted to decode: "%s"', ...
            ME_decode.message, jsonString);
        newExc = addCause(newExc, ME_decode); % Preserve original error cause
        throw(newExc);
    end
end

% Helper function for conditional string in sprintf (inline if-else)
function out = ternary(condition, trueVal, falseVal)
    if condition
        out = trueVal;
    else
        out = falseVal;
    end
end
