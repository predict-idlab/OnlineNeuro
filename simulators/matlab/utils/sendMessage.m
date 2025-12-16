function sendMessage(tcpClientObj, dataToSend)
% sendMessage Serializes MATLAB data to JSON and sends it over a tcpclient
%             connection using a robust 4-byte length-prefix framing protocol.
%
% Syntax:
%   sendMessage(tcpClientObj, dataToSend)
%
% Inputs:
%   tcpClientObj - A connected tcpclient object.
%   dataToSend   - Any MATLAB data that can be encoded by jsonencode
%                  (e.g., struct, cell array, matrix).
%
% Throws:
%   An error if the input is not a connected client or if writing fails.

    % if ~(isa(tcpClientObj, 'matlab.net.tcpclient') && tcpClientObj.Connected)
    %     error('sendMessage:InputError', 'Input must be a connected tcpclient object.');
    % end

    try
        % 1. Serialize the MATLAB data to a JSON string.
        jsonString = jsonencode(dataToSend);

        % 2. Encode the JSON string into a byte array using UTF-8 encoding.
        messageBytes = uint8(unicode2native(jsonString, 'UTF-8'));

        % 3. Get the length of the message in bytes.
        messageLength = numel(messageBytes);
        fprintf('MATLAB sendData: Preparing to send message of %d bytes.\n', messageLength);

        % 4. Pack the length into a 4-byte, big-endian, unsigned integer header.
        %    - Convert length to uint32.
        %    - swapbytes ensures big-endian (network byte order), matching Python's '!'.
        %    - typecast converts the 32-bit integer into four 8-bit bytes (uint8).
        headerBytes = typecast(swapbytes(uint32(messageLength)), 'uint8');

        % 5. Concatenate the header and the message bytes.
        payload = [headerBytes, messageBytes];

        % 6. Send the complete payload.
        write(tcpClientObj, payload);

        fprintf('MATLAB sendData: Successfully sent %d total bytes (%d header + %d message).\n', ...
                numel(payload), numel(headerBytes), messageLength);

    catch ME
        fprintf('MATLAB sendData: An error occurred during sending: %s\n', ME.message);
        rethrow(ME);
    end
end
