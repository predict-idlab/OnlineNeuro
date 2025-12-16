% File: receiveMessage.m

function decodedJson = receiveMessage(tcpClientObj)
% receiveMessage Reads a complete data payload from a tcpclient object
%                using a 4-byte length-prefix framing protocol.
%
% Syntax:
%   decodedJson = receiveMessage(tcpClientObj)
%
% Inputs:
%   tcpClientObj - A connected tcpclient object. The Timeout property on
%                  this object determines how long MATLAB will wait for data.
%
% Outputs:
%   decodedJson  - The decoded JSON data as a MATLAB struct or array.
%
% Throws:
%   An error with a specific identifier if reading fails, the connection
%   is closed, or JSON decoding fails.

    % if ~(isa(tcpClientObj, 'matlab.net.tcpclient') && tcpClientObj.Connected)
    %     error('receiveMessage:InputError', 'Input must be a connected tcpclient object.');
    % end

    try
        fprintf('MATLAB receiveMessage: Waiting to read 4-byte header (timeout: %.2f s)...\n', tcpClientObj.Timeout);

        % 1. Read the 4-byte header to get the message length.
        %    The third argument 'uint8' ensures we get bytes.
        headerBytes = read(tcpClientObj, 4, 'uint8');

        if numel(headerBytes) < 4
            error('receiveMessage:ConnectionClosed', ...
                'Connection closed by peer or timed out while reading message header. Received only %d bytes.', numel(headerBytes));
        end

        % 2. Unpack the header to determine the message length.
        %    - typecast converts the four 8-bit bytes into one 32-bit integer.
        %    - swapbytes converts from big-endian (network byte order) to the machine's native order.
        %    - double() converts from uint32 to a standard double for use in read().
        messageLength = double(swapbytes(typecast(headerBytes, 'uint32')));

        fprintf('MATLAB receiveMessage: Header received. Expecting message body of %d bytes.\n', messageLength);

        % Sanity check
        if messageLength > 1e9 % 1 GB limit
            error('receiveMessage:MessageTooLarge', 'Advertised message length (%d bytes) exceeds safety limit.', messageLength);
        end
        if messageLength == 0
            % Handle case of an empty message (e.g., sending "{}")
            decodedJson = jsondecode('{}');
            return;
        end

        % 3. Read exactly 'messageLength' bytes for the message body.
        messageBytes = read(tcpClientObj, messageLength, 'uint8');

        if numel(messageBytes) < messageLength
             error('receiveMessage:ConnectionClosed', ...
                'Connection closed by peer or timed out while reading message body. Expected %d bytes, got %d.', messageLength, numel(messageBytes));
        end

        % 4. Decode the UTF-8 byte array back into a character string.
        jsonString = native2unicode(messageBytes, 'UTF-8');

        fprintf('MATLAB receiveMessage: Message body received. Attempting to decode JSON...\n');

        % 5. Decode the JSON string into a MATLAB data structure.
        decodedJson = jsondecode(jsonString);

        fprintf('MATLAB receiveMessage: JSON decoding successful.\n');

    catch ME
        fprintf('MATLAB receiveMessage: An error occurred: %s\n', ME.message);
        % Add more context to timeout errors
        if strcmp(ME.identifier, 'MATLAB:networklib:tcpclient:read:operationTimeout')
             error('receiveMessage:ReadTimeout', ...
                 'Timeout (%.2f s) occurred while waiting for data from Python. Last operation: %s', ...
                 tcpClientObj.Timeout, ME.stack(1).name);
        else
            rethrow(ME);
        end
    end
end
