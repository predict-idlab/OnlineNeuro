% File: handshake.m
function response = handshake(tcpClientObj)
% handshake Performs a handshake with the Python server.
%
% Syntax:
%   response = handshake(tcpClientObj)
%
% Inputs:
%   tcpClientObj - A connected tcpclient object.
%
% Outputs:
%   response     - The struct received from the Python server.
%
% Throws:
%   An error if the handshake fails at any step.

    fprintf('MATLAB: Initiating handshake...\n');

    try
        % 1. PREPARE and SEND the initial request
        request = struct();
        request.dummyNumber = randi(1000); % Generate a random number
        request.message = 'Hello from MATLAB';

        fprintf('MATLAB: Sending handshake request with dummyNumber: %d\n', request.dummyNumber);
        disp(tcpClientObj)

        sendMessage(tcpClientObj, request);

        % 2. RECEIVE the response from the server
        fprintf('MATLAB: Waiting for handshake response...\n');
        response = receiveMessage(tcpClientObj);

        disp(response)
        % 3. VALIDATE the response
        if ~isstruct(response) || ~isfield(response, 'message') || ~isfield(response, 'dummyNumber')
            error('initiateHandshake:InvalidResponse', 'Received an invalid or malformed response from the server.');
        end

        if response.dummyNumber ~= request.dummyNumber
            error('initiateHandshake:MismatchError', 'Server did not return the correct random number.');
        end

        fprintf('MATLAB: Handshake successful! Server message: "%s"\n', response.message);

    catch ME
        fprintf('MATLAB: Handshake FAILED.\n');
        rethrow(ME);
    end
end
