function receivedData = readBlocking(tcpipClient)
    % Initialize received data
    receivedData = '';
    
    % Loop until data is received
    while true
        % Check if there are bytes available to read
        if tcpipClient.BytesAvailable > 0
            % Read the available data
            newData = fread(tcpipClient, tcpipClient.BytesAvailable, 'char')';
            
            % Append the new data to the received data
            receivedData = [receivedData, newData];
            
            % Check if the received data contains a newline character (indicating end of message)
            if contains(receivedData, newline)
                % If a complete message is received, break out of the loop
                break;
            end
        end
    end
end