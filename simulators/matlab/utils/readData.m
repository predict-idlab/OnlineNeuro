function decodedData = readData(obj)
    % Read data from the TCP/IP connection
    data = readline(obj);  % Read all available bytes
    % Decode the received data (assuming it's JSON)
    decodedData = jsondecode(data);
end
