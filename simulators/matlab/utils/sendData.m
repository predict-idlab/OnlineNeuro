function sendData(dataToSend, tcpipClient, size_lim)
    display(dataToSend)
    jsonData = jsonencode(dataToSend);

    if length(jsonData) > size_lim
        fprintf("Breaking down message\n");
        
        % Calculate number of packages
        pckgs = ceil(length(jsonData) / size_lim);

        for i = 1:pckgs
            fprintf("Package %d of %d\n", i, pckgs);
            start_ix = (i - 1) * size_lim + 1;
            end_ix = min(i * size_lim, length(jsonData));

            % Send JSON chunks properly
            chunk = struct();
            chunk.package = i;
            chunk.data = jsonData(start_ix:end_ix); % Slice JSON string
            chunk.tot_pckgs = pckgs;

            jsonChunk = jsonencode(chunk); % Encode full structure
            %jsonChunk = strcat(jsonChunk, '\n'); % Ensure newline separation
            write(tcpipClient, jsonChunk, 'char');
        end
    else
        %jsonData = strcat(jsonData, '\n');
        write(tcpipClient, jsonData, 'char');
    end
end