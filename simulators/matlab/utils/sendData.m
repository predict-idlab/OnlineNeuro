function sendData(dataToSend, tcpipClient, size_lim)
    display(dataToSend)
    jsonData = jsonencode(dataToSend);
    if length(jsonData)>size_lim 
        fprintf("breaking down message")
        pckgs = floor(length(dataToSend)/size_lim) + 1;
        row_per_pckg = ceil(length(jsonData)/pckgs);

        for i=1:pckgs
            fprintf("pakage %d", i)

            start_ix = (i-1)*row_per_pckg + 1;
            end_ix = min(i*row_per_pckg , length(jsonData));
            chunk = struct();
            chunk.('data') = dataToSend(start_ix:end_ix);
            chunk.('tot_pckgs') = pckgs;
            
            jsonChunk = jsonencode(chunk);
            display(jsonChunk)
            write(tcpipClient, jsonChunk);
        end
    else
        write(tcpipClient, jsonData);
    end

end