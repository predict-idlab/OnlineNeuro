function y = exp_decay(L, k, num_points)
    % exp_decay generates an exponential decay over a fixed length.
    %
    % Parameters:
    %   L          - Length over which the decay occurs
    %   k          - Decay constant controlling the steepness
    %   num_points - Number of points in the output vector
    %
    % Returns:
    %   y - A vector of decayed values from 1 to 0 over the interval [0, L]
   
    % Generate time points (or length steps)
    t = linspace(0, L, num_points);
    
    % Calculate the exponential decay
    y = exp(-k * t);
    
    % Scale the result to ensure it reaches 0 at the end
    y = y - y(end);
    
    % Normalize to ensure the maximum value is 1 at the start
    y = y / y(1);
end