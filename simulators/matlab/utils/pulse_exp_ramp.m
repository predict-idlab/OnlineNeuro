function y = pulse_exp_ramp(t, delay, amplitude, pulse_width, interphase_gap, decay_width, k, ramp_width)
    % Parameters:
    % t               : Time to evaluate
    % delay           : Time delay before pulse onset
    % amplitude       : Pulse amplitude
    % pulse_width     : Width of the pulse
    % interphase_gap : Gap between the pulse-decay and the ramp
    % k               : Decay constant for the exponential decay
    % ramp_width      : Width of the inverted ramp (positive slope)

    % Initialize output
    %Default duration is 5 ms with 0.01 steps
    % dur/dtout = 500
    dtout = 0.01;
    dur = 5;
    time_constant = dur;
    %t = (1:dur/dtout);

    y = 0;

    %% Pulse signal (rectangular)
    pulse_start = delay;
    pulse_end = pulse_start + pulse_width;
    %y((t >= pulse_start) & (t < pulse_end)) = amplitude;
    if t>= pulse_start && t < pulse_end
        y = amplitude;
        return;
    end

    %% Exponential decay following the pulse
    decay_start = pulse_end;
    decay_end = decay_start + decay_width;
    %decay_time = t((t >= decay_start) & (t < decay_end)) - decay_start;

    % Time for the decay to reach eactly zero
    if t>= decay_start && t < decay_end
        decay_time = t-decay_start;
        if k == 1
            %linear_decay = amplitude*(1-decay_time/decay_width);
            %y((t >= decay_start) & (t < decay_end)) = linear_decay;
            y = amplitude * (1-decay_time/decay_width);
            %e_decay=0;
        else
            %exp_decay(5, 1-i/10, 100)
            %e_decay = amplitude * exp_decay(dur, 1-k, decay_width); % Exponential decay
            % Blend linear and exponential based on k
            %y((t >= decay_start) & (t < decay_end)) = e_decay;
            if decay_width/dtout<1
                y = amplitude;
                return
            end
            yvec = amplitude *exp_decay(time_constant, 1-k, decay_width/dtout);
            steps = linspace(decay_start, decay_end, decay_width/dtout+1);
            ix = find(abs(steps-t)<1e-6);
            y = yvec(ix);
        end
        return;
    end
    %% Between pulses
    if t > decay_end && t < decay_end + interphase_gap
        y = 0;
        return;
    end
    %% RAMP
    % Calculate the area of the initial pulse for matching the ramp

    ramp_start = decay_end + interphase_gap;
    ramp_end = ramp_start + ramp_width;

    if t>= ramp_start && t < ramp_end
        %pulse_area = (amplitude * pulse_width) + sum(e_decay);
        pulse_area = (amplitude * pulse_width);

        % Approximate area of exponential decay (integral of exponential)
        if k ==1
            decay_area = amplitude * (decay_width / 2);  % Linear decay area
        else
            yvec = amplitude *exp_decay(time_constant, 1-k, decay_width/dtout);
            decay_area = sum(yvec)/(decay_width/dtout);
        end

        total_area = pulse_area + decay_area;
        max_ramp_value = total_area*2/ramp_width;
        ramp_fraction = (t-ramp_start)/ramp_width;

        y = max_ramp_value*ramp_fraction;

        if amplitude > 0
            y=-y;
        end
        %y((t >= ramp_start) & (t < ramp_end)) = ramp;
        return;

    end

end
