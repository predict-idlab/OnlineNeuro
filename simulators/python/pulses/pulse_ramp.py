import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def exp_decay(time_constant, k, steps):
    """
    Exponential decay function.

    Parameters:
    time_constant (float): Time constant for decay.
    k (float): Decay rate.
    steps (int): Number of steps for decay.

    Returns:
    np.ndarray: Exponential decay values.
    """
    return np.exp(-np.linspace(0, time_constant * k, steps))


def pulse_exp_ramp(t, delay, amplitude, pulse_width, interphase_gap, decay_width, k, ramp_width):
    """
    Generate a pulse signal with exponential decay and inverted ramp.

    Parameters:
    t (float): Time to evaluate.
    delay (float): Time delay before pulse onset.
    amplitude (float): Pulse amplitude.
    pulse_width (float): Width of the pulse.
    interphase_gap (float): Gap between the pulse-decay and the ramp.
    decay_width (float): Width of the exponential decay.
    k (float): Decay constant for the exponential decay.
    ramp_width (float): Width of the inverted ramp (positive slope).

    Returns:
    float: Signal value at time t.
    """
    dtout = 0.01
    dur = 5
    time_constant = dur

    y = 0

    # Pulse signal (rectangular)
    pulse_start = delay
    pulse_end = pulse_start + pulse_width
    if (pulse_start <= t) &  (t < pulse_end):
        y = amplitude
        return y

    # Exponential decay following the pulse
    decay_start = pulse_end
    decay_end = decay_start + decay_width
    if decay_start <= t < decay_end:
        decay_time = t - decay_start
        if k == 1:
            y = amplitude * (1 - decay_time / decay_width)
        else:
            if decay_width / dtout < 1:
                y = amplitude
                return y
            yvec = amplitude * exp_decay(time_constant, 1 - k, int(decay_width / dtout))
            steps = np.linspace(decay_start, decay_end, int(decay_width / dtout) + 1)
            ix = np.argmin(np.abs(steps - t))
            y = yvec[ix]
        return y

    # Between pulses
    if decay_end < t < decay_end + interphase_gap:
        y = 0
        return y

    # Ramp
    ramp_start = decay_end + interphase_gap
    ramp_end = ramp_start + ramp_width
    if ramp_start <= t < ramp_end:
        pulse_area = amplitude * pulse_width
        if k == 1:
            decay_area = amplitude * (decay_width / 2)
        else:
            yvec = amplitude * exp_decay(time_constant, 1 - k, int(decay_width / dtout))
            decay_area = np.sum(yvec) / int(decay_width / dtout)
        total_area = pulse_area + decay_area
        max_ramp_value = (total_area * 2) / ramp_width
        ramp_fraction = (t - ramp_start) / ramp_width
        y = max_ramp_value * ramp_fraction
        if amplitude > 0:
            y = -y
        return y

    return y


# Example usage
if __name__ == "__main__":
    delay = 0.5
    amplitude = 1.0
    pulse_width = 1.0
    interphase_gap = 0.5
    decay_width = 1.0
    k = 0.8
    ramp_width = 6

    t = np.linspace(0, 10, 1000)
    signal_value = [pulse_exp_ramp(t_, delay, amplitude, pulse_width, interphase_gap, decay_width, k, ramp_width) for t_
                    in t]
    plt.plot(signal_value)
    plt.show()