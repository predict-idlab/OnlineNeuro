from typing import TypedDict
from cajal.nrn.stimuli import Stimulus
from cajal.units.utils import unitdispatch, strip_units
from cajal.common.logging import logger
import numpy as np

# TODO
# Make Conifg Dict Types ?
# Extend NerveBlock configuration accordingly
# Plots to visualize things


class PulseRamp(Stimulus):
    """
    A class to produce a pulse followed by a decaying exponenital and balanced with a ramp function.

    """
    name = 'pulse_ramp'

    @unitdispatch
    def __init__(self, delay: 'ms', interphase_gap: 'ms',
                 decay_width: 'ms', k: float,
                 tau: float = 1,
                 amplitude=None, amp=None,
                 pw: 'ms' = None, pulse_width: 'ms' = None,
                 ramp_width: 'ms' = None,
                 ramp_amplitude: 'V' = None, balanced_charge: bool = True,
                 offset=0, *args, **kwargs):
        """
        @param delay:
        @param interphase_gap:
        @param decay_width:
        @param k:
        @param tau:
        @param amplitude:
        @param amp:
        @param pw:
        @param pulse_width:
        @param ramp_width:
        @param ramp_amplitude:
        @param balanced_charge:
        @param offset:
        @param args:
        @param kwargs:
        """
        super(PulseRamp, self).__init__()
        if balanced_charge:
            if (ramp_width is None) == (ramp_amplitude is None):
                raise ValueError(
                    'When balanced_charge is True, exactly one of ramp_width or ramp_amplitude must be defined.'
                )

        if (amp is None) == (amplitude is None):
            raise ValueError(
                'Exactly one of amp or amplitude must be defined.'
            )
        if (pw is None) == (pulse_width is None):
            raise ValueError(
                'Exactly one of pw or pulse_width must be defined.'
            )

        self.amp = amp if amp is not None else amplitude
        self.pw = pw if pw is not None else pulse_width
        self.interphase_gap = interphase_gap
        self.decay_width = decay_width
        self.k = k
        self.tau = tau
        self.ramp_width = ramp_width
        self.ramp_amplitude = ramp_amplitude
        self.balanced_charge = balanced_charge
        self.delay = delay
        self.offset = offset

    def timecourse(self, t):
        tstop = self.pw + self.delay + self.interphase_gap + self.decay_width + self.ramp_width
        if tstop >= t[-1]:
            logger.info('The pulse ends after the simulation end time.')

        y = strip_units(np.zeros_like(t))
        # Pulse phase
        pulse_start = self.delay
        pulse_end = pulse_start + self.pw
        pulse_mask = (t > pulse_start) & (t <= pulse_end)
        y[pulse_mask] += self.amp

        # Decay phase
        decay_start = pulse_end
        decay_end = decay_start + self.decay_width
        decay_mask = (t > decay_start) & (t <= decay_end)
        decay_time = t[decay_mask] - decay_start

        if np.any(decay_mask):
            if self.k == 1:
                linear_decay = self.amp * (1 - decay_time / self.decay_width)
                y[decay_mask] += linear_decay
            else:
                # Exponential decay
                raw_exp_decay = self.amp * np.exp(-decay_time * (1 - self.k) * self.tau)
                normalized_exp_decay = raw_exp_decay - raw_exp_decay[-1]  # End to zero
                normalized_exp_decay *= self.amp / np.abs(normalized_exp_decay[0])  # Scale to the desired `amp`
                y[decay_mask] += normalized_exp_decay

        # Interphase gap
        gap_start = decay_end
        gap_end = gap_start + self.interphase_gap

        y += 0 * ((t > gap_start) & (t <= gap_end))

        # Ramp phase
        ramp_start = gap_end
        ramp_end = ramp_start + self.ramp_width
        ramp_mask = (t > ramp_start) & (t <= ramp_end)

        if np.any(ramp_mask):
            if self.balanced_charge:
                # TODO, add logic in case amplitude is given (then we define the width)
                t_diff = np.diff(t, prepend=t[0])

                pulse_area = np.sum(y[pulse_mask] * t_diff[pulse_mask])
                decay_area = np.sum(y[decay_mask] * t_diff[decay_mask])

                total_area = pulse_area + decay_area
                max_ramp_value = 2 * total_area / self.ramp_width

                ramp_fraction = (t[ramp_mask] - ramp_start) / self.ramp_width
                y[ramp_mask] += -max_ramp_value * ramp_fraction if self.amp > 0 else max_ramp_value * ramp_fraction

            else:
                ramp_fraction = (t[ramp_mask] - ramp_start) / self.ramp_width
                y[
                    ramp_mask] += -self.ramp_amplitude * ramp_fraction if self.amp > 0 else self.ramp_amplitude * ramp_fraction

        y += self.offset

        return y

    def __mul__(self, scale):
        """
        When scaling, we adjust the amplitude of the Pulse and the Ramp (if given),
        Notice that we do not adjust the exponential decay, and this remains dependent on k and tau.
        @param scale:
        @return:
        """

        if self.ramp_amplitude is not None:
            ramp_amplitude = scale * self.ramp_amplitude

        return PulseRamp(amplitude=self.amp * scale, pw=self.pw,
                         delay=self.delay, interphase_gap=self.interphase_gap,
                         decay_width=self.decay_width, k=self.k, tau=self.tau,
                         ramp_width=self.ramp_width,
                         ramp_amplitude=ramp_amplitude, balanced_charge=self.balanced_charge,
                         offset=self.offset)
