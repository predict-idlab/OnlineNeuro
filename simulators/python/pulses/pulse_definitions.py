import numpy as np
from cajal.common.logging import logger
from cajal.nrn.stimuli import CosineWave, IncreasingTriangular, SineWave, Stimulus
from cajal.units import mA, ms
from cajal.units.utils import strip_units, unitdispatch

# TODO
# Make Conifg Dict Types
# Extend NerveBlock configuration accordingly


class PulseRamp(Stimulus):
    """
    A stimulus class that produces a pulse followed by a decaying exponential,
    optionally balanced with a linear ramp to ensure charge balance.

    Attributes
    ----------
    amp : mA
        Pulse amplitude.
    pw : ms
        Pulse width.
    delay : ms
        Delay before the pulse starts.
    interphase_gap : ms
        Time gap between the pulse and decay/ramp.
    decay_width : ms
        Duration of the decay phase.
    k : float
        Exponential decay parameter.
    tau : float
        Time constant for exponential decay.
    ramp_width : Optional[ms]
        Duration of the ramp phase.
    ramp_amplitude : Optional[V]
        Amplitude of the ramp phase.
    balance_charge : bool
        Whether to balance the total charge with a ramp.
    offset : float
        DC offset added to the stimulus.

    Internal Attributes (Calculated during timecourse if balanced):
    _ramp_width_calc : ms
        The effective ramp width used.
    _ramp_amplitude_calc : mA
        The effective ramp amplitude (peak current) used.

    """

    name = "pulse_ramp"

    @unitdispatch
    def __init__(
        self,
        delay: "ms",
        interphase_gap: "ms",
        decay_width: "ms",
        k: float,
        tau: float = 1.0,
        amplitude: "mA" = None,
        amp: "mA" = None,
        pw: "ms" = None,
        pulse_width: "ms" = None,
        ramp_width: "ms" = None,
        ramp_amplitude: "mA" = None,
        balance_charge: bool = True,
        offset=0,
        *args,
        **kwargs,
    ):
        """
        Initialize the PulseRamp stimulus.

        Parameters
        ----------
        delay : ms
            Delay before pulse onset.
        interphase_gap : ms
            Gap between pulse and decay/ramp.
        decay_width : ms
            Duration of exponential decay.
        k : float
            Decay shape parameter (1 = linear, <1 = exponential).
        tau : float, optional
            Time constant for exponential decay (default 1.0).
        amplitude : mA, optional
            Pulse amplitude. One of `amplitude` or `amp` must be defined.
        amp : mA, optional
            Pulse amplitude. One of `amplitude` or `amp` must be defined.
        pw : ms, optional
            Pulse width. One of `pw` or `pulse_width` must be defined.
        pulse_width : ms, optional
            Pulse width. One of `pw` or `pulse_width` must be defined.
        ramp_width : ms, optional
            Width of linear ramp phase.
        ramp_amplitude : mA, optional
            Amplitude of linear ramp phase.
        balance_charge : bool
            If True, ramp is calculated to balance total charge.
        offset : float
            DC offset added to the stimulus.

        Notes
        -----
            Normally we would use correct linting here, however,
            due to the unit_dispatch this throws warnings either way.
        """
        super(PulseRamp, self).__init__()

        if balance_charge:
            if (ramp_width is None) == (ramp_amplitude is None):
                raise ValueError(
                    "When balance_charge is True, exactly one of ramp_width or ramp_amplitude must be defined."
                )

            # Initialize calculated attributes as None
            self._ramp_width_calc = None
            self._ramp_amplitude_calc = None

            # Store the input attributes (which might be None)
            self._ramp_width_input = ramp_width
            self._ramp_amplitude_input = ramp_amplitude

        else:
            # If not balanced, both must be defined
            if ramp_width is None or ramp_amplitude is None:
                raise ValueError(
                    "When balance_charge is False, both ramp_width and ramp_amplitude must be defined."
                )

            self._ramp_width_calc = ramp_width
            self._ramp_amplitude_calc = ramp_amplitude
            self._ramp_width_input = ramp_width
            self._ramp_amplitude_input = ramp_amplitude

        if (amp is None) == (amplitude is None):
            raise ValueError("Exactly one of amp or amplitude must be defined.")
        if (pw is None) == (pulse_width is None):
            raise ValueError("Exactly one of pw or pulse_width must be defined.")

        self.amp = amp if amp is not None else amplitude
        self.pw = pw if pw is not None else pulse_width

        self.interphase_gap = interphase_gap
        self.decay_width = decay_width
        self.k = k
        self.tau = tau
        self.ramp_width = ramp_width
        self.ramp_amplitude = ramp_amplitude
        self.balance_charge = balance_charge
        self.delay = delay
        self.offset = offset

    def timecourse(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the stimulus timecourse.

        Parameters
        ----------
        t : np.ndarray
            Time vector (ms) at which to evaluate the stimulus.

        Returns
        -------
        np.ndarray
            Stimulus amplitude at each time point.
        """
        tstop = (
            self.pw
            + self.delay
            + self.interphase_gap
            + self.decay_width
            + self.ramp_width
        )
        if tstop >= t[-1]:
            logger.info("The pulse ends after the simulation end time.")

        t_unitless = strip_units(t.copy())
        t_diff = np.diff(t_unitless, prepend=t[0])
        y = np.zeros_like(t_unitless)

        # Pulse phase
        pulse_start = self.delay
        pulse_end = pulse_start + self.pw
        pulse_mask = (t > pulse_start) & (t <= pulse_end)
        y[pulse_mask] += self.amp

        # Decay phase
        decay_start = pulse_end
        decay_end = decay_start + self.decay_width
        decay_mask = (t > decay_start) & (t <= decay_end)

        if np.any(decay_mask):
            decay_time = t[decay_mask] - decay_start

            if self.k == 1:
                linear_decay = self.amp * (1 - decay_time / self.decay_width)
                y[decay_mask] += linear_decay
            else:
                tau_eff = (
                    1.0 / ((1 - self.k) * self.tau)
                    if (1 - self.k) * self.tau != 0
                    else np.inf
                )

                exp_at_t = np.exp(-decay_time / tau_eff)
                exp_at_end = np.exp(-self.decay_width / tau_eff)

                # Scale the decay function to ensure I(0)=amp and I(decay_width)=0
                numerator = exp_at_t - exp_at_end
                denominator = 1.0 - exp_at_end

                if denominator < 1e-8:
                    # Treat as linear if decay is too slow over the given width
                    y[decay_mask] += self.amp * (1 - decay_time / self.decay_width)
                else:
                    y[decay_mask] += self.amp * numerator / denominator

        # Interphase gap
        gap_end = decay_end + self.interphase_gap

        # Ramp phase

        # Ramp calculation if balance_charge
        if self.balance_charge:
            pulse_area = np.sum(y[pulse_mask] * t_diff[pulse_mask]).astype(float)
            decay_area = np.sum(y[decay_mask] * t_diff[decay_mask]).astype(float)
            total_area = pulse_area + decay_area  # Total charge in mA*ms

            # Required ramp area = -total_area
            Q_ramp_required = -total_area
            ramp_sign = -np.sign(total_area)

            # Case 1: Ramp width defined, calculate Amplitude
            if self._ramp_width_input is not None:
                self._ramp_width_calc = self._ramp_width_input

                if self._ramp_width_calc == 0:  # dge case
                    self._ramp_amplitude_calc = 0.0
                else:
                    calculated_amplitude_peak = (
                        2 * np.abs(Q_ramp_required) / self._ramp_width_calc
                    )
                    self._ramp_amplitude_calc = calculated_amplitude_peak * ramp_sign

            # Case 2: Ramp amplitude defined, calculate Width
            elif self._ramp_amplitude_input is not None:
                # Use the magnitude of the input amplitude for calculation
                A_in = self._ramp_amplitude_input

                if A_in == 0:  # Edge case
                    self._ramp_width_calc = 0.0
                    self._ramp_amplitude_calc = 0.0
                else:
                    calculated_width = 2 * np.abs(Q_ramp_required) / np.abs(A_in)
                    self._ramp_width_calc = calculated_width

                    # If the user input amplitude sign is incorrect, we warn and enforce the correct sign
                    if np.sign(A_in) != ramp_sign:
                        logger.warning(
                            f"Ramp amplitude sign ({np.sign(A_in)}) conflicts with required charge balance sign ({ramp_sign}). Enforcing required sign."
                        )
                        self._ramp_amplitude_calc = np.abs(A_in) * ramp_sign
                    else:
                        self._ramp_amplitude_calc = A_in

        ramp_start = gap_end

        effective_ramp_width = self._ramp_width_calc

        ramp_end = ramp_start + effective_ramp_width
        ramp_mask = (t > ramp_start) & (t <= ramp_end)

        if np.any(ramp_mask) and effective_ramp_width > 0:
            ramp_time = t[ramp_mask] - ramp_start
            ramp_fraction = ramp_time / self._ramp_width_calc

            # Linear ramp from 0 up to _ramp_amplitude_calc
            y[ramp_mask] += self._ramp_amplitude_calc * ramp_fraction

        # Offset
        y += self.offset

        return y

    def __mul__(self, scale: float) -> "PulseRamp":
        """
        Scale the amplitude of the pulse and ramp components.

        Parameters
        ----------
        scale : float
            Scaling factor.

        Returns
        -------
        PulseRamp
            New PulseRamp instance with scaled amplitude.
        """

        if self.ramp_amplitude is not None:
            ramp_amplitude = scale * self.ramp_amplitude
        else:
            raise ValueError(
                "Amplitude of PulseRamp component is not defined. This shouldn't occur"
            )

        return PulseRamp(
            amplitude=self.amp * scale,
            pw=self.pw,
            delay=self.delay,
            interphase_gap=self.interphase_gap,
            decay_width=self.decay_width,
            k=self.k,
            tau=self.tau,
            ramp_width=self.ramp_width,
            ramp_amplitude=ramp_amplitude,
            balance_charge=self.balance_charge,
            offset=self.offset,
        )


class IncreasingSine(Stimulus):
    """
    Stimulus combining a sinusoidal waveform with an increasing triangular envelope.
    """

    def __init__(self, amp: float, freq: float, duration: float, delay: float):
        """
        Initialize an IncreasingSine stimulus.

        Parameters
        ----------
        amp : float
            Sinusoid amplitude.
        freq : float
            Frequency in Hz.
        duration : float
            Duration in ms.
        delay : float
            Delay before stimulus onset.
        """
        super().__init__()

        self.amp = amp
        self.freq = freq
        self.duration = duration
        self.delay = delay

        self.sinusoid = SineWave(amp=self.amp, freq=self.freq, delay=self.delay)
        self.triangle = IncreasingTriangular(amp=1, pw=self.duration, delay=self.delay)
        self.stimulus = self.sinusoid * self.triangle

    def timecourse(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the stimulus timecourse.

        Parameters
        ----------
        t : np.ndarray
            Time vector (ms).

        Returns
        -------
        np.ndarray
            Stimulus amplitude at each time point.
        """
        y0 = self.sinusoid.timecourse(t)
        y1 = self.triangle.timecourse(t)

        return y0 * y1


class IncreasingCosine(Stimulus):
    """
    Stimulus combining a cosine waveform with an increasing triangular envelope.
    """

    def __init__(self, amp, freq, duration, delay):
        """
        Initialize an IncreasingCosine stimulus.

        Parameters
        ----------
        amp : float
            Cosine amplitude.
        freq : float
            Frequency in Hz.
        duration : float
            Duration in ms.
        delay : float
            Delay before stimulus onset.
        """
        super().__init__()

        self.amp = amp
        self.freq = freq
        self.duration = duration
        self.delay = delay

        self.sinusoid = CosineWave(amp=self.amp, freq=self.freq, delay=self.delay)
        self.triangle = IncreasingTriangular(amp=1, pw=self.duration, delay=self.delay)
        self.stimulus = self.sinusoid * self.triangle

    def timecourse(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the stimulus timecourse.

        Parameters
        ----------
        t : np.ndarray
            Time vector (ms).

        Returns
        -------
        np.ndarray
            Stimulus amplitude at each time point.
        """
        y0 = self.sinusoid.timecourse(t)
        y1 = self.triangle.timecourse(t)

        return y0 * y1
