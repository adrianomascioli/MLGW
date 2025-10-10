import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from functools import partial
from scipy.signal.windows import tukey
from .GW_generator import mode_generator_NN, GW_generator
#from bilby.gw.conversion import bilby_to_lalsimulation_spins

solar_mass = 1.988409870698050731911960804878414216e30



def create_frequency_array_bilby(sampling_frequency, duration):
    """ Create a frequency series with the correct length and spacing.

    Parameters
    ==========
    sampling_frequency: float
    duration: float

    Returns
    =======
    array_like: frequency series

    """
    number_of_samples = int(np.round(duration * sampling_frequency))
    number_of_frequencies = int(np.round(number_of_samples / 2) + 1)

    return np.linspace(start=0,
                       stop=sampling_frequency / 2,
                       num=number_of_frequencies)

def get_sampling_frequency_bilby(time_array):
    """
    Calculate sampling frequency from a time series

    Attributes
    ==========
    time_array: array_like
        Time array to get sampling_frequency from

    Returns
    =======
    Sampling frequency of the time series: float

    Raises
    ======
    ValueError: If the time series is not evenly sampled.

    """
    tol = 1e-10
    if np.ptp(np.diff(time_array)) > tol:
        raise ValueError("Your time series was not evenly sampled")
    else:
        return np.round(1. / (time_array[1] - time_array[0]), decimals=_TOL)

def get_sampling_frequency_and_duration_from_time_array_bilby(time_array):
    """
    Calculate sampling frequency and duration from a time array

    Attributes
    ==========
    time_array: array_like
        Time array to get sampling_frequency/duration from: array_like

    Returns
    =======
    sampling_frequency, duration: float, float

    Raises
    ======
    ValueError: If the time_array is not evenly sampled.

    """

    sampling_frequency = get_sampling_frequency_bilby(time_array)
    duration = len(time_array) / sampling_frequency
    return sampling_frequency, duration

class GW_FD_generator:


  def __init__(self, duration, sampling_frequency, final_time=0.01, modes=(2,2), alpha_left=0.1, alpha_right=0.001):
  
    self.duration=duration
    self.sampling_frequency=sampling_frequency
    self.modes=modes
    self.final_time_mlgw=final_time
    self._create_time_array_for_mlgw()
    self.gwgen=GW_generator()
    self.alpha_left=alpha_left
    self.alpha_right=alpha_right
    self.frequency_array=create_frequency_array_bilby(self.sampling_frequency, self.duration)


  def create_time_array_for_mlgw(self, sampling_frequency, duration):
    """

    Parameters
    ==========
    sampling_frequency: float
    duration: float
    starting_time: float, optional

    Returns
    =======
    float: An equidistant time series given the parameters

    """
    final_time=self.final_time_mlgw
    number_of_samples = int(duration * sampling_frequency)
    return jnp.linspace(start=final_time-duration,
                       stop=final_time - 1 / sampling_frequency,
                       num=number_of_samples)

  def _create_time_array_for_mlgw(self):
    self.time_array_mlgw=self.create_time_array_for_mlgw(self.sampling_frequency,self.duration)

  def tukey_asymmetric(self, N):
    """
    Asymmetric Tukey window:
    - alpha_left: fraction of window tapered at the start
    - alpha_right: fraction of window tapered at the end
    """
    
    x = jnp.linspace(0, 1, N)
    w = jnp.ones_like(x)
    alpha_left=self.alpha_left
    alpha_right=self.alpha_right

    left_part = 0.5 * (1 + jnp.cos(2 * jnp.pi * (x / alpha_left - 0.5)))
    right_part = 0.5 * (1 + jnp.cos(2 * jnp.pi * ((x - 1) / alpha_right + 0.5)))

    w = jnp.where(x < alpha_left/2, left_part, w)
    w = jnp.where(x > 1 - alpha_right/2, right_part, w)

    return w


  def tukey_waveform_mlgw_jax_TD_dur_samp(self, hp,hc):
    lenght=len(hp)
    assert lenght == len(hc)
    tukey_window=self.tukey_asymmetric(lenght)
    h_p_tukey=hp*tukey_window
    h_c_tukey=hc*tukey_window
    return h_p_tukey, h_c_tukey

  def nfft_jax(self, time_domain_strain, sampling_frequency):
    """ Perform an FFT while keeping track of the frequency bins. Assumes input
        time series is real (positive frequencies only).

    Parameters
    ==========
    time_domain_strain: array_like
        Time series of strain data.
    sampling_frequency: float
        Sampling frequency of the data.

    Returns
    =======
    frequency_domain_strain, frequency_array: (array_like, array_like)
        Single-sided FFT of time domain strain normalised to units of
        strain / Hz, and the associated frequency_array.

    """
    frequency_domain_strain = jnp.fft.rfft(time_domain_strain)
    frequency_domain_strain /= sampling_frequency

    frequency_array = jnp.linspace(
        0, sampling_frequency / 2, len(frequency_domain_strain))

    return frequency_domain_strain, frequency_array


  def infft_jax(self, frequency_domain_strain, sampling_frequency):
    """ Inverse FFT for use in conjunction with nfft.

    Parameters
    ==========
    frequency_domain_strain: array_like
        Single-sided, normalised FFT of the time-domain strain data (in units
        of strain / Hz).
    sampling_frequency: int, float
        Sampling frequency of the data.

    Returns
    =======
    time_domain_strain: array_like
        An array of the time domain strain
    """
    time_domain_strain_norm = jnp.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain


  def waveform_td(self, theta_):

    h_p,h_c = self.gwgen.get_WF(theta_,self.time_array_mlgw, modes=self.modes)
    return h_p,h_c

  def waveform_td_windowed(self,theta_):
      h_p,h_c = self.waveform_td(theta_)
      h_plus_squeezed = jnp.squeeze(h_p)
      h_cross_squeezed = jnp.squeeze(h_c)
      h_p_tukey, h_c_tukey= self.tukey_waveform_mlgw_jax_TD_dur_samp(h_plus_squeezed,h_cross_squeezed)
      return {"plus": h_p_tukey, "cross": h_c_tukey}

  def frequency_domain_strain(self, theta_):
    h = self.waveform_td_windowed(theta_)
    h_p=h["plus"]
    h_c=h["cross"]
    h_p_f=self.nfft_jax(h_p,self.sampling_frequency)
    h_c_f=self.nfft_jax(h_c,self.sampling_frequency)
    return {"plus": h_p_f, "cross": h_c_f}
