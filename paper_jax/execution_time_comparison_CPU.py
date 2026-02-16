import numpy as np
import mlgw
from mlgw.GW_FD_generator import GW_FD_generator
from matplotlib import pyplot as plt
import timeit
import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", " *", category=UserWarning)

#waveform generation settings 
s_frequency = 2048 #Hz --> delta_t = 1/s_frequency
duration = 2 #s duration in seconds of the signal
f_low = 10. #starting frequency in Hz
f_high = s_frequency/2 #f_nyquist
delta_f = 1/duration
fs = jnp.arange(f_low, f_high, delta_f)
f_ref = f_low
#t_step = 1/(2*2048.)
modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

#Jitting the ripple WF generator
@jax.jit
def ripple_WF(theta):
    hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(fs, theta, f_ref)
    return hp, hc

#Summoning mlgw generator in frequency domain
mlgw_gen = GW_FD_generator(duration=duration, sampling_frequency=s_frequency)

#Jitting the MLGW generator
@jax.jit
def mlgw_WF(theta):
    h = mlgw_gen.frequency_domain_strain(theta)
    return h

#Defining calling functions with jax.block_until_ready() to actually measure 
#the execution time of the generator

def block_pytree(pytree):
    leaf = jax.tree_util.tree_leaves(pytree)[0]
    leaf.block_until_ready()

def call_mlgw_WF(theta):
    h = mlgw_WF(theta)
    block_pytree(h)
    return h

def call_ripple_WF(theta):
    hp, hc = ripple_WF(theta)
    block_pytree((hp, hc))
    return hp, hc

#Warm up calls for the jitted functions:
theta_mlgw_test   = jnp.array((10., 1., 0.3, 0.2, 100., 0., 0.))
theta_ripple_test = jnp.array((10., 0.25, 0.3, -0.2, 100, 0., 0., 0.))

h_mlgw = call_mlgw_WF(theta_mlgw_test)
hp_ripple_warmup, hc_ripple_warmup = call_ripple_WF(theta_ripple_test)

import lal
import lalsimulation as lalsim

def lal_WF(theta):
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        theta[0]*lalsim.lal.MSUN_SI, #m1
        theta[1]*lalsim.lal.MSUN_SI, #m2
        0, 0, theta[2], #s1x,y,z
        0, 0, theta[3], #s2x,y,z
        theta[4]*1e6*lalsim.lal.PC_SI, #distante in pc
        theta[5], #inclination
        np.pi/2 - theta[6], #phi_ref
        0., #longAscNodes,
        0., #eccentricity
        0., #meanPerAno
        delta_f,
        f_low,
        f_high,
        f_low,
        lal.CreateDict(),
        lalsim.SEOBNRv4_ROM)
    return hp, hc

#Reading in the thetas and the times generated in mlgw_vanilla_times.ipynb
times_vanilla = np.loadtxt('mlgw_vanilla_times.txt')
theta_vanilla = np.loadtxt('mlgw_vanilla_thetas.txt')

#Adapting theta_vanilla to ripple
m1_van, m2_van, s1z_van, s2z_van, d_van, i_van, phi_0_van = theta_vanilla.T

#For Ripple
theta_vanilla_len = m1_van.shape[0]
Mc_van, eta_van = ms_to_Mc_eta(jnp.array([m1_van, m2_van]))
theta_vanilla_ripple = np.stack([Mc_van, eta_van, s1z_van, s2z_van, d_van, np.zeros(theta_vanilla_len), phi_0_van, i_van], axis=1)

#Lists to store the times 
times_mlgw   = []
times_ripple = []
times_lal    = []

#Generating the waveform
for i in range(theta_vanilla.shape[0]):

    #Jaxing the parameters array
    theta_vanilla_ith        = jnp.array(theta_vanilla[i][:], dtype = jnp.float32)
    theta_vanilla_ripple_ith = jnp.array(theta_vanilla_ripple[i][:], dtype = jnp.float32)

    #Actual call to time the generation
    time_mlgw   = timeit.timeit(lambda: call_mlgw_WF(theta_vanilla_ith), number=10) / 10
    time_ripple = timeit.timeit(lambda: call_ripple_WF(theta_vanilla_ripple_ith), number=10) /  10
    time_lal    = timeit.timeit(lambda: lal_WF(theta_vanilla[i][:]), number=10) / 10

    #Storing the times
    times_mlgw.append(time_mlgw)
    times_ripple.append(time_ripple)
    times_lal.append(time_lal)

times_mlgw = np.array(times_mlgw)
times_ripple = np.array(times_ripple)
times_lal = np.array(times_lal)

#Binning
bins = np.linspace(-3.4, -2.0, 101)

#Plotting 
plt.hist(np.log10(times_ripple), bins=bins, label='IMRPhenomD ripple', alpha=0.5)
plt.hist(np.log10(times_mlgw), bins=bins, label='MLGW JAX', alpha=0.5)
plt.hist(np.log10(times_vanilla), bins=bins, label='MLGW vanilla (TD)', alpha=0.5)
plt.hist(np.log10(times_lal), bins=bins, label='SEOBNRv4_ROM', alpha=0.5)

plt.xlabel('log_10(Execution time)')
plt.ylabel('Counts')
plt.legend()

plt.savefig('execution_time_comparison_CPU.png', dpi=300)

#Plotting the cumulative distribution
plt.hist(np.log10(times_ripple), bins=bins, label='IMRPhenomD ripple', cumulative=True, histtype='step', density = True)
plt.hist(np.log10(times_mlgw), bins=bins, label='MLGW JAX', cumulative=True, histtype='step', density = True)
plt.hist(np.log10(times_vanilla), bins=bins, label='MLGW vanilla (TD)', cumulative=True, histtype='step', density = True)
plt.hist(np.log10(times_lal), bins=bins, label='SEOBNRv4_ROM', cumulative=True, histtype='step', density = True)

#plt.yscale('log')
plt.xlabel('log_10(Execution time)')
plt.ylabel('Cumulative distribution')
plt.legend(loc='lower right')

plt.savefig('execution_time_comparison_CPU_cumulative.png', dpi=300)