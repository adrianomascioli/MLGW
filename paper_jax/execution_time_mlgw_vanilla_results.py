import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import mlgw
from matplotlib import pyplot as plt
import timeit

#physical parameters
M_range = (20., 20.) #total mass
q_range = (1., 10.) #mass ratio 
s1_range = (-0.9, 0.9) #spin1 aligned component
s2_range = (-0.9, 0.9) #spin1 aligned component
d_range = (1, 100.) #luminosity distance
i_range = (0, np.pi) #inclination angle
phi_0_range = (0, 2*np.pi) #phase

range_low  = np.array([M_range[0], q_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]])
range_high = np.array([M_range[1], q_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]])

#waveform generation settings 
s_frequency = 2048 #Hz --> delta_t = 1/s_frequency
delta_t = 1/s_frequency
duration = 2. #s duration in seconds of the signal
len_t_grid = duration/delta_t
t_grid = np.linspace(-duration, 0.0, int(len_t_grid))
#f_low = 10. #starting frequency in Hz
#f_high = s_frequency/2 #f_nyquist
#delta_f = 1/duration
#fs = np.arange(f_low, f_high, delta_f)
#f_ref = f_low
#t_step = 1/(2*2048.)
#modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

#Summoning mlgw generator in time domain
mlgw_gen = mlgw.GW_generator(0)

#Defining the calling function for mlgw 
def mlgw_WF(theta):
    hp, hc = mlgw_gen.get_WF(theta, t_grid)
    return hp, hc

#Warm up calls for the generator, since the first is always the slowest:
theta_mlgw_test = np.array((20., 10., 0.3, 0.2, 100., 0., 0.))

hp_mlgw_warmup, hc_mlgw_warmup = mlgw_WF(theta_mlgw_test)

#Number of waveforms to generate
n_wfs = 10000

#list to append times
times_mlgw = []
theta_par = []

#Generating the waveform
for i in range(n_wfs):
    #Generating theta and converting to m1, m2
    theta = np.random.uniform(low = range_low, high = range_high, size = (7, ))
    theta[:2] = theta[0]*theta[1]/(1+theta[1]), theta[0]/(1+theta[1])
   
    #Actual call to time the generation
    time_mlgw   = timeit.timeit(lambda: mlgw_WF(theta), number=2) / 2

    #Storing the parameters
    theta_par.append(theta)

    #Storing the times
    times_mlgw.append(time_mlgw)

np.savetxt('mlgw_vanilla_times.txt', times_mlgw)
np.savetxt('mlgw_vanilla_thetas.txt', theta_par)