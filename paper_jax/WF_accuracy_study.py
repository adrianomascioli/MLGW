import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", " *", category=UserWarning)
import numpy as np
import mlgw
from mlgw.GW_generator import GW_generator
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp
import pycbc
from pycbc.conversions import mass1_from_mtotal_q, mass2_from_mtotal_q
import pycbc.filter
from pycbc.types import timeseries
from tqdm import tqdm
import time
import lal
import lalsimulation as lalsim

#Useful functions
def mass1_mass2_from_mtotal_q(mtotal, q):
    mass1 = mass1_from_mtotal_q(mtotal, q)
    mass2 = mass2_from_mtotal_q(mtotal, q)
    return mass1, mass2

#Check if there's an equivalent pycbc function (probably yes)
def get_random_antenna_patterns():

	N  =1
	polarization = np.random.uniform(0, 2*np.pi, N)
	latitude = np.arcsin(np.random.uniform(-1.,1., N)) #FIXME: check this is right!
	longitude = np.random.uniform(-np.pi, np.pi, N)
	
	theta = np.pi/2 - np.asarray(latitude)
	
	F_p = - 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.cos(2*polarization)
	F_p -= np.cos(theta)* np.sin(2*longitude)* np.sin(2*polarization) 
	F_c = 0.5*(1 + np.cos(theta)**2)* np.cos(2*longitude)* np.sin(2*polarization)
	F_c -= np.cos(theta)* np.sin(2*longitude)* np.cos(2*polarization) 

	return F_p, F_c

#physical parameters
M_range = (20., 20.) #total mass
q_range = (1., 10.) #mass ratio 
s1_range = (-0.9, 0.9) #spin1 aligned component
s2_range = (-0.9, 0.9) #spin1 aligned component
d_range = (1., 100.) #luminosity distance
i_range = (0, np.pi) #inclination angle
phi_0_range = (0, 2*np.pi) #phase

range_low  = np.array([M_range[0], q_range[0], s1_range[0], s2_range[0], d_range[0], i_range[0], phi_0_range[0]])
range_high = np.array([M_range[1], q_range[1], s1_range[1], s2_range[1], d_range[1], i_range[1], phi_0_range[1]])

f_low_low = 15. #Hz
f_low_high = 75. #HZ

#waveform generation settings 
s_frequency = 8192 #Hz --> delta_t = 1/s_frequency
delta_t = 1/s_frequency
duration = 2 #s duration in seconds of the signal
f_high = s_frequency/2 #f_nyquist
delta_f = 1/duration
modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

#Summoning mlgw generator in frequency domain
mlgw_gen = GW_generator()


approx = lalsim.SimInspiralGetApproximantFromString("SEOBNRv4PHM")

def lal_WF(theta, f_low):
    hp, hc = lalsim.SimInspiralChooseTDWaveform( 
		theta[0]*lalsim.lal.MSUN_SI, #m1
		theta[1]*lalsim.lal.MSUN_SI, #m2
		0, 0, theta[2], #s1x,y,z
		0, 0, theta[3], #s2x,y,z
		theta[4]*1e6*lalsim.lal.PC_SI, #distante in pc
		theta[5], #inclination
		np.pi/2 - theta[6], #phi_ref
		0., #longAscNodes
		0., #eccentricity
		0., #meanPerAno
		delta_t, # time incremental step
		f_low, # lowest value of freq
		f_low, #some reference value of freq
		lal.CreateDict(), #some lal dictionary
		approx #approx method for the model
    )
    return hp, hc

def lal_modes(theta, f_low):
    hlm = lalsim.SimInspiralChooseTDModes(
        0.,
        delta_t,
        theta[0]*lalsim.lal.MSUN_SI, #m1
        theta[1]*lalsim.lal.MSUN_SI, #m2
        0, 0, theta[2], #s1x,y,z
        0, 0, theta[3], #s2x,y,z
        f_low,
        f_low,
        1e6*lalsim.lal.PC_SI,
        lal.CreateDict(),
        2,
        approx
    )
    return hlm

def lal_extract_22(theta, hlm):

    prefactor = 4.7864188273360336e-20 #GM_sun/c^2 Mpc
    q = theta[0]/theta[1]
    M = theta[0] + theta[1]
    nu = np.divide(q, np.square(1+q))
    amp_prefactor = prefactor*M
    
    h22 = lalsim.SphHarmTimeSeriesGetMode(hlm, 2, 2).data.data/amp_prefactor 

    return h22

#Extracting the parameters:
N_wfs = 10000
np.random.seed(1997)
theta = np.random.uniform(range_low, range_high, size = (N_wfs, 7))

m1, m2 = mass1_mass2_from_mtotal_q(theta[:, 0], theta[:, 1])
theta = np.stack([m1, m2, theta[:,2], theta[:,3], theta[:,4], theta[:,5], theta[:,6]], axis=1 )


#Jaxing 
theta_jax = jnp.asarray(theta, dtype = jnp.float32)

#Extracting the frequencies: 
np.random.seed(1997)
f_lows = np.random.uniform(f_low_low, f_low_high, size = N_wfs)

#MLGW generator
def mlgw_WF(theta, t_grid):
    h = mlgw_gen.get_WF(theta, t_grid, modes = modes)
    return h


#WARMUP

##Preliminary call to LAL 22 mode to extrapolate the time grid
h_lm_wu = lal_modes(theta[0][:], f_lows[0])
h_22_wu = lal_extract_22(theta[0][:], h_lm_wu)

#Extrapolating the time grid
times_len_wu = len(h_22_wu)
times_wu = np.linspace(0. , times_len_wu*delta_t, times_len_wu)
times_wu = times_wu - times_wu[np.argmax(np.abs(h_22_wu))]
#Jaxing
times_wu_jax = jnp.asarray(times_wu, dtype=jnp.float32)

#Warm up call for MLGW:
h_mlgw_test = mlgw_WF(theta_jax[0][:], times_wu_jax)

#Empty list to store mismatch
mismatches = []

#For timing
t0 = time.time()

#Generating the waveform
for i in tqdm(range(theta.shape[0]), desc="WF generations and match computation"):

    #Generating LAL 22 mode to extrapolate the time grid
    h_lal_lm = lal_modes(theta[i][:], f_lows[i])
    h_lal_22 = lal_extract_22(theta[i][:], h_lal_lm)

    #Extrapolating the time grid
    times_len = len(h_lal_22)
    times = np.linspace(0.0, times_len*delta_t, times_len)
    times = times - times[np.argmax(np.abs(h_lal_22))]
    #Jaxing
    times_jax = jnp.asarray(times, dtype=jnp.float32)

    #Generating LAL and MLGW waveforms
    hp_lal, hc_lal = lal_WF(theta[i][:], f_lows[i])
    hp_mlgw, hc_mlgw = mlgw_WF(theta_jax[i][:], times_jax)

    #Unjaxing
    hp_mlgw = np.array(hp_mlgw, dtype=np.float64)
    hc_mlgw = np.array(hc_mlgw, dtype=np.float64)

    #Matching LAL convention
    hc_mlgw = -hc_mlgw

    #Preparing match computation
    F_p, F_c = get_random_antenna_patterns()
    h_lal = F_p * hp_lal.data.data + F_c * hc_lal.data.data
    
    #Loading inside pycbc
    hp_mlgw_pycbc = timeseries.TimeSeries(hp_mlgw, delta_t = delta_t)
    hc_mlgw_pycbc = timeseries.TimeSeries(hc_mlgw, delta_t = delta_t)
    h_lal_pycbc   = timeseries.TimeSeries(h_lal, delta_t = delta_t)

    #Normalization
    hp_mlgw_pycbc = hp_mlgw_pycbc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hp_mlgw_pycbc))
    hc_mlgw_pycbc = hc_mlgw_pycbc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(hc_mlgw_pycbc))
    h_lal_pycbc = h_lal_pycbc / np.sqrt(pycbc.filter.matchedfilter.sigmasq(h_lal_pycbc))
    
    #Matched filtering
    hp_match = pycbc.filter.matchedfilter.matched_filter(h_lal_pycbc, hp_mlgw_pycbc)
    hc_match = pycbc.filter.matchedfilter.matched_filter(h_lal_pycbc, hc_mlgw_pycbc)

    #Cross overlap between the polarizations
    hpc_overlap = pycbc.filter.matchedfilter.overlap_cplx(hp_mlgw_pycbc, hc_mlgw_pycbc, psd=None).real

    #Match maximized over sky-localization
    match = pycbc.filter.matchedfilter.compute_max_snr_over_sky_loc_stat_no_phase(np.array(hp_match), np.array(hc_match), hpc_overlap, hpnorm=1, hcnorm=1)

    #Minimum mismatch
    mismatch = 1- np.max(np.array(match))

    #Storing the mismatch
    mismatches.append(mismatch)

#For timing
t1 = time.time()
total_time = t1 - t0
print("Total time for the loop: {0} s".format(total_time))

mismatches = np.array(mismatches)

to_save = np.hstack([
    theta,
    f_lows.reshape(-1, 1),
    mismatches.reshape(-1, 1)
])

np.savetxt('mismatches.txt', to_save, header="mass1 mass2 s1z s2z d i phi0 theta6 theta7 f_low mismatch")
    
