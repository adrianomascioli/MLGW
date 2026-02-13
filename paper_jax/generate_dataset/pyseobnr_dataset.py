import argparse
import sys
import os
import numpy as np
import pyseobnr
from pyseobnr.generate_waveform import GenerateWaveform
from tqdm import tqdm
import scipy.signal

#############################
#OPTIONS
#############################

parser = argparse.ArgumentParser(__doc__)

parser.add_argument(
	"--n-wfs", type = int, required = True,
	help="Number of WFs to generate")

parser.add_argument(
	"--n-grid", type = int, required = True,
	help="Number of grid points")

parser.add_argument(
	"--modes", type = str, required = False, nargs = '+', default = ['22'],
	help="the modes to generate, in the format lm. Example: --modes 22 21")

parser.add_argument(
	"--basefilename", type = str, required = True,
	help="a base filename to save the datasets at (each mode will be saved at basefilename.lm)")

parser.add_argument(
	"--t-coal", type = float, required = True,
	help="time to coalescence for each generated WF (in s/M_sun)")

parser.add_argument(
	"--t-step", type = float, required = True,
	help="Time step for the input waveform in the given model")

parser.add_argument(
	"--alpha", type = float, required = True, default = 0.5,
	help="distortion parameter alpha for the time grid (between 0 and 1); a value of 0.3-0.5 is advised.")

parser.add_argument(
	"--approximant", type = str, required = True,
	help="Time domain waveform approximant to be used among those available on pyseobnr")

parser.add_argument(
	"--q-range", type = float, required = True, nargs = 2,
	help="Mass ratio range in which WF are generated")

parser.add_argument(
	"--s1-range", type = float, required = False, nargs = 2, default = (-0.9, 0.9),
	help="S1 range in which WF are generated")

parser.add_argument(
	"--s2-range", type = float, required = False, nargs = 2, default = (-0.9, 0.9),
	help="S2 range in which WF are generated")

parser.add_argument(
	"--m2-range", type = float, required = False, nargs = 2, default = None,
	help="Range for the m2 quantity. If not given, the total mass of the system is set to be 20 M_sun")

args = parser.parse_args()

#############################
#HELPER FUNCTIONS and CONSTANTS
#############################

prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)

def f_minimum(tau, q, M):
	"""
	Computes the approximate minimum frequency of a waveform, given the total mass, mass ratio and the length of the reduced time grid (s/M_sun)
	"""
	#return (151*(tau_min)**(-3./8.) * (((1+q)**2)/q)**(3./8.))/M
	return 151*np.power(np.square(1+q)/(q*np.abs(tau)), 3/8)/M

def locate_peak(amp, start = 0.3):
	"""
	Given a time grid and an amplitude of the mode, it returns the peak of the amplitude.
	Input:
		amp (D,)		amplitude
		start			points to be skipped at begining of the amplitude (in fraction of total points)
	Output:
		argpeak			index at which the amplitude has a peak
	"""
	assert amp.ndim == 1
	id_start = int(len(amp)*start)
	extrema = scipy.signal.argrelextrema(np.abs(amp[id_start:]), np.greater)
	if len(extrema[0]):
		return extrema[0][0]+id_start
	else:
		return np.argmax(np.abs(amp[id_start:]))+id_start


##################################
#FUNCTION TO GENERATE THE DATASET
##################################

def create_dataset_TD(N_data, N_grid, modes, basefilename, t_coal = 0.5, q_range = (1.,10.), m2_range = None, s1_range = (-0.9,0.9), s2_range = (-0.9,0.9), t_step = 1e-5, alpha = 0.35, approximant = "SEOBNRv5HM"):

	#checking if N_grid is fine
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")

	if isinstance(m2_range, tuple):
		D_theta = 4 #m2 must be included as a feature
	else:
		D_theta = 3
	
	######setting the time grid
	time_grid_list = []
	t_end_list = []

	for mode in modes:
		t_end_list.append(5.2e-4) #estimated maximum time for ringdown: WF will be killed after that time

	print("Generating modes: "+str(modes))

	#creating time_grid
	for i,mode in enumerate(modes):
		
		#Generating the time grid by hand from the time of coalescence and number of points set by the user
		time_grid = np.linspace(-np.power(np.abs(t_coal), alpha), np.power(t_end_list[i], alpha), N_grid)
		time_grid = np.multiply( np.sign(time_grid) , np.power(np.abs(time_grid), 1./alpha))

		#Adding 0 to the time grid
		index_0 = np.argmin(np.abs(time_grid))
		time_grid[index_0] = 0.

		#Storing 
		time_grid_list.append(time_grid)

		#setting the frequency of coalescence for generating a waves
		if np.abs(t_coal) < 0.05:
			t_coal_freq = 0.05
		else:
			t_coal_freq = np.abs(t_coal)



	#####create a list of buffer to save the WFs
	buff_list = []
	
	for i, mode in enumerate(modes):
		filename = basefilename+'.'+str(mode[0])+str(mode[1])
		
		if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
			
			filebuff = open(filename,'w')
			print("New file ", filename, " created")
			time_header = np.concatenate((np.zeros((D_theta,)), time_grid_list[i], time_grid_list[i]) )[None,:]
			np.savetxt(filebuff, time_header, header = "#Mode:"+ str(mode[0])+str(mode[1]) +"\n# row: theta "+str(D_theta)+" | amp (None,"+str(N_grid)+")| ph (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | t_coal ="+str(t_coal)+" | t_step ="+str(t_step)+" | q_range = "+str(q_range)+" | m2_range = "+str(m2_range)+" | s1_range = "+str(s1_range)+" | s2_range = "+str(s2_range), newline = '\n')
			
		else:
			filebuff = open(filename,'a')
			
		buff_list.append(filebuff)

	
	###Generation of the waveform

	for n_WF in tqdm(range(N_data), desc = 'Generating dataset'):

		#Setting the values for the intrinsic parameters
		#m2
		if isinstance(m2_range, (tuple, list)):
			m2 = np.random.uniform(m2_range[0],m2_range[1])
		elif m2_range is not None:
			m2 = float(m2_range)

		#q
		if isinstance(q_range, (tuple, list)):
			#biased q distribution for the boundaries
			x = np.random.uniform()
			if x < 0.3:
				q = np.min(np.random.uniform(low=q_range[0],high=q_range[1],size=5))
			elif 0.3 <= x < 0.8:
				q = np.random.uniform(low=q_range[0],high=q_range[1])
			else:
				q = np.max(np.random.uniform(low=q_range[0],high=q_range[1],size=5))
		else:
			q = float(q_range)

		#spins
		if isinstance(s1_range, (tuple, list)):
			spin1z = np.random.uniform(s1_range[0],s1_range[1])
		else:
			spin1z = float(s1_range)
		if isinstance(s2_range, (tuple, list)):
			spin2z = np.random.uniform(s2_range[0],s2_range[1])
		else:
			spin2z = float(s2_range)

		#masses
		if m2_range is None:
			m2 = 20. / (1+q)
			m1 = q * m2
		else:
			m1 = q* m2
   
		nu = np.divide(q, np.square(1+q)) #symmetric mass ratio

		#computing f_min
		f_min = .9*f_minimum(t_coal_freq, q, m1+m2)
		#f_minimum is the right scaling formula for frequency in order to get always the right reduced time
		#this should be multiplied by a prefactor (~1) for dealing with some small variation due to spins

		if isinstance(m2_range, tuple):
			temp_theta = [m1, m2, spin1z, spin2z]
		else:
			temp_theta = [m1/m2, spin1z, spin2z]

		#Generating
		amp_list, ph_list = [None for i in range(len(modes))],[None for i in range(len(modes))]

		#Defining the dictionary parameter to provide to pyseobnr
		#Building the input for the WF generator
		params_dict = {
			"mass1": m1,
			"mass2": m2,
			"spin1x": 0.,
			"spin1y": 0.,
			"spin1z": spin1z,
			"spin2x": 0.,
			"spin2y": 0.,
			"spin2z": spin2z,
			"deltaT": t_step,
			"f22_start": f_min,
			"phi_ref": 0.,
			"distance": 1.,
			"approximant": approximant,
			"ModeArray": modes
			}

		#Calling the generator
		wfs_gen = GenerateWaveform(params_dict)

		#Generating the modes
		_, hlm = wfs_gen.generate_td_modes()

		amp_prefactor = prefactor*(m1+m2)/1.
		#Extracting amplitude and phase

		for i, lm in enumerate(modes):
			temp_amp = np.abs(hlm[lm]) / amp_prefactor / nu
			temp_ph = np.unwrap(np.angle(hlm[lm]))
			amp_list[i] = temp_amp
			ph_list[i] = temp_ph
			if (lm[0], lm[1]) == (2,2): #get grid
				argpeak = locate_peak(temp_amp) #aligned at the peak of the 22
		time_full = np.linspace(0.0, len(temp_amp)*t_step, len(temp_amp)) #time grid at which wave is computed


		#computing waves to the chosen std grid by interpolating and saving to file
		for i in range(len(amp_list)):
			temp_amp, temp_ph = amp_list[i], ph_list[i]
			temp_amp = np.interp(time_grid_list[i], time_full, temp_amp)
			temp_ph = np.interp(time_grid_list[i], time_full, temp_ph)
			temp_ph = temp_ph - temp_ph[0] #all phases are shifted by a constant to make sure every wave has 0 phase at beginning of grid
			to_save = np.concatenate((temp_theta, temp_amp, temp_ph))[None,:] #(1,D)
			np.savetxt(buff_list[i], to_save)
	
		del temp_theta, temp_amp, temp_ph
		del amp_list, ph_list
		del to_save
		del hlm
			
	filebuff.close()
	return

##################################
#CALLING THE FUNCTION
##################################
modes = []
	#validating modes
for lm in args.modes:
	try:
		assert len(lm)==2
		l, m = int(lm[0]), int(lm[1])
		assert l>=m
		assert m>0
	except (AssertionError, ValueError):
		raise ValueError("Wrong format for the mode '{}'".format(lm))
	modes.append((l,m))

create_dataset_TD(args.n_wfs, N_grid = args.n_grid, modes = modes, basefilename = args.basefilename,
	t_coal = args.t_coal, q_range = args.q_range, m2_range = args.m2_range, s1_range = args.s1_range, s2_range = args.s2_range,
	t_step = args.t_step, alpha = args.alpha,
	approximant = args.approximant)