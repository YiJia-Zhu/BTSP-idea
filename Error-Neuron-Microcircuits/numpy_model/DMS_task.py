import os
import numpy as np
import multiprocess as mp
N_MAX_PROCESSES = 12 # defined by compute setup

from src.microcircuit import *
from src.save_exp import *
import src.plot_exp as plot_exp
import src.save_exp as save_exp
from src.init_MC import init_weights
import sys
import logging
from functools import partial, update_wrapper

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
					level=logging.INFO)

import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import warnings
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

try:
	plt.style.use('matplotlib_style.mplstyle')
except:
	plt.rcParams['text.usetex'] = False
	plt.rc('font', size=10,family='serif')

PATH = 'experiments/DMS_plots/'

# common parameters
dt = 0.01 # in ms
tauxi = None
Tpres = 0.0
epochs = 10
N_SEEDS = 10
train_samples = 50
val_samples = 50
test_samples = 50

# Init model: weights, voltages
layers = [1, 2, 1]
error_layers = [0] + layers[1:]

# conductances
gl = 0.3 #0.03 # increase this to decrease taueff
gbas = 0.1
gapi = 0.06
gden = 0.1
gnI = 0.06
gntgt = 0.06

# noise
WT_noise = 0.5
init_WPP_range = [-1, 1]
init_BII_range = [ 1, 1]
init_BPI_range = [-0.3, 0.3]
init_WIP_range = [-0.3, 0.3]

# learning rates in 1/ms
eta_fw = [4000.0,4000.0]
eta_bw = [0.0] * (len(layers) - 2)
eta_IP = [0.0] * (len(layers) - 1)
eta_PI = [0.0] * (len(layers) - 1)

params =   {'layers': layers,
			'error_layers': error_layers,
			'init_WPP_range': init_WPP_range,
			'init_BII_range': init_BII_range,
			'init_BPI_range': init_BPI_range,
			'init_WIP_range': init_WIP_range,
			"WT_noise": WT_noise,
			"init_WIP_identity": True,
			"init_BPI_identity": True,
			"init_realistic_connectivity": False,
			"fw_connection_mode": 'layered',
			"bw_connection_mode": 'layered',
			"mc_model": "errormc"}

# wrapper for passing functions to partial, keeping __name__ attribute
# needed for noise on activation function
def wrapped_partial(func, *args, **kwargs):
	partial_func = partial(func, *args, **kwargs)
	update_wrapper(partial_func, func)
	return partial_func

def binary_output(x,pre_offset=0.5,post_offset=0):
	return np.heaviside(x-pre_offset, 0) + (1-np.heaviside(x-pre_offset, 0)) * (post_offset)

def generate_inputs_and_targets(N_samples, seed, first_stim=None, second_stim=None):
	rng = np.random.default_rng(seed)
	
	input_arr = []
	target_arr = []
	
	for _ in range(N_samples):
 
		# input rates:
		T_DELAY = rng.integers(50,250)
		# T_DELAY = 100
		# fixation + first stim + delay + second stim
		inputs = np.zeros(250+250+T_DELAY+250)
		# first stim
		fstm = rng.choice([-1,1]) if first_stim == None else first_stim
		fstm = np.array(fstm)
		inputs[250:500] = fstm.copy()
		# second stim
		sstm = rng.choice([-1,1]) if second_stim == None else second_stim
		sstm = np.array(sstm)
		inputs[250+250+T_DELAY:] = sstm.copy()
	   
		input_arr.append(inputs)
		
		target = np.abs(fstm + sstm) -1
		target_arr.append(target)
	
	return input_arr, target_arr


def init_mc(seed):

	rng = np.random.default_rng(seed)

	# activation function
	act_func_list = [soft_relu,linear]

	# initialise voltages
	uP_init = []
	for i in range(1, len(layers)):
		uP_init.append(rng.normal(0, 0, size=layers[i]))
		
	uI_init = []
	for i in range(1, len(layers)):
		uI_init.append(rng.normal(0, 0, size=error_layers[i]))
		
	# # forward pp weights: skip connections
	# WPP_init = []
	# for i in range(len(layers)-1):
	# 	WPP_init.append(rng.uniform(-1, 1, size=(layers[i+1], layers[i])))
	# # WPP_init = init_nonhierarchical_WPP([-1,1], layers, [0,0])

	# # rep to error: connects k to k
	# WIP_init = []
	# for i in range(1, len(layers)):
	# #     WIP_init.append(rng.uniform(-1, 1, size=(layers[i], layers[i])))
	# 	WIP_init.append(np.eye(N=layers[i], M=layers[i]))

	# # backwards error to error: skip connections
	# BII_init = []
	# for i in range(1, len(error_layers)-1):
	# 	BII_init.append(rng.uniform(-1, 1, size=(error_layers[i], error_layers[i+1])))
	# # BII_init = init_nonhierarchical_BII([-1,1], error_layers, [0,0])


	# # error to rep: connects k to k
	# BPI_init = []
	# for i in range(1, len(layers)):
	# #     BPI_init.append(rng.uniform(-1, 1, size=(layers[i], layers[i])))
	# 	BPI_init.append(np.eye(N=layers[i], M=layers[i]))

	error_layers_init = error_layers[1:]

	WPP_init, WIP_init, BPP_init, BPI_init, BII_init = init_weights(layers,
																	error_layers_init,
																	params,
																	seed=seed,
																	mc_model='errormc',
																	teacher=False)

	rep_lookahead = [False, False]
	error_lookahead = [True, True]


	mc = errormc_model(fw_connection_mode='layered',
					 bw_connection_mode='layered',
					 dWPP_use_activation=False,
					 varphi_transfer=True,
					 dt=dt, 
					 dtxi=dt,
					 tauHP=dt,
					 tauLO=dt,
					 Tpres=Tpres,
					 noise_scale=dt,
					 alpha=None,
					 pyr_hi_pass=False,
					 dWPP_low_pass=False,
					 noise_type=None,
					 noise_mode='vapi',
					 model='BP', 
					 activation=act_func_list,
					 error_activation=linear,
					 layers=layers,
					 uP_init=uP_init, 
					 uI_init=uI_init,
					 WPP_init=WPP_init, 
					 WIP_init=WIP_init, 
					 BII_init=BII_init, 
					 BPI_init=BPI_init,
					 gl=gl, 
					 gden=gden, 
					 gbas=gbas, 
					 gapi=gapi, 
					 gnI=gnI, 
					 gntgt=gntgt,
					 eta_fw=eta_fw, 
					 eta_bw=eta_bw, 
					 eta_PI=eta_PI, 
					 eta_IP=eta_IP,
					 rep_lookahead=rep_lookahead,
					 error_lookahead=error_lookahead,
					 WT_noise=WT_noise)

	mc.uP_init = deepcopy_array(uP_init)

	mc.mc_model = 'errormc'
	mc.input_signal = None
	mc.epochs = epochs

	mc.rec_per_steps = 100
	mc.rec_MSE = True
	mc.rec_rate_MSE = True
	mc.rec_error = False
	mc.rec_input = True
	mc.rec_target = False

	mc.rec_WPP = True
	mc.rec_WIP = False
	mc.rec_BPP = True
	mc.rec_BII = False
	mc.rec_BPI = False

	mc.rec_dWPP = False
	mc.rec_dWIP = False
	mc.rec_dBPP = False
	mc.rec_dBII = False
	mc.rec_dBPI = False

	mc.rec_uP = False
	mc.rec_uP_breve = True
	mc.rec_rP_breve = True
	mc.rec_rP_breve_HI = False
	mc.rec_uI = False
	mc.rec_uI_breve = True
	mc.rec_rI_breve = True

	mc.rec_vbas = False
	mc.rec_vapi = False
	mc.rec_vapi_noise = False
	mc.rec_noise = False
	mc.rec_epsilon = False
	mc.rec_epsilon_LO = False
	mc.rec_lat_mismatch = False

	mc.init_record(rec_per_steps=mc.rec_per_steps,
		rec_MSE=mc.rec_MSE,
		rec_error=mc.rec_error,
		rec_input=mc.rec_input,
		rec_target=mc.rec_target,
		rec_WPP=mc.rec_WPP,
		rec_WIP=mc.rec_WIP,
		rec_BPP=mc.rec_BPP,
		rec_BII=mc.rec_BII,
		rec_BPI=mc.rec_BPI,
		rec_dWPP=mc.rec_dWPP,
		rec_dWIP=mc.rec_dWIP,
		rec_dBPP=mc.rec_dBPP,
		rec_dBII=mc.rec_dBII,
		rec_dBPI=mc.rec_dBPI,
		rec_uP=mc.rec_uP,
		rec_uI=mc.rec_uI,
		rec_uP_breve=mc.rec_uP_breve,
		rec_uI_breve=mc.rec_uI_breve,
		rec_rI_breve=mc.rec_rI_breve,
		rec_rP_breve=mc.rec_rP_breve,
		rec_rP_breve_HI=mc.rec_rP_breve_HI,
		rec_vbas=mc.rec_vbas,
		rec_vapi=mc.rec_vapi,
		rec_vapi_noise=mc.rec_vapi_noise,
		rec_epsilon=mc.rec_epsilon,
		rec_noise=mc.rec_noise,
		rec_epsilon_LO=mc.rec_epsilon_LO,
		rec_lat_mismatch=mc.rec_lat_mismatch)

	mc.settling_time = 0

	# override output layer activation
	mc.activation[-1] = wrapped_partial(tanh,offset=0.5,slope=20)
	# mc.activation[-1] = wrapped_partial(binary_output,pre_offset=0.5,post_offset=-1)

	# set switches for lookahead enabled/disabled
	mc.rep_lookahead = [False, False]
	mc.error_lookahead = [True, True]

	return mc


def train_model(mc, input_set, target_set):

	# iterate over whole dataset
	# # shuffle dataset
	# input_set, target_set = unison_shuffled_copies(input_set, target_set, mc.rng)

	for inputs, target in zip(input_set, target_set):
		# reset voltage
		mc.set_voltages(uP=mc.uP_init, uP_breve=mc.uP_init)
		# iterate over time series in input
		for input_frame in inputs:
			mc.evolve_system(r0=[input_frame], u_tgt=[np.array([target])], learn_weights=False)
		# calculate error on last step
		mc.evolve_system(r0=[input_frame], u_tgt=[np.array([target])])
		# logging.info("Train", mc.rP_breve[-1], target)
		# print("rP_breve", mc.rP_breve)
		# print("uI", mc.uI_breve)
		# print("dWPP", mc.dWPP)

	return mc

def test_model(mc, input_set, target_set, validation=False, testing=True):

	# iterate over whole dataset
	for inputs, target in zip(input_set, target_set):
		# reset voltage
		mc.set_voltages(uP=mc.uP_init, uP_breve=mc.uP_init)
		# iterate over time series in input
		for input_frame in inputs:
			mc.evolve_system(r0=[input_frame], validation=validation, testing=testing, learn_weights=False)
		# calculate error on last step
		# print("val", mc.rP_breve[-1], target)
		mc.evolve_system(r0=[input_frame], u_tgt=[np.array([target])], validation=validation, testing=testing, learn_weights=False)

	return mc

def plot_examples(mc, path):

	for first_stim in [+1, -1]:
		for second_stim in [+1, -1]:

			# try one input
			input_ideal, target_ideal = generate_inputs_and_targets(10, seed=44, first_stim=first_stim, second_stim=second_stim)
			inputs = input_ideal[0]
			target = target_ideal[0]

			# re-init recording of params
			mc.init_record(rec_per_steps=1,
							rec_uI_breve=True, rec_uP_breve=True, 
							rec_rI_breve=True, rec_rP_breve=True,
							rec_uP=True, rec_uI=True, 
							rec_WPP=True, rec_BII=True, rec_WIP=True, rec_BPI=True, rec_vapi=True,
							rec_MSE=True
						   )
			# reset voltage
			mc.set_voltages(uP=mc.uP_init)

			for i in range(len(inputs)):
				mc.evolve_system(r0=[inputs[i]], learn_weights=False)

			fig = plt.figure()

			plt.plot(inputs, label="input")

			x = [v[0] for v in mc.uP_breve_time_series]
			x = np.array(x)
			plt.plot(x, label="$\\breve{u}_P^0$")

			x = [v[0] for v in mc.rP_breve_time_series]
			x = np.array(x)
			plt.plot(x, label="$r_P^0$", ls='--')

			x = [v[-1] for v in mc.uP_breve_time_series]
			x = np.array(x)
			plt.plot(x.ravel(), label="$\\breve{u}_P^1$")

			x = [v[-1] for v in mc.rP_breve_time_series]
			x = np.array(x)
			plt.plot(x.ravel(), label="$r_P^1$", ls='--')

			plt.scatter(len(inputs)+1, target, marker='x', c=f"C{i}", label="target")

			plt.ylim(-1.5,1.5)
			plt.grid()
			
			plt.title(str(first_stim) + ", " + str(second_stim))

			plt.legend(prop={'size': 11},loc='upper left')#, ncol=8, bbox_to_anchor=(1.2, .95))
			plt.xlabel('dt')

			plt.savefig(path + '/example_' + \
						str(first_stim).replace("-1","down").replace("1","up") + \
						str(second_stim).replace("-1","down").replace("1","up") + '.png', dpi=200)

def train_and_validate(mc):
	for epoch in range(epochs):
		logging.info(f"Seed {mc.seed}: Validating before training epoch {epoch}")
		mc = test_model(mc, input_val, target_val, validation=True, testing=False)
		logging.info(f"Seed {mc.seed}: Training epoch {epoch}")
		mc = train_model(mc, input_train, target_train)

	logging.info(f"Seed {mc.seed}: Testing model")
	mc = test_model(mc, input_test, target_test, validation=False, testing=True)

	return mc


if __name__ == '__main__':

	# create output folder
	if not os.path.exists(PATH):
		os.makedirs(PATH)

	# mc_ideal = init_mc(seed=123)

	# setting correct weights by hand
	# mc_ideal.WPP[0] = np.array([[1.0],[-1.0]])
	# mc_ideal.WPP[0] = np.array([[1.0],[-1.0]]) * 5
	# mc_ideal.WPP[1] = np.array([[1.0,1.0]]) * 5


	logging.info("Training networks")

	input_train, target_train = generate_inputs_and_targets(train_samples, seed=42)
	input_val, target_val = generate_inputs_and_targets(val_samples, seed=123456)
	input_test, target_test = generate_inputs_and_targets(test_samples, seed=44)

	MC_list = []
	for i in range(N_SEEDS):
		seed = i + np.random.randint(12345)
		MC_list.append(init_mc(seed=seed))
		MC_list[-1].seed = seed
		MC_list[-1].input_train = input_train
		MC_list[-1].target_train = target_train
		MC_list[-1].input_val = input_val
		MC_list[-1].target_val = target_val
		MC_list[-1].input_test = input_test
		MC_list[-1].target_test = target_test

	N_PROCESSES = len(MC_list) if N_MAX_PROCESSES > len(MC_list) else N_MAX_PROCESSES

	with mp.Pool(N_PROCESSES) as pool:
			MC_list = pool.map(train_and_validate, MC_list)
			pool.close()

	logging.info(f"Plotting validation MSE")

	plt.figure()

	data = np.array([mc.MSE_val_time_series for mc in MC_list])
	# rearrange into epochs
	data = data.reshape(N_SEEDS, epochs, -1)
	median = np.median(data, axis=-1)

	for x in median:
		plt.scatter(np.arange(len(x)), x)

	# median = np.median(data, axis=0)
	# q25, q75  = np.quantile(data, 0.25, axis=0), np.quantile(data, 0.75, axis=0)
	# plt.errorbar(np.arange(len(median)), median, yerr=[q25, q75], ls='none', marker='o')

	plt.xlabel("epoch")
	plt.ylabel("MSE loss")
	plt.title("Validation loss (median and quartile)")

	# x = np.arange(0, epochs*val_samples, val_samples)
	# for x in x:
	# 	plt.axvline(x=x, color='black', ls='dashed', alpha=0.5)
	plt.ylim(-.1,4.1)
	plt.tight_layout()
	plt.savefig(PATH + '/MSE_val.png', dpi=200)

	# save plots
	logging.info(f"Plotting voltages")
	plot_exp.plot(MC_list, path=PATH)

	# save model
	logging.info(f"Saving model to {PATH}")
	save_exp.save(MC_list, path=PATH)

	logging.info(f"Plotting examples")
	plot_examples(MC_list[0], path=PATH)






