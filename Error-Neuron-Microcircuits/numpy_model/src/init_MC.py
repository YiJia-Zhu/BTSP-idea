import numpy as np
from src.microcircuit import *
import logging
from functools import partial, update_wrapper
retry_queue = []

# wrapper for passing functions to partial, keeping __name__ attribute
# needed for noise on activation function
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# define mappings from str input to functions
function_mappings = {
	'linear': linear,
	'relu': relu,
	'soft_relu': soft_relu,
	'logistic': logistic,
	'tanh': tanh,
	# 'tanh_offset': tanh_offset,
	'hard_sigmoid': hard_sigmoid
}

logging.basicConfig(format='Init model -- %(levelname)s: %(message)s',
					level=logging.INFO)

# defines the voltage inits
def init_voltages(layers, error_layers, params, seed):

	np.random.seed(seed)

	uP_init = []
	for i in range(1, len(layers)):
		uP_init.append(np.random.normal(0, 1, size=layers[i]))

	uI_init = []
	for i in range(0, len(error_layers)):
		uI_init.append(np.random.normal(0, 1, size=error_layers[i]))

	return uP_init, uI_init
		

def init_weights(layers, error_layers, params, seed, mc_model, teacher=False):

	np.random.seed(seed)

	if teacher:
		WPP_range = params['init_teacher_WPP_range']
		WIP_range = params['init_teacher_WIP_range']
		if mc_model in ['sacramento2018', 'ann', 'dPC']:
			BPP_range = params['init_teacher_BPP_range']
		BPI_range = params['init_teacher_BPI_range']
		if mc_model == 'errormc':
			BII_range = params['init_teacher_BII_range']
	else:
		WPP_range = params['init_WPP_range']
		WIP_range = params['init_WIP_range']
		if mc_model in ['sacramento2018', 'ann', 'dPC']:
			BPP_range = params['init_BPP_range']
		BPI_range = params['init_BPI_range']
		if mc_model == 'errormc':
			BII_range = params['init_BII_range']

	if mc_model == 'errormc' and "init_realistic_connectivity" in params and params["init_realistic_connectivity"]:
		if params["fw_connection_mode"] != 'skip':
			raise ValueError("init_realistic_connectivity only works with fw_connection_mode='skip'")
		WPP_init, _ = init_viz_cort_conn(layers=params['layers'], error_layers=params['error_layers'],
										 init_WPP_range=params["init_WPP_range"],
										 init_BII_range=params["init_BII_range"])

	elif params["fw_connection_mode"] == 'layered':
		# forward pp weights: connects all layers k, k+1
		WPP_init = []
		for i in range(len(layers)-1):
			WPP_init.append(np.random.uniform(WPP_range[0], WPP_range[1], size=(layers[i+1], layers[i])))
	elif params["fw_connection_mode"] == 'skip':
		WPP_init = init_nonhierarchical_WPP(WPP_range, params['layers'], params['WPP_skip_connection_range'])

	# p to inter: connects k to k
	WIP_init = []
	for i in range(len(error_layers)):
		WIP_init.append(np.random.uniform(WIP_range[0], WIP_range[1], size=(error_layers[i], layers[i+1])))
		if mc_model == 'errormc' and params['init_WIP_identity']:
			WIP_init[-1] += np.eye(N=error_layers[i], M=layers[i+1])

	if mc_model in ['sacramento2018', 'ann', 'dPC']:
		if params["model_type"] != 'BP':
			# backwards p to p: connects k+1 to k
			BPP_init = []
			for i in range(1, len(layers)-1):
				BPP_init.append(np.random.uniform(BPP_range[0], BPP_range[1], size=(layers[i], layers[i+1])))
		else:
			BPP_init = [W.T for W in WPP_init]
	else:
		BPP_init = None

	# backwards inter to p: connects k to k
	BPI_init = []
	for i in range(len(error_layers)):
		BPI_init.append(np.random.uniform(BPI_range[0], BPI_range[1], size=(layers[i+1], error_layers[i])))
		if mc_model == 'errormc' and params['init_BPI_identity']:
			BPI_init[-1] += np.eye(N=layers[i+1], M=error_layers[i])

	if mc_model == 'errormc' and "init_realistic_connectivity" in params and params["init_realistic_connectivity"]:
		if params["bw_connection_mode"] != 'skip':
			raise ValueError("init_realistic_connectivity only works with bw_connection_mode='skip'")
		_, BII_init = init_viz_cort_conn(layers=params['layers'], error_layers=params['error_layers'],
										 init_WPP_range=params["init_WPP_range"],
										 init_BII_range=params["init_BII_range"])

	elif mc_model == 'errormc' and params["bw_connection_mode"] == 'layered':
		BII_init = []
		for i in range(1, len(layers)-1):
			if teacher and "teacher_error_layers" in params:
				BII_init.append(np.random.uniform(BII_range[0], BII_range[1],
					size=(params['teacher_error_layers'][i], params['teacher_error_layers'][i+1]))
				)
			else:
				BII_init.append(np.random.uniform(BII_range[0], BII_range[1],
					size=(layers[i], layers[i+1]))
				)
				
	elif mc_model == 'errormc' and params["bw_connection_mode"] == 'skip':
		if teacher and "teacher_error_layers" in params:
			BII_init = init_nonhierarchical_BII(BII_range,  params['teacher_error_layers'],
															params['BII_skip_connection_range'])
		else:
			BII_init = init_nonhierarchical_BII(BII_range,  params['error_layers'], 
															params['BII_skip_connection_range'])
	else:
		BII_init = None


	return WPP_init, WIP_init, BPP_init, BPI_init, BII_init


def get_slopes_and_offsets(params, disable_noise=False, layers=None, activation_list=None):
	if "activation_slope" in params:
		assert len(params["activation_slope"]) == len(activation_list), \
			f"'activation_slope' is not same length as activations ({len(params['activation_slope'])} vs. {len(activation_list)})"
	if "activation_offset" in params:
		assert len(params["activation_offset"]) == len(activation_list), \
			f"'activation_offset' is not same length as activations ({len(params['activation_offset'])} vs. {len(activation_list)})"

	# different slopes and offsets, including noise
	if "activation_slope" in params:
		slopes = params["activation_slope"]
	else:
		slopes = np.array([1.0 for _ in activation_list])
	if "activation_slope_std" in params and not disable_noise:
		# every neuron can have individual slope
		slopes = [np.repeat(slope, layers[1:][i]) for i, slope in enumerate(slopes)]
		slope_stds = [np.random.normal(0.0, params["activation_slope_std"], size=layer) for layer in layers[1:]]
		slopes = [slope + slope_std for slope, slope_std in zip(slopes, slope_stds)]

	if "activation_offset" in params:
		offsets = params["activation_offset"]
	else:
		offsets = np.array([0.0 for _ in activation_list])
	if "activation_offset_std" in params and not disable_noise:
		# every neuron can have individual offset
		offsets = [np.repeat(offset, layers[1:][i]) for i, offset in enumerate(offsets)]
		offset_stds = [np.random.normal(0.0, params["activation_offset_std"], size=layer) for layer in layers[1:]]
		offsets = [offset + offset_std for offset, offset_std in zip(offsets, offset_stds)]

	return slopes, offsets


# defines the microcircuit models
def init_MC(params, seeds, teacher=False):

	MC_list = []

	if teacher:
		# init a single teacher based on the first seed in seed list
		np.random.seed(seeds[0])
		seeds = [np.random.randint(100000, 999999)]
		if "teacher_mc_model" in params:
			mc_model = params["teacher_mc_model"]
		else:
			mc_model = params["mc_model"]

	else:
		mc_model = params["mc_model"]

	if mc_model in ["sacramento2018", "ann", "dPC"] \
		and 'fw_connection_mode' in params and params['fw_connection_mode'] != 'layered':
			logging.warning(f"fw_connection_mode is {params['fw_connection_mode']}," \
				+ f" but only 'layered' is currently implemented for {mc_model}. Falling back to 'layered'.")
			params['fw_connection_mode'] = 'layered'
	if mc_model in ["sacramento2018", "ann", "dPC"] \
		and 'bw_connection_mode' in params and params['bw_connection_mode'] != 'layered':
			logging.warning(f"bw_connection_mode is {params['bw_connection_mode']}," \
				+ f" but only 'layered' is currently implemented for {mc_model}. Falling back to 'layered'.")
			params['bw_connection_mode'] = 'layered'
	# if mc_model == "errormc" and params["model_type"] == "BP" \
	# 	and params["layers"][1:] != params["error_layers"][1:]:
	# 	logging.warning(f"Unequal number of representation and error units is incompatible with BP. " \
	# 		"Falling back to error_layers = layers.")
	# 	params["error_layers"] = params["layers"]

	for seed in seeds:
		if teacher == True and params["input_signal"] == "cartpole":
			# hardcode a special teacher for cartpole task
			layers = [4, 4, 1]
			if mc_model in ['sacramento2018', "dPC"]:
				error_layers_init = [1]
			else:
				error_layers_init = [4, 1]
				error_layers = params['error_layers']

			uP_init, uI_init = init_voltages(layers, error_layers_init, params, seed)

			WPP_init = []
			w_rescaling_factor = (params["gbas"] / (params["gbas"] + params["gapi"] + params["gl"]))**-(len(layers) - 1)
			target_rescale_factor = params["target_rescale_factor"] # empirical factor to get the right output

			_, WIP_init, BPP_init, BPI_init, BII_init = init_weights(layers,
																	 error_layers_init,
																	 params,
																	 seed,
																	 mc_model,
																	 teacher=True)
			WPP_init.append(np.eye(layers[0])*w_rescaling_factor)
			WPP_init.append(np.array([1., 4.2, 62.8, 16.3]).reshape(layers[2], layers[1]) / target_rescale_factor)

			activation_list = [linear, linear]
			error_activation_list = [linear, linear]

		else:
			if teacher:
				if "teacher_layers" in params:
					layers = params["teacher_layers"]
				else:
					layers = params["layers"]
				if "teacher_error_layers" in params and mc_model == 'errormc':
					error_layers = params["teacher_error_layers"]
				else:
					error_layers = params["error_layers"]
			else:
				layers = params['layers']
				error_layers = params['error_layers']

			if mc_model in ['sacramento2018', "dPC"]:
				error_layers_init = layers[2:]
			elif mc_model == 'ann':
				error_layers_init = layers[1:]
			else:
				error_layers_init = error_layers[1:]

			uP_init, uI_init = init_voltages(layers, error_layers_init, params, seed)

			WPP_init, WIP_init, BPP_init, BPI_init, BII_init = init_weights(layers,
																			error_layers_init,
																			params,
																			seed,
																			mc_model,
																			teacher=teacher)


			# if a list of activations has been passed, use it
			if isinstance(params["activation"], list):
				activation_list = [function_mappings[activation] for activation in params["activation"]]
			# else, set same activation for all layers
			else:
				activation_list = [function_mappings[params["activation"]] for _ in params["layers"][1:]]

			# for dPC, we may want to set activation to linear to be faithful to Mikulasch et al.
			if mc_model == 'dPC' and params["activation"] != 'linear' and not teacher:
				logging.warning("Setting activation to 'linear' for dPC")
				activation_list = [linear for _ in params["layers"][1:]]

			if activation_list[-1] != linear:
				logging.info('Output layer activation is not linear -- ' \
					'watch out when comparing sacramento2018 and errormc!')

			slopes, offsets = get_slopes_and_offsets(params, disable_noise=teacher, layers=layers, activation_list=activation_list)
			activation_list = [wrapped_partial(func, slope=slope, offset=offset) for (func, slope, offset) in zip(activation_list, slopes, offsets)]

			if mc_model == "errormc":
				if isinstance(params["error_activation"], list):
					error_activation_list = [function_mappings[error_activation] for error_activation in params["error_activation"]]
				else:
					error_activation_list = [function_mappings[params["error_activation"]] for _ in params["layers"][1:]]

				# no noise for error units
				slopes, offsets = get_slopes_and_offsets(params, disable_noise=True, layers=error_layers, activation_list=error_activation_list)
				error_activation_list = [wrapped_partial(func, 
											slope=slope, offset=offset
											) for (func, slope, offset) in zip(error_activation_list, slopes, offsets)]

				# reshape learning rate for skip connections
				if params["fw_connection_mode"] == 'skip' and np.array(params["eta_fw"]).shape != (len(layers)-1, len(layers)-1):
					params["eta_fw"] = convert_learning_rate_for_skip(params["eta_fw"], params["eta_fw_conversion"])

		if "WT_noise" in params and not teacher:
				WT_noise = params["WT_noise"]
		else:
			WT_noise = 0.0

		if mc_model == 'sacramento2018' and params["model_type"] in ["BP", "FA"]:
			if error_layers in params["error_layers"] and params["error_layers"] !=  params["layers"]:
				logging.warning(f"error_layers not implemented for Sacramento2018, using same number as layers")

			MC_list.append(
				base_model(
					seed=seed,
					bw_connection_mode='layered',
					dWPP_use_activation=params["dWPP_use_activation"],
					dt=params["dt"],
					Tpres=params["Tpres"],
					model=params["model_type"],
					activation=activation_list,
					layers=layers,

					uP_init=uP_init,
					uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"],

					WT_noise = WT_noise
					)
				)

		elif mc_model == 'sacramento2018' and params["model_type"] in ["PAL"]:
			noise_deg = params["noise_deg"] if "noise_deg" in params else None
			taueps = params["taueps"] if "taueps" in params else None
			tauxi = params["tauxi"] if "tauxi" in params else None
			varphi_regularizer = params["varphi_regularizer"] if "varphi_regularizer" in params else False

			MC_list.append(
				noise_model(
					seed=seed,
					bw_connection_mode='layered',
					dWPP_use_activation=params["dWPP_use_activation"],
					dt=params["dt"],
					dtxi=params["dtxi"],
					tauHP=params["tauHP"],
					tauLO=params["tauLO"],
					Tpres=params["Tpres"],
					noise_scale=params["noise_scale"],
					alpha=params["alpha"],
					inter_low_pass=params["inter_low_pass"],
					pyr_hi_pass=params["pyr_hi_pass"],
					dWPP_low_pass=params["dWPP_low_pass"],
					dWPP_r_low_pass=params["dWPP_r_low_pass"],
					dWPP_post_low_pass=params["dWPP_post_low_pass"],
					gate_regularizer=params["gate_regularizer"],

					noise_type=params["noise_type"],
					noise_mode=params["noise_mode"],
					model=params["model_type"],
					activation=activation_list,
					layers=layers,

					uP_init=uP_init,
					uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"],

					noise_deg=noise_deg,
					taueps=taueps,
					tauxi=tauxi,
					varphi_regularizer=varphi_regularizer

					)
				)

		elif mc_model == 'errormc':

			if params['model_type'] not in ["BP", "FA"]:
				raise ValueError('Only BP and FA defined for errormc.')

			MC_list.append(
				errormc_model(
					seed=seed,
					fw_connection_mode=params["fw_connection_mode"],
					bw_connection_mode=params["bw_connection_mode"],
					dWPP_use_activation=params["dWPP_use_activation"],
					varphi_transfer=params['varphi_transfer'],
					dt=params["dt"],
					dtxi=params["dtxi"],
					tauHP=params["tauHP"],
					tauLO=params["tauLO"],
					Tpres=params["Tpres"],
					noise_scale=params["noise_scale"],
					alpha=params["alpha"],
					inter_low_pass=params["inter_low_pass"],
					pyr_hi_pass=params["pyr_hi_pass"],
					dWPP_low_pass=params["dWPP_low_pass"],
					dWPP_r_low_pass=params["dWPP_r_low_pass"],
					dWPP_post_low_pass=params["dWPP_post_low_pass"],
					gate_regularizer=params["gate_regularizer"],

					noise_type=params["noise_type"],
					noise_mode=params["noise_mode"],

					model=params["model_type"],
					activation=activation_list,
					error_activation=error_activation_list,
					layers=layers,
					error_layers=error_layers,

					uP_init=uP_init,
					uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BII_init=BII_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"],

					tauxi=params["tauxi"],
					WT_noise=WT_noise
					)
				)

		elif mc_model == 'ann' and params["model_type"] in ["BP", "FA"]:
			if error_layers in params["error_layers"] and params["error_layers"] !=  params["layers"]:
				logging.warning(f"error_layers not implemented for ANN, using same number as layers")
			MC_list.append(
				ann_model(
					seed=seed,
					dt=params["dt"],
					Tpres=params["Tpres"],
					model=params["model_type"],
					activation=activation_list,
					layers=layers,

					uP_init=uP_init,
					uI_init=uI_init,

					WPP_init=WPP_init,
					BPP_init=BPP_init,

					gl=params["gl"],
					gbas=params["gbas"],
					eta_fw=params["eta_fw"]
					)
				)
		elif mc_model == 'dPC' and params["model_type"] in ["BP", "FA"]:
			if error_layers in params["error_layers"] and params["error_layers"] !=  params["layers"]:
				logging.warning(f"error_layers not implemented for dPC, using same number as layers")

			MC_list.append(
				dPC_model(
					seed=seed,
					bw_connection_mode='layered',
					dWPP_use_activation=params["dWPP_use_activation"],
					dt=params["dt"],
					Tpres=params["Tpres"],
					model=params["model_type"],
					activation=activation_list,
					layers=layers,

					uP_init=uP_init,
					uI_init=uI_init,

					WPP_init=WPP_init,
					WIP_init=WIP_init,
					BPP_init=BPP_init,
					BPI_init=BPI_init,

					gl=params["gl"],
					gden=params["gden"],
					gbas=params["gbas"],
					gapi=params["gapi"],
					gnI=params["gnI"],
					gntgt=params["gntgt"],
					eta_fw=params["eta_fw"],
					eta_bw=params["eta_bw"],
					eta_PI=params["eta_PI"],
					eta_IP=params["eta_IP"],

					WT_noise = WT_noise
					)
				)


		else:
			raise NotImplementedError(f"mc_model {mc_model} and model_type {model_type} not implemented.")
		# save seed of mc and other params
		MC_list[-1].seed = seed
		MC_list[-1].mc_model = mc_model
		MC_list[-1].input_signal = params["input_signal"]
		MC_list[-1].dataset_size = params["dataset_size"]
		MC_list[-1].settling_time = params["settling_time"]
		MC_list[-1].copy_teacher_weights = params["copy_teacher_weights"]
		MC_list[-1].copy_teacher_voltages = params["copy_teacher_voltages"]
		if teacher:
			MC_list[-1].epochs = 1
		else:
			MC_list[-1].epochs = params["epochs"]
		MC_list[-1].init_in_SPS = params["init_in_SPS"]
		# data recording options
		MC_list[-1].rec_per_steps=params["rec_per_steps"]
		if teacher:
			MC_list[-1].rec_MSE=False
			MC_list[-1].rec_error=False
		else:
			MC_list[-1].rec_MSE=params["rec_MSE"]
			MC_list[-1].rec_error=params["rec_error"]
		MC_list[-1].rec_input=params["rec_input"]
		MC_list[-1].rec_target=params["rec_target"]

		if "rate_target" in params:
			MC_list[-1].rate_target = params["rate_target"]
		if "rec_rate_MSE" in params:
			MC_list[-1].rec_rate_MSE = params["rec_rate_MSE"]

		MC_list[-1].rec_WPP=params["rec_WPP"]
		MC_list[-1].rec_dWPP=params["rec_dWPP"]

		MC_list[-1].rec_WIP=params["rec_WIP"]
		MC_list[-1].rec_dWIP=params["rec_dWIP"]

		if mc_model != 'errormc':
			MC_list[-1].rec_BPP=params["rec_BPP"]
			MC_list[-1].rec_dBPP=params["rec_dBPP"]
			MC_list[-1].rec_BII=False
			MC_list[-1].rec_dBII=False
		else:
			MC_list[-1].rec_BPP=False
			MC_list[-1].rec_dBPP=False
			MC_list[-1].rec_BII=params["rec_BII"]
			MC_list[-1].rec_dBII=params["rec_dBII"]

		MC_list[-1].rec_BPI=params["rec_BPI"]
		MC_list[-1].rec_dBPI=params["rec_dBPI"]

		MC_list[-1].rec_uP=params["rec_uP"]
		MC_list[-1].rec_uP_breve=params["rec_uP_breve"]
		MC_list[-1].rec_rP_breve=params["rec_rP_breve"]
		MC_list[-1].rec_uI=params["rec_uI"]
		MC_list[-1].rec_uI_breve=params["rec_uI_breve"]
		MC_list[-1].rec_rI_breve=params["rec_rI_breve"]
		MC_list[-1].rec_vapi=params["rec_vapi"]
		MC_list[-1].rec_vbas=params["rec_vbas"]
		if "rec_lat_mismatch" in params:
			MC_list[-1].rec_lat_mismatch=params["rec_lat_mismatch"]
		# some variables only exist in PAL
		if params["model_type"] in ["PAL"]:
			MC_list[-1].rec_rP_breve_HI=params["rec_rP_breve_HI"]
			MC_list[-1].rec_vapi_noise=params["rec_vapi_noise"]
			MC_list[-1].rec_noise=params["rec_noise"]
			MC_list[-1].rec_epsilon=params["rec_epsilon"]
			MC_list[-1].rec_epsilon_LO=params["rec_epsilon_LO"]
		else:
			MC_list[-1].rec_rP_breve_HI=False
			MC_list[-1].rec_vapi_noise=False
			MC_list[-1].rec_noise=False
			MC_list[-1].rec_epsilon=False
			MC_list[-1].rec_epsilon_LO=False
		if teacher:
			MC_list[-1].rec_uP_breve=True
		
	if mc_model in ['sacramento2018', 'dPC']:
		for mc in MC_list:
			if mc.init_in_SPS:
				mc.set_self_predicting_state()
	if params["input_signal"] == "cartpole":
		for mc in MC_list:
			mc.target_rescale_factor = params["target_rescale_factor"]

	# attach connectivity variables
	if mc_model in ["sacramento2018", "ann", "dPC"]:
		for mc in MC_list:
			mc.fw_connection_mode = params['fw_connection_mode']
			mc.bw_connection_mode = params['bw_connection_mode']

	return MC_list
