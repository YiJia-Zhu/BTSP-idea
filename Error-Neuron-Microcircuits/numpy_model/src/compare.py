import numpy as np
from src.microcircuit import *
import src.init_MC as init_MC
import src.init_signals as init_signals
import src.run_exp as run_exp
import logging

def extract_offdiagonal(mat, layers):

	offdiag = [None for _ in layers[1:]]

	for idx, _ in enumerate(layers[:-1]):

		lower_row = np.sum(layers[:idx], dtype=int)
		upper_row = np.sum(layers[:idx+1])

		lower_col = np.sum(layers[:idx+1])
		upper_col = np.sum(layers[:idx+2])

		# generate hierarchical part of WPP
		offdiag[idx] = mat.T[lower_row:upper_row, lower_col:upper_col].T

	return offdiag



def calc_dWPP_ANN(algorithm, mc, W_list, B_list, activation_list, d_activation_list, r0, target):
	'''

		Calculates the weight updates in an ANN using BP
		for given input and weights.

		Input:  mc: microcircuit (needed for parameters)
				W_list: list of weights for ANN
				r0: input vector
				target: target vector
		Returns: list of updates dWPP in ANN

	'''

	# assert that mcs are using layered forward structure (no skip connection)
	if mc.mc_model != 'errormc' and mc.fw_connection_mode == 'skip':
		logging.warning("fw_connection_mode is not 'layered', cannot compare to ANN. Skipping")
		return MC_list

	if algorithm == 'BP':
		B_list = [W.T for W in W_list]
	elif algorithm == 'FA':
		# we need to prepend one element because there is one B less defined in the microcircuit
		B_list = [None] + B_list

	# forward pass
	# voltages correspond to vbashat
	voltages = [np.zeros_like(vec) for vec in mc.uP_breve]
	rates = [np.zeros_like(vec) for vec in mc.rP_breve]
	voltages[0] = mc.gbas / (mc.gbas + mc.gapi + mc.gl) * W_list[0] @ r0

	for i in range(len(W_list)-1):
		rates[i] = activation_list[i](voltages[i])
		# calculate vbashat
		voltages[i+1] = mc.gbas / (mc.gbas + mc.gapi + mc.gl) * W_list[i+1] @ rates[i]
	# correct output layer voltage for sacramento2018
	# in absence of target nudging
	if mc.mc_model in ['sacramento2018', 'dPC']:
		voltages[-1] = (mc.gbas + mc.gapi + mc.gl) / (mc.gbas + mc.gl) * voltages[-1]
	rates[-1] = activation_list[-1](voltages[-1])

	# backward pass:
	dWPP_BP_list = [np.zeros_like(W) for W in W_list]

	if mc.rate_target:
		# calculate output error on rate level at ouput
		error = np.diag(d_activation_list[-1](voltages[-1])) @ (target - rates[-1])
	else:
		# alternatively, define error on voltages
		error = (target - voltages[-1])

	# propagate error backwards
	for i in range(len(dWPP_BP_list)-1, 0, -1):
		dWPP_BP_list[i] = np.outer(error, rates[i-1])
		error = np.diag(d_activation_list[i-1](voltages[i-1])) @ B_list[i] @ error
	dWPP_BP_list[0] = np.outer(error, r0)

	# print("ANN voltages:")
	# print(voltages)
	# print("mc voltages:")
	# print(mc.uP_breve)
	# print("ANN error:")
	# print(error)
	# print("mc voltages I:")
	# print(mc.uI_breve)

	return dWPP_BP_list




def compare_updates(mc, model, params):
	"""
		Compares updates of mc (trained microcircuit model)
		given input/output pairs with an ANN with same weights

	"""

	# compare updates of mc object to dWPP of BP

	# warn if mcs are using layered forward structure (no skip connection),
	# because the ANN only has layered connectivity
	if mc.mc_model == 'errormc' and mc.fw_connection_mode == 'skip':
		logging.warning("fw_connection_mode is 'skip'. Comparing only hierchical weights to ANN.")
	elif mc.mc_model != 'errormc' and mc.fw_connection_mode == 'skip':
		logging.warning("fw_connection_mode is not 'layered', cannot compare to ANN. Skipping")
		return mc

	# check that records of WPP and backwards matrices exist
	if mc.mc_model in ['sacramento2018', 'ann', 'dPC']:
		Bmat_name = 'BPP_time_series'
	elif mc.mc_model == 'errormc':
		Bmat_name = 'BII_time_series'
	for attr in ['WPP_time_series', Bmat_name]:
		if not hasattr(mc, attr):
			logging.warning(f"No record of {attr} found, cannot compare to ANN. Skipping.")
			return mc

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped
	TPRE = int(mc.settling_time / mc.dt / mc.rec_per_steps)

	if model == "BP":

		# mc.gntgt = 1e-3
		# mc.gapi = 1e-3

		# mc.taueffP, mc.taueffP_notgt, mc.taueffI = mc.calc_taueff()

		if mc.model == 'PAL':
			# disable noise for PAL evaluation
			mc.noise_scale = [0 for _ in mc.noise_scale]
			mc.noise = [np.zeros_like(noise) for noise in mc.noise]

		# record dWPP for given sequence
		# for mc in MC_list:
		d_activation_list = mc.d_activation

		logging.info(f'Evaluating dWPP for microcircuit {mc}')
		mc.angle_BP_updates_time_series = []	# angle between the entries of the above two lists

		# forward weights
		Wmat_time_series = mc.WPP_time_series

		# lateral weights
		if mc.mc_model in ['sacramento2018', 'errormc', 'dPC']:
			WIP_time_series = mc.WIP_time_series
			BPI_time_series = mc.BPI_time_series
		elif mc.mc_model == 'ann':
			WIP_time_series = [mc.WIP for _ in mc.WPP_time_series]
			BPI_time_series = [mc.BPI for _ in mc.BPP_time_series]

		# different backwards matrices for each model
		if mc.mc_model in ['sacramento2018', 'ann', 'dPC']:
			Bmat_time_series = mc.BPP_time_series
		elif mc.mc_model == 'errormc':
			Bmat_time_series = mc.BII_time_series

		# we need to set the WPP learning rate to non-zero in order to calculate fictitious weight updates
		eta_fw_buffer = copy.deepcopy(mc.eta_fw)
		# mc.eta_fw = [1.0 for _ in eta_fw_buffer]
		mc.eta_fw = np.ones_like(eta_fw_buffer)

		for time, (WPP, WIP, Bmat, BPI) in enumerate(zip(Wmat_time_series[TPRE:], WIP_time_series[TPRE:], Bmat_time_series[TPRE:], BPI_time_series[TPRE:])):
			logging.info(f"Evaluating next set of weights {time+1}/{len(mc.WPP_time_series[TPRE:])}")
			# set etwork to recorded weights at this time step
			if mc.mc_model in ['sacramento2018', 'ann', 'dPC']:
				mc.set_weights(WPP=WPP, WIP=WIP, BPP=Bmat, BPI=BPI)
				# mc.set_self_predicting_state()
			elif mc.mc_model == 'errormc':
				mc.set_weights(WPP=WPP, WIP=WIP, BII=Bmat, BPI=BPI)

			# run with input, output pairs and record dWPP
			for i, (r0, target) in enumerate(zip(mc.input, mc.target)):
				mc.evolve_system(r0=r0, u_tgt=[target], learn_weights=False, learn_lat_weights=False, learn_bw_weights=False, record=False, compare_dWPP=True)

				# record dWPP after every presentation time
				if (i+1) % int(mc.Tpres / mc.dt) == 0:

					# need to extract hierarchical weights for skip fw connection mode
					if mc.mc_model == 'errormc' and mc.fw_connection_mode == 'skip':
						# logging.info("Extracting hierchical weights from full WPP matrix")
						WPP_ANN = extract_offdiagonal(mc.WPP[0], mc.layers)
						dWPP_MC = extract_offdiagonal(mc.dWPP[0], mc.layers)

					else:
						WPP_ANN = mc.WPP
						dWPP_MC = mc.dWPP

					# get weights in MC and ANN
					BP_dWPP = calc_dWPP_ANN(
								algorithm="BP", mc=mc, W_list=WPP_ANN, B_list=mc.BPP, activation_list=mc.activation,
								d_activation_list=d_activation_list, r0=r0, target=target
							)

					# calculate angle between weights
					mc.angle_BP_updates_time_series.append([
						deg(cos_sim(mc_dWPP_layer, BP_dWPP_layer)) for mc_dWPP_layer, BP_dWPP_layer in zip(dWPP_MC, BP_dWPP)
						])
					# print("mc.dWPP", mc.dWPP)
					# print("mc.dWBP", mc.dWPP_time_series_BP_ANN[-1])
					# print(mc.angle_BP_updates_time_series[-1])

		# reset learing rate
		mc.eta_fw = copy.deepcopy(eta_fw_buffer)

		return mc


def compare_updates_bw_only(MC_list, model):
	# compare updates of mc object to dWPP of BP, even if eta_fw = 0
	# this is done by reconstructing the dWPP of our model

	# assert that mcs are using layered forward structure (no skip connection)
	if MC_list[0].fw_connection_mode != 'layered':
		logging.warning("fw_connection_mode is not 'layered', cannot compare to ANN. Skipping.")
		return MC_list

	# check that records of WPP and backwards matrices exist
	if MC_list[0].mc_model in ['sacramento2018', 'ann', 'dPC']:
		Bmat_name = 'BPP_time_series'
	elif MC_list[0].mc_model == 'errormc':
		Bmat_name = 'BII_time_series'
	for attr in ['WPP_time_series', Bmat_name]:
		if not hasattr(MC_list[0], attr):
			logging.warning(f"No record of {attr} found, cannot compare to ANN. Skipping.")
			return MC_list

	# define the number of recorded time steps which belong to the pre-training
	# and therefore should be skipped
	TPRE = int(MC_list[0].settling_time / MC_list[0].dt / MC_list[0].rec_per_steps)

	if model == "BP":

		for mc in MC_list:
			d_activation_list = mc.d_activation

			mc.angle_dWPP_bw_only_time_series = []

			# for every time step
			for i in range(len(mc.dWPP_time_series[TPRE:])):
				angle_dWPP_bw_only_arr = []

				# for every layer
				for j in range(len(mc.layers)-1):
					# print("j", j)
					if j == 0:
						r_in = r0
					else:
						r_in = mc.rP_breve_time_series[TPRE:][i][j-1]

					# construct weight update for BP net

					dWPP_BP = np.outer(mc.error_time_series[i], r_in)

					# multiply phi' @ W.T from left
					for k in range(len(mc.layers)-2, j, -1):
						# print(j, k)
						dWPP_BP = np.diag(d_activation_list[k-1](mc.uP_breve_time_series[TPRE:][i][k-1])) @ mc.WPP_time_series[TPRE:][i][k].T @ dWPP_BP

					cos = cos_sim(mc.dWPP_time_series[TPRE:][i][j], -dWPP_BP)

					angle_dWPP_bw_only_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_dWPP_bw_only_time_series.append(angle_dWPP_bw_only_arr)

		return MC_list


def compare_weight_matrices(MC_list, model):

	# assert that mcs are using same forward and backward structure (skip or layered)
	if MC_list[0].fw_connection_mode != MC_list[0].bw_connection_mode:
		logging.warning(f"fw_connection_mode is '{MC_list[0].fw_connection_mode}', " \
			f"but bw_connection_mode is '{MC_list[0].bw_connection_mode}'. Skipping.")
		return MC_list

	# check that records of WPP and backwards matrices exist
	if MC_list[0].mc_model in ['sacramento2018', 'ann', 'dPC']:
		Bmat_name = 'BPP_time_series'
	elif MC_list[0].mc_model == 'errormc':
		Bmat_name = 'BII_time_series'
	for attr in ['WPP_time_series', Bmat_name]:
		if not hasattr(MC_list[0], attr):
			logging.warning(f"No record of {attr} found, cannot compare to ANN. Skipping.")
			return MC_list

	if model == "BP":

		for mc in MC_list:

			if mc.mc_model in ['sacramento2018', 'dPC']:

				mc.angle_WPPT_BPP_time_series = []
				# for every time step
				for i in range(len(mc.BPP_time_series)):
					angle_WPPT_BPP_arr = []
					# for every hidden layer
					for j in range(len(mc.layers)-2):

						if MC_list[0].bw_connection_mode == 'skip':
							WPP_T = np.eye(mc.layers[-1])
							# multiply phi' @ W.T from left
							for k in range(len(mc.layers)-2, j, -1):
								WPP_T = mc.WPP_time_series[i][k].T @ WPP_T
							BPP = mc.BPP_time_series[i][j]

						elif MC_list[0].bw_connection_mode == 'layered':
							WPP = mc.WPP_time_series[i][j+1]
							BPP = mc.BPP_time_series[i][j]
						
						if WPP.T.shape == BPP.shape:
							fwd_mat = WPP.T
						else:
							raise NotImplementedError("Shapes of B and W.T incompatible")
						
						cos = cos_sim(fwd_mat, BPP)
						angle_WPPT_BPP_arr.append(deg(cos))

					mc.angle_WPPT_BPP_time_series.append(angle_WPPT_BPP_arr)

			elif mc.mc_model == 'errormc':

				# for skip connections, we can only compare W and B
				# if they are the same shape (same number of rep and error units)
				if MC_list[0].layers[1:] != MC_list[0].error_layers[1:]:
					logging.warning(f"Unequal number of error and representation units, " \
						f"cannot compare BPP and WPP. Skipping.")
					return MC_list

				mc.angle_WPPT_BII_time_series = []
				# for every time step
				for i in range(len(mc.BII_time_series)):
					angle_WPPT_BII_arr = []
					# for every fwd/bck matrix pair
					for j in range(len(mc.WPP_time_series[0])):

						if MC_list[0].bw_connection_mode == 'skip':
							# omit input layer
							WPP = mc.WPP_time_series[i][j][mc.layers[0]:,mc.layers[0]:]
							BII = mc.BII_time_series[i][j]

							fwd_mat = WPP.T

						elif MC_list[0].bw_connection_mode == 'layered':
							if j >= len(mc.WPP_time_series[i])-1:
								break
							WPP = mc.WPP_time_series[i][j+1]
							BII = mc.BII_time_series[i][j]
						
							# multiply id with padding
							WIP_eye = np.eye(N=mc.WIP_time_series[i][j+1].shape[0], M=mc.WIP_time_series[i][j+1].shape[1])
							BPI_eye = np.eye(N=mc.BPI_time_series[i][j].shape[0], M=mc.BPI_time_series[i][j].shape[1])

							fwd_mat = (WIP_eye @ WPP @ BPI_eye).T

						cos = cos_sim(fwd_mat, BII)
						angle_WPPT_BII_arr.append(deg(cos))

					mc.angle_WPPT_BII_time_series.append(angle_WPPT_BII_arr)

		return MC_list


def compare_jacobians(MC_list, model):

	# check that records of WPP and backwards matrices exist
	if MC_list[0].mc_model in ['sacramento2018', 'ann', 'dPC']:
		Bmat_name = 'BPP_time_series'
	elif MC_list[0].mc_model == 'errormc':
		Bmat_name = 'BII_time_series'
	for attr in ['WPP_time_series', Bmat_name]:
		if not hasattr(MC_list[0], attr):
			logging.warning(f"No record of {attr} found, cannot compare to ANN. Skipping.")
			return MC_list

	if model == "BP":

		for mc in MC_list:
			if mc.mc_model != 'sacramento2018':
				raise NotImplementedError("compare_jacobians only implemented for sacramento2018")
			d_activation_list = mc.d_activation

			mc.angle_jacobians_BP_time_series = []

			# for every time step
			for i in range(len(mc.BPP_time_series)):
				angle_jacobians_BP_arr = []

				# for every hidden layer
				for j in range(len(mc.layers)-2):

					if MC_list[0].bw_connection_mode == 'skip':
						# construct Jacobian J_f^T
						J_f_T = np.eye(mc.layers[-1])
						# multiply phi' @ W.T from left
						for k in range(len(mc.layers)-2, j, -1):
							J_f_T = np.diag(d_activation_list[k-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

						# construct Jacobian J_g
						J_g = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][-1]))

					elif MC_list[0].bw_connection_mode == 'layered':
						J_f_T = np.diag(d_activation_list[j](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j])) @ mc.WPP_time_series[i][j+1].T
						J_g   = mc.BPP_time_series[i][j] @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))

					

					cos = cos_sim(J_g, J_f_T)

					angle_jacobians_BP_arr.append(deg(cos))
					# print(angle_dWPP_arr[-1])

				mc.angle_jacobians_BP_time_series.append(angle_jacobians_BP_arr)

		return MC_list


def compare_B_RHS(MC_list, model):

	# check that records of WPP and backwards matrices exist
	if MC_list[0].mc_model in ['sacramento2018', 'ann', 'dPC']:
		Bmat_name = 'BPP_time_series'
	elif MC_list[0].mc_model == 'errormc':
		Bmat_name = 'BII_time_series'
	for attr in ['WPP_time_series', Bmat_name]:
		if not hasattr(MC_list[0], attr):
			logging.warning(f"No record of {attr} found, cannot compare to ANN. Skipping.")
			return MC_list

	# similar to compare_jacobians, but involves the 'correct' result of what BPP/BII should converge to
	# in equations, that should be BPP ~ phi' WPP.T phi'

	# assert that mcs are using layered forward structure (no skip connection)
	if MC_list[0].fw_connection_mode != 'layered':
		logging.warning("fw_connection_mode is not 'layered', cannot compare to PAL RHS. Skipping.")
		return MC_list
	# assert that mcs are using layered backward structure (no skip connection)
	if MC_list[0].bw_connection_mode != 'layered':
		logging.warning("bw_connection_mode is not 'layered', cannot compare to PAL RHS. Skipping.")
		return MC_list

	if model == "BP":

		for mc in MC_list:
			d_activation_list = mc.d_activation

			if mc.mc_model == 'sacramento2018':

				mc.angle_BPP_RHS_time_series = []

				# for every time step
				for i in range(len(mc.BPP_time_series)):
					angle_BPP_RHS_arr = []

					# for every hidden layer
					for j in range(len(mc.layers)-2):

						if MC_list[0].bw_connection_mode == 'skip':
							# construct Jacobian J_f^T
							J_f_T = np.eye(mc.layers[-1])
							# multiply phi' @ W.T from left
							for k in range(len(mc.layers)-2, j, -1):
								RHS = np.diag(d_activation_list[k-1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][k-1])) @ mc.WPP_time_series[i][k].T @ J_f_T

							RHS = RHS @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))
							# construct Jacobian J_g
							BPP = mc.BPP_time_series[i][j]

						elif MC_list[0].bw_connection_mode == 'layered':
							RHS = np.diag(d_activation_list[j](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j])) @ mc.WPP_time_series[i][j+1].T
							RHS = RHS @ np.diag(d_activation_list[j+1](mc.gbas / (mc.gbas + mc.gapi + mc.gl) * mc.vbas_time_series[i][j+1]))
							BPP = mc.BPP_time_series[i][j]
						

						cos = cos_sim(BPP, RHS)

						angle_BPP_RHS_arr.append(deg(cos))
						# print(angle_dWPP_arr[-1])

					mc.angle_BPP_RHS_time_series.append(angle_BPP_RHS_arr)


			elif mc.mc_model == 'errormc':

				mc.angle_BII_RHS_time_series = []
				# for every time step
				for i in range(len(mc.BII_time_series)):
					angle_BII_RHS_arr = []
					# for every hidden layer
					for j in range(len(mc.layers)-2):

						if MC_list[0].bw_connection_mode == 'skip':
							WPP = np.eye(mc.layers[-1])
							# multiply phi' @ W.T from left
							for k in range(len(mc.layers)-2, j, -1):
								WPP_T = mc.WPP_time_series[i][k].T @ WPP.T
							BII = mc.BII_time_series[i][j]

						elif MC_list[0].bw_connection_mode == 'layered':
							WPP = mc.WPP_time_series[i][j+1]
							BII = mc.BII_time_series[i][j]
						
						# lateral weights
						WIP = mc.WIP_time_series[i][j+1]
						BPI = mc.BPI_time_series[i][j]

						# phi_prime
						pp_r_below = np.diag(d_activation_list[j](mc.uP_breve_time_series[i][j]))
						pp_r_above = np.diag(d_activation_list[j+1](mc.uP_breve_time_series[i][j+1]))
						pp_e_below = np.diag(d_activation_list[j](mc.uI_breve_time_series[i][j]))
						pp_e_above = np.diag(d_activation_list[j+1](mc.uI_breve_time_series[i][j+1]))

						# construct full ideal PAL result
						fwd_mat = (pp_e_above @ WIP @ pp_r_above @ WPP @ pp_r_below @ BPI @ pp_e_below).T
						cos = cos_sim(fwd_mat, BII)
						angle_BII_RHS_arr.append(deg(cos))

					mc.angle_BII_RHS_time_series.append(angle_BII_RHS_arr)

		return MC_list
