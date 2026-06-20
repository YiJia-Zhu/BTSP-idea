# Loads a paramater file and executes microcircuit.py

# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt

from src.microcircuit import *
import src.init_MC as init_MC
import src.init_signals as init_signals
import src.run_exp as run_exp
import src.plot_exp as plot_exp
import src.save_exp as save_exp
import src.compare as compare

import sys
import os
import argparse
import logging
import json
import multiprocessing as mp
import functools
import time

N_MAX_PROCESSES = 12 # defined by compute setup


logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
					level=logging.INFO)

def parse_experiment_arguments():
	"""
		Parse the arguments for the test and train experiments
	"""

	parser = argparse.ArgumentParser(description='Train a model on a teacher-student task ' \
		'dataset.')
	parser.add_argument('--params', type=str,
						help='Path to the parameter .py-file.')
	parser.add_argument('--task', type=str,
						help='Choose from <fw_only> or <bw_only> or <fw_bw>.',
						default='fw_only')
	parser.add_argument('--load', type=str,
						help='Load a previously saved model from a .pkl file.',
						default=None)
	parser.add_argument('--cont', type=str,
						help='Initialize using weights another model lodaded from a .pkl file.',
						default=None)
	parser.add_argument('--compare', type=str,
						help='Compare weight updates to backprop (BP).',
						default=None)
	args = parser.parse_args()

	return args


def main(params, task='fw_bw', seeds=[667], load=None, cont=None, compare_model=None):



	t_start = time.time()

	if load is None:

		logging.info(f'Initializing student MCs')
		MC_list = init_MC.init_MC(params, seeds)
		logging.info(f'List of initialized student MCs: {MC_list}')

		if params['input_signal'] == 'genMNIST':
			logging.info(f"Input signal is {params['input_signal']}, disabling teacher.")
			task = 'fw_only_no_teacher'


		if task in ['fw_only', 'fw_bw']:
			logging.info(f'Initializing teacher MC')
			MC_teacher = init_MC.init_MC(params, seeds, teacher=True)
			if MC_teacher[0].mc_model in ['sacramento2018', 'dPC']:
				MC_teacher[0].set_self_predicting_state()
			logging.info(f'Teacher initialized with seed {MC_teacher[0].seed}')

			MC_teacher = init_signals.init_r0(MC_list=MC_teacher, form=params["input_signal"], seed=MC_teacher[0].seed)

			for mc in MC_list:
				if mc.copy_teacher_weights:
						logging.info(f'Copying teacher weights')
						mc.set_weights(model=MC_teacher[0])
				if mc.copy_teacher_voltages:
					logging.info(f'Copying teacher voltages')
					mc.set_voltages(model=MC_teacher[0])

			# make sure input and output layers agree between teacher and students
			for mc in MC_list:
				assert mc.layers[0] == MC_teacher[0].layers[0]
				assert mc.layers[-1] == MC_teacher[0].layers[-1]

		# load weights from file
		if cont is not None:
			logging.info(f'Loading results from {cont}')
			MC_list_tmp = save_exp.load(cont)

			if task in ['fw_bw', 'fw_only']:
				MC_teacher_tmp, MC_list_tmp = [MC_list_tmp[0]], MC_list_tmp[1:]
				MC_teacher[0].set_weights(model=MC_teacher_tmp)

			for mc, mc_tmp in zip(MC_list, MC_list_tmp):
				mc.set_weights(model=mc_tmp)


		# init input signal
		# if no teacher is present, use seed of first microcircuit
		if task in ['bw_only', 'fw_only_no_teacher']:
			MC_list = init_signals.init_r0(MC_list=MC_list, form=params["input_signal"], seed=MC_list[0].seed)
		else:
			MC_list = init_signals.init_r0(MC_list=MC_list, form=params["input_signal"], seed=MC_teacher[0].seed)


		logging.info(f'Model: {params["model_type"]}')
		logging.info(f'Task: {task}')

		if 'taur' in params:
			import types
			taur = params['taur']
			logging.info(f'Prospecivity will be modulated with taur: {taur}')
			# overwrite prospective_voltage function of model
			def _modulated_prospective_voltage(self, uvec, uvec_old, tau, dt=None):
				# returns an approximation of the lookahead of voltage vector u at current time
				if dt == None:
						dt = self.dt
				return uvec_old + taur * tau * (uvec - uvec_old) / dt
			for mc in MC_list:
				mc.prospective_voltage = types.MethodType(_modulated_prospective_voltage, mc)


		if task in ['fw_only', 'fw_bw']:
			logging.info(f'Running teacher to obtain target signal')

			rate_target = MC_list[0].rate_target
			if MC_teacher[0].rate_target != MC_list[0].rate_target:
				logging.info(f"Teacher and student are not set to same target (voltage vs. rate). " \
					" Using student settings: rate_target = {rate_target}")

			MC_teacher = [run_exp.run(MC_teacher[0], learn_weights=False, learn_lat_weights=False, learn_bw_weights=False, teacher=True, rate_target=rate_target)]

			if rate_target:
				logging.info(f'Setting rP_breve in output layer of teacher as target')
			else:
				logging.info(f'Setting uP_breve in output layer of teacher as target')

			for mc in MC_list:
				# use target time series after settling time
				mc.target = MC_teacher[0].target
				mc.target_testing = MC_teacher[0].target_testing
				mc.target_validation = MC_teacher[0].target_validation

		logging.debug(f"Current state of networks:")
		for mc in MC_list:
			logging.debug(f"Voltages: {mc.uP}, {mc.uI}")
			logging.debug(f"Weights: {mc.WPP}, {mc.WIP}, {mc.BPP}, {mc.BPI}")
			logging.debug(f"Input: {mc.input}")

		N_PROCESSES = len(MC_list) if N_MAX_PROCESSES > len(MC_list) else N_MAX_PROCESSES

		logging.info(f'Setting up and running {N_PROCESSES} processes')

		# init a partial function to pass additional arguments
		if task == 'bw_only':
			partial_run = functools.partial(run_exp.run, learn_weights=False, learn_lat_weights=True, learn_bw_weights=True, teacher=False)
		elif task == 'fw_bw':
			partial_run = functools.partial(run_exp.run, learn_weights=True, learn_lat_weights=True, learn_bw_weights=True, teacher=False)
		elif task in ['fw_only', 'fw_only_no_teacher']:
			partial_run = functools.partial(run_exp.run, learn_weights=True, learn_lat_weights=True, learn_bw_weights=False, teacher=False)
		with mp.Pool(N_PROCESSES) as pool:
			MC_list = pool.map(partial_run, MC_list)
			pool.close()

		t_diff = time.time() - t_start
		logging.info(f'Training finished in {t_diff}s.')

	else:
		logging.info(f'Loading results from {load}')
		MC_list = save_exp.load(load)
		if task in ['fw_bw', 'fw_only']:
			MC_teacher, MC_list = [MC_list[0]], MC_list[1:]


	if compare_model is not None:
		logging.info(f'Generating comparison to {compare_model}')

		N_PROCESSES = len(MC_list) if N_MAX_PROCESSES > len(MC_list) else N_MAX_PROCESSES

		# generate comparison with BP weight updates
		partial_run = functools.partial(compare.compare_updates, model=compare_model, params=params)
		with mp.Pool(N_PROCESSES) as pool:
			MC_list = pool.map(partial_run, MC_list)
			pool.close()
		
		# create angle between WPP.T and BPP
		MC_list = compare.compare_weight_matrices(MC_list=MC_list, model=compare_model)
		# create angle between Jacobians
		if MC_list[0].mc_model == 'sacramento2018':
			MC_list = compare.compare_jacobians(MC_list=MC_list, model=compare_model)
		# create angle between BPP/BII and phi' WPP.T phi'
		MC_list = compare.compare_B_RHS(MC_list=MC_list, model=compare_model)

	logging.info(f'Plotting results')
	# plot.
	if task in ['fw_only', 'fw_bw']:
		plot_exp.plot(MC_list, MC_teacher, path=PATH)
	else:
		plot_exp.plot(MC_list, path=PATH)

	logging.info(f'Saving results')
	if task in ['fw_only', 'fw_bw']:
		# if teacher is loaded, append to list of microcircuits
		save_exp.save(MC_teacher + MC_list,path=PATH)
	else:
		save_exp.save(MC_list, path=PATH)

	t_diff = time.time() - t_start
	logging.info(f"Done. Total time: {t_diff}s")






if __name__ == '__main__':

	ARGS = parse_experiment_arguments()

	if ARGS.params is None:
		raise FileNotFoundError("params file missing")

	# get path of parameter file
	PATH = os.path.dirname(ARGS.params)
	if PATH == '':
		PATH = None

	# logging.info('Importing parameters')
	with open(ARGS.params, 'r+') as f:
		PARAMETERS = json.load(f)
	logging.info('Sucessfully imported parameters')
	logging.info(PARAMETERS)

	if "mc_model" not in PARAMETERS:
		raise ValueError("No valid microcircuit model given. Use 'model':'sacramento2018', 'ann', 'dPC' or 'errormc'")

	if ARGS.task not in ['fw_only', 'bw_only', 'fw_bw', 'fw_only_no_teacher']:
		raise ValueError("Task not recognized. Use 'fw_only', 'bw_only', 'fw_bw', 'fw_only_no_teacher'")

	if ARGS.compare not in ['BP', None]:
		raise ValueError("Model to compare to unkown. Use 'BP' or 'none'")

	main(params=PARAMETERS, task=ARGS.task, seeds=PARAMETERS['random_seed'], load=ARGS.load, cont=ARGS.cont, compare_model=ARGS.compare)



